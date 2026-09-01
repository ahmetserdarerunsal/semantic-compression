# -*- coding: utf-8 -*-
"""Semantic Compression Lab — web arayüzü.

Kullanım:
    python app.py
Sonra tarayıcıda http://127.0.0.1:7860 açılır.

Sekmeler (Karşılaştır / DCT Lab / DWT Lab / Semantik ROI / Teori) ve tasarım
ilkesi: KADEMELİ AÇIKLAMA (progressive disclosure) — birincil ekran yalnız
sonucu anlamak için gerekli olanı gösterir (görüntüler + PSNR/SSIM/BPP);
ayrıntılı metrikler ve ham teknik veriler "Gelişmiş Ayarlar" / "Detaylar" /
"Teknik Detaylar" akordeonlarına taşınmıştır. Backend (src/engines,
src/compare, src/metrics) bu yeniden düzenlemede DEĞİŞMEMİŞTİR.

GÖRÜNTÜ STATE MİMARİSİ (mega-spec "FINAL STATE/CALLBACK/TYPOGRAPHY FIX"):
Uygulamada TEK bir global aktif görüntü kaynağı vardır — `active_img` /
`active_id` / `active_meta` (bkz. build_active_image_bar()). Compare, DCT
Lab, DWT Lab ve Semantik ROI sekmelerinin HİÇBİRİNİN kendi görüntü seçicisi
YOKTUR; hepsi bu üç bileşeni PAYLAŞIR (single source of truth). `active_id`
her yeni seçim/yüklemede artan bir sayaçtır ve tüm sekmelerin "görüntü
değişti → eski türetilmiş sonuçları geçersiz kıl" zincirlerinin TEK tetik-
leyicisidir (bkz. _reveal_workspace, dwt_reset_results, sem_reset_state).

Gradio'nun varsayılan kuyruk davranışı (bu uygulamada concurrency_count
ayarlanmamıştır) bir oturumun olaylarını SIRALI işler; bu nedenle "eski
görüntünün YOLO sonucu yeni görüntünün üstüne geç kalarak yazılır" türünden
gerçek bir yarış koşulu yapısal olarak oluşamaz — asıl hata her zaman
STALE STATE'tir (görüntü değişince eski sonucun temizlenmemesi), bu dosyada
her sekme için ayrı ayrı düzeltilmiştir.

Tasarım sistemi src/ui_theme.py (renk/CSS) ve src/viz/cards.py (HTML metrik
kartları) içindedir; bu dosya yalnız düzeni ve callback'leri barındırır.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import gradio as gr
import numpy as np
import pywt
from PIL import Image

import config
import src.compare as compare
from src.pgm_io import parse_pgm
from src.engines import dct_engine, wavelet_engine
from src.engines.dct_engine import JPEG_LUMA_QTABLE
from src.engines.jpeg2000_engine import JPEG2000_AVAILABLE
from src.engines.real_jpeg_engine import REAL_JPEG_AVAILABLE
from src.engines.wavelet_engine import (decompose_for_viz,
                                        max_decomposition_level,
                                        quantize_for_viz)
from src.metrics.quality import evaluate, mse, ratio_to_bpp
from src.roi.bit_allocation import mask_to_block_importance, match_bpp, rectangle_mask
from src.semantic.importance_map import (fuse_instance_masks,
                                         get_importance_instances)
from src.ui_theme import FORCE_DARK_JS, LAB_CSS, LAB_THEME
from src.viz import cards, dct_block, subbands
from src.viz.interactive_plots import rd_interactive_figure
from src.viz.plots import draw_rect_overlay, overlay_mask
from src.viz.style import ACCENT_CYAN, ACCENT_GREEN, ACCENT_PURPLE, METHOD_REAL_JPEG

MAX_SIDE = 768  # büyük yüklemeler hız için küçültülür


# =============================================================================
# Görüntü tazelik takibi (mega-spec "AUTOMATIC FULL ANALYSIS PIPELINE" Part
# 17): otomatik analiz zinciri (DWT/Compare/Semantic) birden çok saniye
# sürebilir; bu sürede kullanıcı BAŞKA bir görüntü seçerse, hâlâ kuyrukta
# bekleyen ESKİ görüntünün işleri BAŞLAMADAN atlanmalı — "if result.image_id
# != global_active_image.image_id: discard result" burada uygulanır.
#
# Gradio'nun kuyruğu OTURUM BAŞINA sıralıdır (bkz. modül docstring'i); bu
# yüzden `request.session_hash` ile anahtarlanan bir sözlük, tek-kullanıcılı
# yerel bir araç için doğru ve yeterlidir (çoklu-oturum güvenliği de sağlar
# — bir oturumun görüntü değişimi başka bir oturumu ETKİLEMEZ). Bu, ÇALIŞAN
# bir işi kesintiye uğratmaz (Python fonksiyon çağrıları önceliksiz/
# non-preemptive'tir — bkz. NUMERICAL/STATE görevlerinin final raporu);
# yalnız HENÜZ BAŞLAMAMIŞ kuyruk işlerinin gereksiz hesaplama yapmasını ve
# yanlış sekmeye eski sonuç yazmasını önler.
# =============================================================================
_LATEST_IMAGE_ID: dict[str, int] = {}


def _mark_latest_image(session_hash: str, image_id) -> None:
    _LATEST_IMAGE_ID[session_hash] = int(image_id)


def _is_stale(session_hash: str, image_id) -> bool:
    return _LATEST_IMAGE_ID.get(session_hash, -1) != int(image_id)


# =============================================================================
# Global analiz durumu (mega-spec Part 3) — "Analyzing image... / ✓ DCT /
# ✓ DWT / ..." kompakt şerit; her satır YALNIZ o aşama GERÇEKTEN bitince
# işaretlenir (uydurulmuş/sabit bir yüzde YOKTUR). Bitince "ANALİZ HAZIR"a
# çöker.
# =============================================================================
_STATUS_STEPS = [
    ("loaded", "Görüntü yüklendi"),
    ("dct", "DCT analizi"),
    ("dwt", "DWT ayrıştırması"),
    ("compare", "JPEG/Wavelet karşılaştırması"),
    ("rd", "Rate–Distortion"),
    ("semantic_detect", "Semantik tespit"),
    ("semantic_compress", "Semantik sıkıştırma"),
]


_ALL_STATUS_KEYS = {key for key, _ in _STATUS_STEPS}


def _status_html(done: set) -> str:
    """"Bitti mi?" durumu HER ZAMAN doğrudan `done` kümesinden türetilir —
    dışarıdan verilen ayrı bir 'collapse' bayrağı KULLANILMAZ. Nedeni: DCT/
    DWT/Compare/Semantic otomatik zincirleri active_id üzerinde birbirinden
    BAĞIMSIZ kayıtlı dinleyicilerdir; Gradio'nun bunları KESİN olarak kayıt
    sırasıyla, hiç örtüşmeden çalıştırdığı garanti değildir (ör. Gradio'nun
    kuyruğu sync fonksiyonları bir thread-pool'a dağıtabilir). Bu yüzden
    "son biten zincir collapse=True yazsın" yaklaşımı gözlemlenen bir hataya
    yol açtı: geç biten ama HENÜZ tüm anahtarları görmeyen bir zincirin
    non-collapsed yazması, daha erken biten collapse=True yazmasının
    ÜZERİNE yazabiliyordu. Kümeden türetmek bu sıralama sorununa bağışıktır
    — HANGİ zincir sona ererse ersin, TÜM anahtarlar mevcutsa render HER
    ZAMAN 'ANALİZ HAZIR' olur."""
    if _ALL_STATUS_KEYS <= done:
        return '<div class="analysis-status analysis-status-ready">ANALİZ HAZIR</div>'
    rows = "".join(
        f'<div class="analysis-status-row{" done" if key in done else ""}">'
        f'<span class="analysis-status-mark">{"✓" if key in done else "…"}</span>'
        f'<span>{label}</span></div>'
        for key, label in _STATUS_STEPS
    )
    return f'<div class="analysis-status">{rows}</div>'


def _mark_done(done: set, *keys: str):
    new_done = set(done) | set(keys)
    return new_done, _status_html(new_done)


def _status_reset():
    """Aktif görüntü değiştiğinde çalışan İLK olay (mega-spec Part 21 STEP
    1) — durum şeridini 'Görüntü yüklendi' dışında hepsi bekliyor durumuna
    döndürür. Bu, bu dosyadaki active_id.change() zincirlerinin EN ÖNCE
    kayıt edileni olmalıdır ki sonraki tüm aşama-tamamlama işaretleri BU
    sıfırlanmış küme üzerine eklensin."""
    done = {"loaded"}
    return done, gr.update(value=_status_html(done), visible=True)


def _prepare(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]).astype(np.float64)


def _reduce_if_true_grayscale(image: np.ndarray) -> np.ndarray:
    """active_img HER ZAMAN RGB'dir (bkz. modül docstring'i — Semantic ROI/
    YOLO için gerekli), gerçek gri kaynaklarda (klasik PNG'ler, gri JPEG/
    PGM yüklemeleri) bu yalnız R=G=B üçlemesidir. Karşılaştırma motorlarına
    (dct_engine/wavelet_engine/jpeg2000_engine/real_jpeg_engine) bu üçleme
    3 kanal olarak geçirilirse HEPSİ zaten var olan ndim==2 (gerçek tek
    bileşenli) dalları yerine YCbCr/renkli yola giriyor — bunun GERÇEK,
    ölçülmüş iki sonucu var: (1) real_jpeg_engine sabit (128) ama SIFIR
    olmayan bayt maliyetli kroma düzlemleri kodluyor (ölçüldü: aynı
    quality'de %2-8 fazla bayt — bu da hedef bpp'ye bisection'la
    tutturulurken luma'dan çalınan bitler yüzünden PSNR'ı düşürüyor), (2)
    calculate_metrics'in "orijinal boyut" tabanı (raw_bits) H*W*3*8 bite
    (768 KiB, 512x512 için) çıkıyor — GERÇEK 256 KiB'lik gri taban yerine
    (mega-spec 'BPP/PSNR GÜVENİLİRLİK DENETİMİ' Part 8/9). Gerçek renkli
    görüntülerde (R,G,B GERÇEKTEN farklı) DOKUNULMAZ, tam YCbCr renkli
    kodlama uygulanmaya devam eder."""
    if image.ndim != 3:
        return image
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if np.array_equal(r, g) and np.array_equal(g, b):
        return r
    return image


# =============================================================================
# Kategorili örnek görüntü kaynağı (yalnız TEK global seçici tarafından
# kullanılır — bkz. build_active_image_bar())
# =============================================================================
_MAX_GALLERY_ITEMS = 6  # 4-6 görünür örnek/kategori


def _category_dir(label: str) -> str:
    return next(k for k, l in config.UI_IMAGE_CATEGORIES if l == label)


def _category_images(label: str) -> list[str]:
    folder = config.DATA_DIR / _category_dir(label)
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.glob("*.png"))[:_MAX_GALLERY_ITEMS]


def _on_category_change(label: str):
    return gr.update(value=_category_images(label))


def build_active_image_bar():
    """Uygulama genelinde TEK canonical görüntü seçici + aktif görüntü
    çubuğu (mega-spec: "AYRI AYRI görüntü sahibi olmayacak... GLOBAL_ACTIVE_
    IMAGE"). Tabların ÜSTÜNE, `gr.Tabs()`'tan ÖNCE yerleştirilir; Compare/
    DCT/DWT/Semantic bu üç bileşeni PAYLAŞIR:

      active_img  — gr.Image(visible=False): canonical numpy dizisi (RGB,
                    zaten _prepare() ile boyutlandırılmış)
      active_id   — gr.Number(visible=False): her yeni seçimde/yüklemede
                    artan tam sayı sayaç; tüm "görüntü değişti" zincirlerinin
                    TEK tetikleyicisi (bkz. modül docstring'i)
      active_meta — gr.State(dict): filename/category/width/height/source_type

    Döner: (active_img, active_id, active_meta, home_btn) — home_btn,
    Karşılaştır sekmesinin toolbar/workspace görünürlüğünü de kapatmak
    üzere build_ui() içinde AYRICA .click() ile bağlanır (bkz. orada)."""
    default_label = config.UI_IMAGE_CATEGORIES[0][1]

    # format="png" (varsayılan "webp" DEĞİL): "webp" LOSSY'DİR — Gradio'nun
    # postprocess/preprocess döngüsünde active_img'in DEĞERİ her aktarımda
    # webp'ye kodlanıp geri çözülüyordu, bu da DCT/DWT/Compare'e giden
    # "orijinal" piksellerin GERÇEK yüklenen/seçilen değerlerle birebir AYNI
    # OLMAMASINA yol açıyordu (doğrulandı: 8-bit kanalda maks fark 20/255 —
    # ihmal edilebilir değil, PSNR/BPP hesaplarını gerçekten etkiliyor).
    # PNG kayıpsızdır; bu satır TÜM görüntü kaynaklarını (örnek/yükleme/PGM)
    # düzeltir.
    active_img = gr.Image(visible=False, type="numpy", format="png")
    active_id = gr.Number(visible=False, value=0, precision=0)
    active_meta = gr.State({})

    with gr.Column(visible=True) as empty_group:
        gr.HTML(cards.hero_empty_html(
            "Bir Görüntü Seçin",
            "Seçtiğiniz görüntü; Karşılaştır, DCT Lab, DWT Lab ve Semantik "
            "ROI sekmelerinin TÜMÜNDE aynı anda kullanılır.",
        ))
        with gr.Row(elem_classes=["hero-buttons"]):
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=0, min_width=420):
                with gr.Row():
                    select_btn = gr.Button("Örnek Seç", size="sm", variant="secondary")
                    upload_btn = gr.Button("Görüntü Yükle", size="sm", variant="secondary")
            with gr.Column(scale=1):
                pass

    with gr.Row(visible=False, elem_classes=["active-image-bar"]) as bar_row:
        bar_html = gr.HTML()
        change_btn = gr.Button("Görüntüyü değiştir", size="sm", variant="secondary")
        home_btn = gr.Button("⌂ Ana Sayfa", size="sm", variant="secondary")

    with gr.Accordion("Kategori ve örnekler", open=False, visible=False) as picker_acc:
        cat_radio = gr.Radio([l for _, l in config.UI_IMAGE_CATEGORIES], value=default_label,
                             label="Kategori")
        gallery = gr.Gallery(value=_category_images(default_label), columns=_MAX_GALLERY_ITEMS, rows=1,
                             height=180, show_label=False, object_fit="contain",
                             allow_preview=False)
    with gr.Group(visible=False) as upload_group:
        # gr.File (gr.Image DEĞİL): gr.Image(type="filepath") bu Gradio
        # sürümünde dosyayı ASLA ham geçirmiyor — preprocess() HER ZAMAN
        # PIL ile açıp RGB'ye çevirip yeniden kaydediyor (bkz.
        # gradio/components/image.py format_image → save_pil_to_cache,
        # koşulsuz), yani "yüklediğimiz dosyanın GERÇEK boyutu" (Path.stat)
        # aslında bu yeniden-kodlanmış kopyanın boyutuydu — doğrulandı:
        # gerçek bir .jpg testinde 77KB'lık dosya için "38KB", gri bir
        # PNG'de 139KB için "238KB" gösteriyordu. gr.File hiç yeniden
        # kodlama yapmadığı için (aynı PGM yükleyicisinde olduğu gibi)
        # boyut artık gerçekten doğru.
        upload_img = gr.File(
            file_types=["image"], type="filepath", label="Görüntünüzü yükleyin")
        # gr.Image'ın tarayıcı tarafı upload filtresi SABİT "image/*"tir
        # (Gradio 4.44'te değiştirilemez) ve .pgm çoğu tarayıcıda
        # application/octet-stream olarak raporlandığından o widget'ta
        # seçilemez/reddedilir. gr.File'ın file_types'ı ise UZANTI bazlı
        # eşleştiği için (bkz. Upload.svelte is_valid_mimetype) güvenilir
        # bir whitelist sağlar — bu yüzden PGM için AYRI, küçük bir
        # yükleyici eklendi.
        upload_pgm = gr.File(
            file_types=[".pgm", ".pnm", "image/x-portable-graymap", "application/octet-stream"],
            type="filepath",
            label="veya .pgm (Netpbm grayscale, P2/P5) yükleyin",
        )

    select_btn.click(
        lambda: (gr.update(visible=True, open=True), gr.update(visible=False)),
        None, [picker_acc, upload_group])
    upload_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False, open=False)),
        None, [upload_group, picker_acc])
    change_btn.click(
        lambda: (gr.update(visible=True, open=True), gr.update(visible=False)),
        None, [picker_acc, upload_group])
    # ANA SAYFA: aktif görüntüyü SİLMEZ (kullanıcı yeni bir görüntü
    # seçince mevcut active_id-tetiklemeli reset zinciri zaten her şeyi
    # doğru şekilde temizler) — yalnız başlangıç ekranına (hero + örnek/
    # yükleme seçici) geri döner. Karşılaştır sekmesinin toolbar/workspace
    # görünürlüğü, o bileşenler bu fonksiyonun kapsamı DIŞINDA
    # tanımlandığı için build_ui() içinde AYRICA bu AYNI butona bağlanır.
    home_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False),
                 gr.update(visible=False, open=False), gr.update(visible=False)),
        None, [empty_group, bar_row, picker_acc, upload_group])
    cat_radio.change(_on_category_change, cat_radio, gallery)

    def _select(cat_label, current_id, evt: gr.SelectData, request: gr.Request):
        path = evt.value["image"]["path"]
        file_size_bytes = Path(path).stat().st_size  # GERÇEK disk dosyası boyutu
        raw = np.array(Image.open(path).convert("RGB"))
        orig_h, orig_w = raw.shape[:2]  # _prepare'DAN ÖNCE — resize gizlenmesin (audit Part 3)
        img = _prepare(raw)
        name = Path(path).stem
        h, w = img.shape[:2]
        meta = {"filename": name, "category": cat_label, "width": w, "height": h,
                "source_type": "sample", "file_size_bytes": file_size_bytes}
        new_id = int(current_id) + 1
        _mark_latest_image(request.session_hash, new_id)  # otomatik analiz zinciri BUNU okur
        return (
            img, new_id, meta,
            cards.active_image_bar_html(name, cat_label, w, h, orig_w, orig_h, file_size_bytes),
            gr.update(visible=True), gr.update(visible=False, open=False),
            gr.update(visible=False),
        )

    def _upload(path, current_id, request: gr.Request):
        if path is None:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update())
        # upload_img artık gr.File (bkz. yukarıdaki yorum) — path GERÇEKTEN
        # yüklenen dosyanın kendisi, Gradio tarafından yeniden kodlanmış bir
        # kopya değil; bu yüzden Path.stat buradan alınan boyut GERÇEK.
        file_size_bytes = Path(path).stat().st_size
        raw = np.array(Image.open(path).convert("RGB"))
        orig_h, orig_w = raw.shape[:2]  # _prepare'DAN ÖNCE — resize gizlenmesin (audit Part 3)
        img = _prepare(raw)
        h, w = img.shape[:2]
        meta = {"filename": "Yüklenen görüntü", "category": None, "width": w, "height": h,
                "source_type": "upload", "file_size_bytes": file_size_bytes}
        new_id = int(current_id) + 1
        _mark_latest_image(request.session_hash, new_id)
        return (
            img, new_id, meta,
            cards.active_image_bar_html("Yüklenen görüntü", None, w, h, orig_w, orig_h, file_size_bytes),
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False),
        )

    def _upload_pgm(path, current_id, request: gr.Request):
        if path is None:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update())
        file_size_bytes = Path(path).stat().st_size  # GERÇEK yüklenen dosya boyutu
        try:
            gray = parse_pgm(path)  # P2/P5, header'dan width/height/maxval — bkz. src/pgm_io
        except ValueError as exc:
            raise gr.Error(f"PGM ayrıştırma hatası: {exc}") from exc
        orig_h, orig_w = gray.shape
        # active_img (type="numpy") her yerde RGB bekleniyor (Semantic ROI/YOLO
        # dahil) — R=G=B üçlemesi KAYIPSIZDIR: _to_gray()'in ağırlıkları
        # (0.299+0.587+0.114=1.0) R=G=B için matematiksel özdeşliktir, yani
        # DCT/DWT'ye ulaşan değerler PGM dosyasındaki GERÇEK piksellerle
        # birebir aynı kalır; klasik gri PNG örnekleri de zaten aynı yoldan
        # geçiyor (bkz. Image.open(...).convert("RGB") yukarıda).
        rgb = np.stack([gray, gray, gray], axis=-1)
        img = _prepare(rgb)
        h, w = img.shape[:2]
        meta = {"filename": "Yüklenen görüntü (PGM)", "category": None, "width": w, "height": h,
                "source_type": "upload", "file_size_bytes": file_size_bytes}
        new_id = int(current_id) + 1
        _mark_latest_image(request.session_hash, new_id)
        return (
            img, new_id, meta,
            cards.active_image_bar_html("Yüklenen görüntü (PGM)", None, w, h, orig_w, orig_h, file_size_bytes),
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False),
        )

    gallery.select(
        _select, [cat_radio, active_id],
        [active_img, active_id, active_meta, bar_html, bar_row, picker_acc, empty_group])
    upload_img.change(
        _upload, [upload_img, active_id],
        [active_img, active_id, active_meta, bar_html, bar_row, upload_group, empty_group])
    upload_pgm.change(
        _upload_pgm, [upload_pgm, active_id],
        [active_img, active_id, active_meta, bar_html, bar_row, upload_group, empty_group])

    return active_img, active_id, active_meta, home_btn


# =============================================================================
# SEKME 1 — KARŞILAŞTIR (JPEG/DCT vs Wavelet/DWT, aynı hedef bpp)
# =============================================================================
_RD_TARGETS = [0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.8, 2.5]


_TARGET_MODE_BPP = "Hedef bpp"
_TARGET_MODE_RATIO = "Hedef Sıkıştırma Oranı"


def _target_to_bpp(mode: str, value: float, channels: int = 3) -> float:
    """channels: 'Hedef Sıkıştırma Oranı' modunda oranın hangi ham tabana
    (H*W*channels*8 bit) göre bpp'ye çevrileceği — GERÇEK gri görüntülerde
    (bkz. _reduce_if_true_grayscale) 1 olmalı, yoksa 'oran 8' seçildiğinde
    hedef bpp 24/8=3.0 yerine yanlışlıkla 8/8=1.0'a karşılık geliyormuş
    gibi hesaplanır (raw_bits'in kendi tanımıyla TUTARSIZ olurdu)."""
    if mode == _TARGET_MODE_BPP:
        return float(value)
    return ratio_to_bpp(float(value), channels=channels)


def target_mode_ui(mode: str):
    """Hedef modu değişince kontrolün etiketi/aralığı/varsayılanı da
    değişir (mega-spec 'FINAL PRE-PRESENTATION QA' Part 1) — önceden
    slider HER ZAMAN 'Hedef Oran' yazıyordu, mod 'Hedef bpp' olsa bile;
    '0.5' değerinin bpp mi yoksa sıkıştırma oranı mı olduğu belirsizdi.
    Sıkıştırma oranı aralığı (2–48) bpp aralığından (0.1–4.0) BİLEREK
    FARKLIDIR — 24/bpp formülüyle (bkz. ratio_to_bpp) eşlenen gerçekçi
    oran değerleridir, aynı sayısal aralık iki modda da anlamsız olurdu."""
    if mode == _TARGET_MODE_BPP:
        return gr.update(label="Hedef bpp", minimum=0.1, maximum=4.0, step=0.05, value=0.5)
    return gr.update(label="Hedef Sıkıştırma Oranı", minimum=2.0, maximum=48.0, step=1.0, value=8.0)


def toggle_real_jpeg_column(include: bool):
    return gr.update(visible=bool(include))


def run_main_comparison(
    image, target_mode, target_value, dct_block_size,
    include_real_jpeg, progress=gr.Progress(),
):
    """Ana 'Karşılaştır' ekranı — mega-spec 'JPEG vs JPEG2000 GERÇEK CODEC
    KARŞILAŞTIRMASI': ikinci sütun artık özel entropi-tahmini DWT motoru
    (wavelet_dd/wavelet_level ile seçilen bior4.4 vb.) DEĞİL, gerçek
    OpenJPEG/JPEG2000 (`compare.run_jpeg2000`) — is_real_codec HER ZAMAN
    True, bpp GERÇEK kodlanmış bayt sayısından. Özel wavelet motoru
    (`compare.run_wavelet`, Shannon order-0 tahmini) SİLİNMEDİ — yalnız
    DWT Lab (aşağıda, ayrı sekme) ve Semantik ROI onu kullanmaya devam
    ediyor; bu fonksiyon artık ona hiç dokunmuyor."""
    if image is None:
        raise gr.Error("Önce üstteki AKTİF GÖRÜNTÜ çubuğundan bir görüntü seçin.")
    if not JPEG2000_AVAILABLE:
        raise gr.Error(
            "Bu ortamda gerçek bir JPEG2000 (OpenJPEG) kurulumu yok; "
            "sahte/simüle bir sonuç gösterilemez."
        )
    image = _prepare(image)
    image = _reduce_if_true_grayscale(image)
    channels = 1 if image.ndim == 2 else image.shape[2]
    target_bpp = _target_to_bpp(target_mode, target_value, channels)
    dct_block_size = int(dct_block_size)

    progress(0.10, desc="JPEG/DCT hedef bpp'ye eşleniyor (bisection)…")
    dct_res = compare.run_jpeg_primary(image, target_bpp, dct_block_size)

    progress(0.45, desc="JPEG2000 (gerçek OpenJPEG kodek) kodlanıyor…")
    wav_res = compare.run_jpeg2000(image, target_bpp)

    def _current_point(res: compare.EngineResult) -> dict:
        return {**res.metrics, "extra": res.extra, "target_bpp": target_bpp}

    target_points = [
        ("JPEG / DCT", ACCENT_CYAN, dct_res.metrics["bpp"]),
        ("JPEG2000 / DWT", ACCENT_PURPLE, wav_res.metrics["bpp"]),
    ]
    current = {
        "JPEG / DCT": _current_point(dct_res),
        "JPEG2000 / DWT": _current_point(wav_res),
    }

    # Gerçek JPEG (libjpeg) çapraz-doğrulaması (mega-spec "JPEG/DCT BPP VE
    # RATE-DISTORTION DENETİMİ" Part 2/3) — dct_engine'in Shannon-entropi
    # TAHMİNİNİN aksine, GERÇEK kodlanmış bayt sayısından hesaplanan bpp/
    # boyut. block_size==8'de dct_res ile AYNI sonucu tekrar gösterir
    # (çapraz-doğrulama amaçlı, isteğe bağlı).
    real_jpeg_img = None
    real_jpeg_compact, real_jpeg_full = "", ""
    if include_real_jpeg and REAL_JPEG_AVAILABLE:
        progress(0.75, desc="Gerçek JPEG (libjpeg) kodlanıyor…")
        rj_res = compare.run_real_jpeg(image, target_bpp)
        real_jpeg_img = rj_res.recon
        real_jpeg_compact = cards.compact_metric_card(
            "JPEG (gerçek)", METHOD_REAL_JPEG, rj_res.metrics, param_label=rj_res.param_label)
        real_jpeg_full = cards.metric_card(
            "JPEG (gerçek kodek — libjpeg)", METHOD_REAL_JPEG, rj_res.metrics,
            size_badge="GERÇEK BOYUT", param_label=rj_res.param_label)
        target_points.append(("JPEG (gerçek)", METHOD_REAL_JPEG, rj_res.metrics["bpp"]))
        current["JPEG (gerçek)"] = _current_point(rj_res)
    elif include_real_jpeg and not REAL_JPEG_AVAILABLE:
        real_jpeg_full = cards.empty_state_html(
            "Gerçek JPEG kodeği bulunamadı",
            "Bu ortamda çalışan bir libjpeg (OpenCV) kurulumu yok; sahte/simüle bir sonuç gösterilmez.",
        )

    progress(0.85, desc="Rate-Distortion taraması (gerçek ölçümler)…")
    targets = sorted(set(_RD_TARGETS + [round(target_bpp, 3)]))
    curves = compare.rd_sweep(image, targets, dct_block_size)
    fig = rd_interactive_figure(curves, target_bpp, current)

    rate_fairness = cards.rate_fairness_html(target_bpp, dct_res.metrics["bpp"], wav_res.metrics["bpp"])
    quality_strip = cards.quality_strip_html(dct_res.metrics, wav_res.metrics, "JPEG2000 / DWT")
    target_full = cards.target_summary_html(target_bpp, target_points)
    # GERÇEK/TAHMİNİ rozeti artık EngineResult.is_real_codec'ten TÜRETİLİR
    # (mega-spec "FINAL FEATURE PASS" Part 18/26: tek kaynak) — motor
    # gerçekten bir bitstream kodek mi (JPEG block_size=8'de, JPEG2000 HER
    # ZAMAN) yoksa entropi tahmini mi (yalnız JPEG'in standart-dışı blok
    # boyutu fallback'i) ÜRETTİĞİ artık compare.py'deki TEK bayraktan okunur.
    dct_badge = "GERÇEK BOYUT" if dct_res.is_real_codec else "ENTROPİ TAHMİNİ"
    wav_badge = "GERÇEK BOYUT" if wav_res.is_real_codec else "ENTROPİ TAHMİNİ"
    dct_full = cards.metric_card("JPEG / DCT", ACCENT_CYAN, dct_res.metrics,
                                 size_badge=dct_badge, param_label=dct_res.param_label)
    wav_full = cards.metric_card("JPEG2000 / DWT", ACCENT_PURPLE, wav_res.metrics,
                                 compare_to=dct_res.metrics, size_badge=wav_badge,
                                 param_label=wav_res.param_label)
    # SIKIŞTIRMA ÖZETİ — kompakt KPI satırı + ORİJİNAL → SIKIŞTIRILMIŞ
    # boyut oku (mega-spec "FINAL FEATURE PASS" Part 2/3/4). AYNI
    # dct_res.metrics/wav_res.metrics'ten üretilir — dct_full/wav_full ile
    # birebir aynı sayılar (Part 18: source-of-truth tutarlılığı).
    dct_summary = cards.compression_summary_card_html("JPEG / DCT", ACCENT_CYAN, dct_res.metrics, dct_badge)
    wav_summary = cards.compression_summary_card_html("JPEG2000 / DWT", ACCENT_PURPLE, wav_res.metrics, wav_badge)

    return (
        dct_res.recon, wav_res.recon,
        rate_fairness, quality_strip, fig,
        target_full, dct_full, wav_full,
        dct_summary, wav_summary,
        real_jpeg_img, real_jpeg_compact, real_jpeg_full,
    )


_BA_METHOD_JPEG = "JPEG / DCT"
_BA_METHOD_WAVELET = "JPEG2000 / DWT"


def render_before_after(orig, dct_recon, wav_recon, method):
    """ÖNCE/SONRA kaydırıcısını doldurur (mega-spec "FINAL FEATURE PASS"
    Part 10-16). YENİ bir hesaplama YAPMAZ (Part 12/14) — orig/dct_recon/
    wav_recon zaten out_orig/out_dct/out_wav bileşenlerinin GÜNCEL
    değerleridir (run_main_comparison'ın ürettiği AYNI rekonstrüksiyon
    dizileri); bu fonksiyon yalnız görüntüleri seçip cards.
    before_after_slider_html'e iletir. Compare parametreleri değiştiğinde
    (hedef bpp, blok boyutu, dalgacık, ...) out_dct/out_wav zaten yeniden
    hesaplandığından, bu fonksiyonun aynı `.then()` zincirinde tekrar
    çağrılması kaydırıcıyı GÜNCEL sonuçla tazeler — slider'ın KENDİSİ
    hareket ederken hiçbir backend çağrısı olmaz."""
    if orig is None:
        return cards.empty_state_html(
            "ÖNCE/SONRA İÇİN SONUÇ YOK", "Önce KARŞILAŞTIR'a basın.")
    recon = dct_recon if method == _BA_METHOD_JPEG else wav_recon
    if recon is None:
        return cards.empty_state_html(
            "ÖNCE/SONRA İÇİN SONUÇ YOK", "Önce KARŞILAŞTIR'a basın.")
    accent = ACCENT_CYAN if method == _BA_METHOD_JPEG else ACCENT_PURPLE
    return cards.before_after_slider_html(orig, recon, "ORİJİNAL", method, accent)


def toggle_cmp_view_mode(mode: str):
    """[ Yan Yana ] [ Önce / Sonra ] — mevcut üçlü yan-yana görünüm
    KALDIRILMAZ (mega-spec Part 16), yalnız bir ALTERNATİFİ eklenir."""
    is_side = mode == "Yan Yana"
    return gr.update(visible=is_side), gr.update(visible=not is_side)


def _reveal_workspace(img, image_id):
    """Aktif görüntü DEĞİŞTİĞİNDE (active_id.change) çalışır — boş durumu
    gizler, deney çubuğunu/iş alanını gösterir, ORİJİNAL önizlemesini
    doldurur VE önceki görüntüye ait TÜM Compare sonuçlarını temizler
    (mega-spec Part 18: "Compare RESET... eski görüntünün karşılaştırma
    sonucu görünmemeli"). Sonuçlar (results_group) yalnız KARŞILAŞTIR
    sonrası tekrar görünür hale gelir — bu iki AYRI aşamadır."""
    return (
        img,                         # out_orig önizleme
        gr.update(visible=True),     # toolbar
        gr.update(visible=True),     # workspace
        gr.update(visible=False),    # results_group (yeni görüntüde eski sonuç saklanmaz)
        None, None,                  # out_dct, out_wav
        "", "",                      # rate_fairness, quality_strip
        None,                        # cmp_plot
        "", "", "",                  # target_full, dct_full, wav_full
        "", "",                      # dct_summary, wav_summary
        "",                          # cmp_ba_slider (eski görüntünün Önce/Sonra'sı KALMAZ)
        gr.update(value="Yan Yana"), # cmp_view_mode (görüntü değişince Önce/Sonra modundan çık)
        None, "", "",                # out_real_jpeg, real_jpeg_compact, real_jpeg_full
    )


def compare_auto_analysis(
    image, image_id, target_mode, target_value, dct_block_size,
    include_real_jpeg, request: gr.Request, progress=gr.Progress(),
):
    """Aktif görüntü seçilir seçilmez Compare sekmesini VARSAYILAN hedefte
    (0.5 bpp) otomatik doldurur (mega-spec "AUTOMATIC FULL ANALYSIS
    PIPELINE" Part 6/8) — kullanıcının ilk sonuç için KARŞILAŞTIR'a basmasına
    gerek YOKTUR. run_main_comparison'ın KENDİSİNİ çağırır; algoritma
    tekrarlanmaz (Part 29). KARŞILAŞTIR butonu, kullanıcı hedefi/parametreleri
    DEĞİŞTİRDİĞİNDE elle yeniden çalıştırmak için kalır (Part 30)."""
    if image is None or _is_stale(request.session_hash, image_id):
        return (gr.update(),) * 13  # out_dct..real_jpeg_full (13 çıktı) — bkz. cmp_btn.click
    return run_main_comparison(image, target_mode, target_value, dct_block_size,
                               include_real_jpeg, progress)


# =============================================================================
# SEKME 2 — DCT LAB
# =============================================================================
def dct_grid_shape(image, block_size):
    if image is None:
        return gr.update(maximum=1), gr.update(maximum=1)
    image = _prepare(image)
    gray = _to_gray(image)
    h, w = gray.shape
    nh = -(-h // int(block_size))
    nw = -(-w // int(block_size))
    return gr.update(maximum=max(nh - 1, 0), value=min(nh // 2, nh - 1)), \
           gr.update(maximum=max(nw - 1, 0), value=min(nw // 2, nw - 1))


def select_block_from_click(block_size, evt: gr.SelectData):
    """Kaynak görüntüye TIKLAYARAK blok seçimi. evt.index, Gradio tarafından
    her zaman görüntünün GERÇEK piksel uzayında raporlanır (CSS ile
    küçültülmüş gösterimden bağımsız — doğrulanmıştır), bu yüzden ek
    ölçekleme gerekmez. Satır/sütun slider'larını günceller; onların
    .change() olayı update_dct_analysis'i TAM olarak yeniden tetikler
    (iki yönlü bağlama: tık→slider→tam analiz)."""
    x, y = evt.index
    block_size = int(block_size)
    return y // block_size, x // block_size


def update_dct_analysis(image, block_size, row, col, quality):
    """TEK canonical DCT hesaplama fonksiyonu (mega-spec "FINAL STATE/
    CALLBACK FIX" Part 10) — kaynak görüntü üzerindeki cyan overlay, seçili-
    blok satırı, TÜM pipeline görselleştirmesi (piksel/seviye-kaydırma/DCT/
    kuantalama/rekonstrüksiyon), zigzag taraması ve ölçüm şeridi hep BU TEK
    fonksiyondan üretilir.

    Kritik düzeltme (Part 1/9): önceden Satır/Sütun/Blok Boyutu değiştiğinde
    yalnız kaynak görüntüdeki cyan overlay tazeleniyordu; sağdaki DCT
    matrisleri/zigzag/rekonstrüksiyon/PSNR YALNIZ "BLOĞU İNCELE" butonuna
    basılınca güncelleniyordu — yani slider'la başka bir blok seçildiğinde
    sağ panel ESKİ bloğun sonucunu göstermeye devam ediyordu (stale result).
    Artık Satır/Sütun/Blok Boyutu/Quality'den HERHANGİ biri değiştiğinde (ve
    ayrıca "BLOĞU İNCELE" butonunda) bu TEK fonksiyon TÜM çıktıları birlikte
    yeniden hesaplar; eski sonucun ekranda kalması yapısal olarak mümkün
    değildir. Backend hesaplaması (inspect_block/dct_block.py) DEĞİŞMEMİŞTİR
    — yalnız hangi callback'lerin onu çağırdığı değişmiştir."""
    if image is None:
        empty = cards.empty_state_html(
            "DCT ANALİZİ İÇİN GÖRÜNTÜ SEÇİN",
            "Üstteki AKTİF GÖRÜNTÜ çubuğundan bir örnek seçin veya yükleyin.")
        return None, "", None, None, empty, ""

    image = _prepare(image)
    h, w = image.shape[:2]
    block_size, row, col = int(block_size), int(row), int(col)

    overlay_img = dct_block.draw_block_overlay(image, block_size, row, col)
    region = dct_block.block_pixel_region((h, w), block_size, row, col)
    line = cards.selected_block_line_html(block_size, row, col, region)

    gray = _to_gray(image)
    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    result = dct_block.inspect_block(gray, block_size, row, col, float(quality), base_table)
    fig = dct_block.block_pipeline_figure(result)
    zigzag_fig = dct_block.zigzag_figure(result["quantized"])

    summary = cards.measurement_strip_html([
        ("BLOK", f"{block_size}×{block_size}"),
        ("DC", f"{result['dc']:.1f}"),
        ("SIFIR-OLMAYAN", f"{result['n_nonzero']} / {result['n_total']}"),
        ("BLOK PSNR", cards.fmt_psnr(result["block_psnr"])),
        ("QUALITY", f"{int(quality)}"),
    ])

    note = ("8×8 dışındaki blok boyutları standart JPEG'in bir parçası değildir; "
           "bu projede eğitim amaçlı bir DCT-boyutu deneyidir — kuantalama "
           "tablosu 8×8 standart JPEG tablosundan enterpolasyonla türetilir."
           if block_size != 8 else
           "8×8: baseline JPEG standardının kullandığı gerçek blok boyutu ve "
           "gerçek luminance kuantalama tablosu.")
    info = (
        f"Sol-üst = düşük uzamsal frekans, sağ-alt yönü = artan frekans; DC katsayısı "
        f"bloğun ortalama parlaklığını temsil eder. Kuantalama sonrası küçük/önemsiz "
        f"yüksek frekans katsayıları sıfırlanır (yukarıdaki SIFIR-OLMAYAN sayısı). "
        f"Blok PSNR, yalnız bu tek bloğun orijinaline göre bozulmasını ölçer. *{note}* "
        f"**Not (renk kanalı):** Bu blok analizi yalnız **luminance (gri ton)** "
        f"kanalı üzerinde çalışır — Karşılaştır sekmesindeki gerçek DCT sıkıştırması "
        f"RGB'yi YCbCr'ye çevirip her kanalı ayrı ayrı işler; buradaki tek-blok "
        f"görselleştirme pedagojik netlik için kasıtlı olarak gri tonlamaya "
        f"basitleştirilmiştir, bu yüzden buradaki PSNR/BPP, Karşılaştır sekmesindeki "
        f"RGB-tüm-görüntü metrikleriyle DOĞRUDAN karşılaştırılamaz."
    )
    return overlay_img, line, fig, zigzag_fig, summary, info


# =============================================================================
# SEKME 3 — DWT LAB
# =============================================================================
def dwt_level_range(image, wavelet):
    if image is None:
        return gr.update(maximum=config.WAVELET_LEVEL_UI_MAX)
    image = _prepare(image)
    cap = min(max_decomposition_level(image.shape[:2], wavelet), config.WAVELET_LEVEL_UI_MAX)
    return gr.update(maximum=cap, value=min(config.WAVELET_LEVELS, cap))


def dwt_reset_results():
    """Aktif görüntü değiştiğinde DWT'nin GÖRÜNTÜYE BAĞLI çıktılarını
    temizler (mega-spec Part 17: "DWT old result yeni image altında
    kalmıyor") — piramit mozaiği, ağaç, LL/LH/HL/HH önizlemeleri,
    seçili-bant önbelleği/durumu, rekonstrüksiyon, istatistik tablosu,
    doğrulama/seyreklik kartları.

    Filtre bankası bilgisi (dwt_filt_flow/dwt_filt/dwt_filt_text) yalnız
    SEÇİLİ DALGACIK AİLESİNE bağlıdır, görüntüden bağımsızdır — bilinçli
    olarak DOKUNULMAZ (Part 17: "current wavelet selection korunabilir").
    Ayrıştırma seviyesi zaten dwt_level_range ile ayrı olarak yeni görüntüye
    clamp edilir."""
    empty = cards.empty_state_html(
        "YENİ GÖRÜNTÜ SEÇİLDİ",
        "AYRIŞTIR'a basarak bu görüntüyü analiz edin.")
    empty_tree = np.zeros((10, 10, 3), dtype=np.uint8)
    # Sıra, _dwt_result_outputs listesiyle (app.py DWT Lab UI bölümü)
    # BİREBİR eşleşmelidir: summary, mosaic, tree, tree_bounds, LL, LH, HL,
    # HH, band_details, selected_band, histogram, recon, diff, energy,
    # stats, validation, sparsity, levels_data, max_level, mosaic_info,
    # recon_info, gray_state, recon_state.
    return (
        empty, None, empty_tree, [],
        gr.update(value=None, label="LL"), gr.update(value=None, label="LH"),
        gr.update(value=None, label="HL"), gr.update(value=None, label="HH"),
        "", "LL", None,
        None, None, None,
        [], "", "",
        {}, 1,
        "", "", None, None,
    )


def dwt_auto_analysis(image, image_id, wavelet, levels, quant_step, mode,
                      rate_mode, target_bpp, request: gr.Request):
    """Aktif görüntü seçilir seçilmez DWT Lab'ı VARSAYılAN parametrelerle
    (mevcut dalgacık/seviye/kuantalama adımı/sınır modu/hız modu) otomatik
    doldurur (mega-spec Part 11) — AYRIŞTIR'a basmaya gerek YOKTUR.
    run_dwt_explorer'ın KENDİSİNİ çağırır (Part 29: algoritma tekrarlanmaz).
    rate_mode/target_bpp de İLETİLİR — kullanıcı "Hedef BPP" modundayken
    görüntü değiştirirse yeni görüntü SESSİZCE "Manuel Δ"ya dönmez (UI'daki
    radyo düğmesiyle GERÇEK hesaplama TUTARLI kalır). Seviye, engine
    içinde zaten görüntüye göre clamp edilir (max_decomposition_level)."""
    if image is None or _is_stale(request.session_hash, image_id):
        return (gr.update(),) * 27  # bkz. dwt_btn.click çıktı listesi
    return run_dwt_explorer(image, wavelet, levels, quant_step, 1, mode, rate_mode, target_bpp)


def dwt_select_level(levels_data: dict, level, selected_band: str):
    """'İncelenecek seviye' değiştiğinde (slider İLE VEYA ağaç tıklamasıyla)
    çalışır. DWT'yi YENİDEN HESAPLAMAZ (mega-spec "DWT LAB FIX" Part 19) —
    yalnız run_dwt_explorer'ın DOLDURDUĞU `levels_data` önbelleğinden
    (bkz. subbands.decompose_levels) GERÇEK katsayı dizilerini okur ve
    seçili seviyeyi ağaçta vurgular.

    Bu, önceki kritik bug'ın (Part 4: pywt'nin coeffs listesi TERSTİR;
    `coeffs[level_index]` doğrudan kullanmak yanlış seviyeyi gösteriyordu)
    düzeltilmesinin merkezi noktasıdır — artık level→dizi eşlemesi TEK
    yerde (decompose_levels) ve doğru yapılır; burası yalnız o eşlemeden
    OKUR."""
    level = int(level)
    if not levels_data or level not in levels_data:
        empty = cards.empty_state_html("Seviye verisi yok", "Önce görüntü seçin veya AYRIŞTIR'a basın.")
        tree_img, bounds = subbands.dyadic_tree_image(max(level, 1), None, selected_band)
        return (
            tree_img, bounds,
            gr.update(value=None, label="LL"), gr.update(value=None, label="LH"),
            gr.update(value=None, label="HL"), gr.update(value=None, label="HH"),
            empty, selected_band, None,
        )
    bands = levels_data[level]
    max_level = max(levels_data.keys())
    band = selected_band if selected_band in bands else "LL"
    # Ağaçta YALNIZ gerçekten seçili TEK düğüm (level+band) vurgulanır —
    # tüm satır değil (mega-spec Part 7); sağdaki subband kartlarına
    # tıklamak da (bkz. _dwt_band_click) BURAYA aynı band ile geri döner,
    # bu yüzden ağaç ve sağ panel HER ZAMAN senkron kalır (Part 6).
    tree_img, bounds = subbands.dyadic_tree_image(max_level, active_level=level, active_band=band)
    details = cards.dwt_band_details_html(level, band, subbands.selected_band_stats(bands[band]))
    hist_fig = subbands.band_histogram_figure(bands[band], f"{band}{level}")

    def _upd(name):
        arr = bands[name]
        return gr.update(value=subbands.band_preview(arr),
                         label=f"{name}{level} — {arr.shape[0]}×{arr.shape[1]}")

    return (
        tree_img, bounds,
        _upd("LL"), _upd("LH"), _upd("HL"), _upd("HH"),
        details, band, hist_fig,
    )


def _dwt_band_click(levels_data: dict, level, band_name: str, evt: gr.SelectData):
    """Bir subband önizleme görüntüsüne tıklanınca (mega-spec "DWT LAB —
    subband görüntülerine tıklama ile katsayı inceleme", Part 12/14):
    o bandı 'seçili' yapar, tıklanan pikseldeki GERÇEK ham katsayı
    değerini gösterir VE tıklanan hücreyi görüntü üzerinde işaretler.

    `band_preview`/`band_preview_with_marker` (_stretch) diziyi YENİDEN
    BOYUTLANDIRMAZ (yalnız değer aralığını 0-255'e gerer), bu yüzden
    evt.index (native piksel uzayı — DCT Lab'da doğrulanmış Gradio
    davranışı) doğrudan ham katsayı dizisinin satır/sütununa karşılık
    gelir — ek bir ölçekleme GEREKMEZ; gösterilen katsayı ÖNİZLEMENİN
    piksel değeri DEĞİL, `levels_data[level][band_name]` içindeki GERÇEK
    katsayıdır.

    Tıklanan bant DIŞINDAKİ diğer 3 önizleme kasıtlı olarak DÜZ (işaretsiz)
    hale döner — hücre seçimi TEK bir banda ait olmalı; başka bir bant
    seçildiğinde/görüntü ya da wavelet değiştiğinde eski işaret ASLA
    kalmamalı (mega-spec: "Yeni subband seçildiğinde eski hücre seçimini
    temizle").

    Ağaç görselini de YENİDEN ÇİZER ve döner (mega-spec "TÜM SUBBAND
    NODE'LARINI TAM INTERAKTİF HALE GETİR" Part 6) — sağdaki kartlardan
    yapılan seçim, soldaki ağacın vurgusuyla senkron kalmalı."""
    level = int(level)
    if not levels_data or level not in levels_data or band_name not in levels_data[level]:
        empty = gr.update()
        return band_name, "", None, gr.update(), gr.update(), empty, empty, empty, empty

    bands = levels_data[level]

    def _plain(name):
        arr = bands[name]
        return gr.update(value=subbands.band_preview(arr),
                         label=f"{name}{level} — {arr.shape[0]}×{arr.shape[1]}")

    band = bands[band_name]
    x, y = evt.index
    pixel = None
    row = col = None
    if 0 <= y < band.shape[0] and 0 <= x < band.shape[1]:
        row, col = int(y), int(x)
        pixel = (row, col, float(band[row, col]))
    stats = subbands.selected_band_stats(band)
    details = cards.dwt_band_details_html(level, band_name, stats, pixel)
    hist_fig = subbands.band_histogram_figure(band, f"{band_name}{level}")
    max_level = max(levels_data.keys())
    tree_img, bounds = subbands.dyadic_tree_image(max_level, active_level=level, active_band=band_name)

    marked = gr.update(value=subbands.band_preview_with_marker(band, row, col),
                       label=f"{band_name}{level} — {band.shape[0]}×{band.shape[1]}")
    img_updates = {name: (marked if name == band_name else _plain(name))
                  for name in ("LL", "LH", "HL", "HH")}
    return (
        band_name, details, hist_fig, tree_img, bounds,
        img_updates["LL"], img_updates["LH"], img_updates["HL"], img_updates["HH"],
    )


def dwt_click_ll(levels_data, level, evt: gr.SelectData):
    return _dwt_band_click(levels_data, level, "LL", evt)


def dwt_click_lh(levels_data, level, evt: gr.SelectData):
    return _dwt_band_click(levels_data, level, "LH", evt)


def dwt_click_hl(levels_data, level, evt: gr.SelectData):
    return _dwt_band_click(levels_data, level, "HL", evt)


def dwt_click_hh(levels_data, level, evt: gr.SelectData):
    return _dwt_band_click(levels_data, level, "HH", evt)


def dwt_tree_click(levels_data: dict, bounds: list, max_level, evt: gr.SelectData):
    """Ağaç görselindeki (gr.Image) HERHANGİ bir düğüme (LL/LH/HL/HH,
    herhangi bir seviye) tıklamak hem O SEVİYEYİ hem O BANDI seçer
    (mega-spec "TÜM SUBBAND NODE'LARINI TAM INTERAKTİF HALE GETİR") —
    piksel sınırları subbands.dyadic_tree_image tarafından ax.transData'dan
    KESİN olarak hesaplanmıştır (tahmini/yaklaşık değildir).

    GERÇEK BUG (bu görevde bulunup düzeltildi): önceki sürüm yalnız
    subbands.level_from_click_y (yalnız Y ekseni) kullanıyordu — LH/HL/HH
    çipleri GÖRSEL olarak ayrı düğümler gibi duruyordu ama hangi X
    aralığına tıklandığı HİÇ kontrol edilmediğinden tıklama her zaman o
    satırın (yalnız) SEVİYESİNİ değiştiriyor, bandı asla değiştirmiyordu.
    Artık subbands.subband_from_click hem X hem Y'yi okuyup gerçek
    (seviye, bant) çiftini çözüyor.

    Bu fonksiyon dwt_select_level'i (LL için zaten çalışan TEK canonical
    seçim mantığı) DOĞRUDAN çağırır — LH/HL/HH için ayrı/yinelenen bir
    render mantığı YAZILMAZ (mega-spec Part 20: "Duplicate LL/LH/HL/HH
    kodları yazma"); yalnız hangi (level, band) ile çağrılacağını çözer."""
    x, y = evt.index
    level, band = subbands.subband_from_click(bounds, x, y, int(max_level))
    (tree_img, new_bounds, ll_upd, lh_upd, hl_upd, hh_upd,
     details, selected_band, hist_fig) = dwt_select_level(levels_data, level, band)
    return (
        level, tree_img, new_bounds, ll_upd, lh_upd, hl_upd, hh_upd,
        details, selected_band, hist_fig,
    )


def dwt_mosaic_click(mosaic_img, levels_data: dict, max_level, evt: gr.SelectData):
    """Piramit Katsayı Haritası'na tıklamak hangi (level, band) bölgesine
    denk geldiğini çözer VE o bölgenin İÇİNDEKİ GERÇEK ham DWT katsayısını
    gösterir (mega-spec "DWT LAB — TIKLAMA İLE NOKTA İNCELEMEYİ 3
    GÖRSELDE AKTİF ET" Part 1/5). subbands.pyramid_regions, mozaiğin
    GERÇEKTEN çizildiği iç içe 2x2 kırpma geometrisini simüle eder — bu
    yüzden çözülen (level, band, satır, sütun) piksel-kesindir; DWT
    YENİDEN HESAPLANMAZ, yalnız zaten mevcut `levels_data` önbelleğinden
    okunur."""
    if mosaic_img is None or not levels_data:
        return gr.update(), gr.update()
    x, y = evt.index
    regions = subbands.pyramid_regions(levels_data, int(max_level))
    hit = subbands.subband_from_mosaic_click(regions, x, y)
    if hit is None:
        return gr.update(), gr.update()
    level, band, row, col = hit["level"], hit["band"], hit["row"], hit["col"]
    value = float(levels_data[level][band][row, col])
    info = cards.mosaic_pixel_html(level, band, row, col, value)
    radius = max(2, min(mosaic_img.shape[:2]) // 150)
    marked = subbands.mark_point(mosaic_img, y, x, radius=radius)
    return info, marked


def _dwt_pixel_from_click(gray, recon, evt: gr.SelectData):
    """Kuantalamalı Rekonstrüksiyon VE Fark Haritası tıklamalarının
    PAYLAŞTIĞI çekirdek mantık (mega-spec Part 20 ruhu: kopya kod
    yazılmaz) — tıklanan (x,y)'den gerçek (satır, sütun, orijinal piksel,
    rekonstrüksiyon pikseli) döner; ikisi de `gray`/`recon` GERÇEK
    dizilerinden GELİR, render edilmiş preview'dan DEĞİL."""
    if gray is None or recon is None:
        return None
    x, y = evt.index
    h, w = gray.shape
    if not (0 <= y < h and 0 <= x < w):
        return None
    row, col = int(y), int(x)
    return row, col, float(gray[row, col]), float(recon[row, col])


def dwt_recon_click(gray, recon, evt: gr.SelectData):
    """Kuantalamalı Rekonstrüksiyon görüntüsüne tıklamak o pikseldeki
    GERÇEK orijinal/rekonstrüksiyon değerlerini ve farkını gösterir
    (mega-spec Part 2) — yalnız rekonstrüksiyon görüntüsü işaretlenir."""
    hit = _dwt_pixel_from_click(gray, recon, evt)
    if hit is None:
        return gr.update(), gr.update()
    row, col, orig_val, recon_val = hit
    info = cards.recon_pixel_html(row, col, orig_val, recon_val)
    radius = max(2, min(gray.shape) // 150)
    marked = subbands.mark_point(np.clip(recon, 0, 255).astype(np.uint8), row, col, radius=radius)
    return info, marked


def dwt_diff_click(gray, recon, evt: gr.SelectData):
    """Fark Haritası'na tıklamak AYNI (orijinal, rekonstrüksiyon, fark)
    üçlüsünü gösterir (mega-spec Part 3) — fark = orijinal - rekonstrüksiyon
    (subbands.reconstruction_diff_image ile AYNI işaret kuralı); yalnız
    fark haritası görüntüsü işaretlenir."""
    hit = _dwt_pixel_from_click(gray, recon, evt)
    if hit is None:
        return gr.update(), gr.update()
    row, col, orig_val, recon_val = hit
    info = cards.recon_pixel_html(row, col, orig_val, recon_val)
    diff_img = subbands.reconstruction_diff_image(gray, recon)
    radius = max(2, min(gray.shape) // 150)
    marked = subbands.mark_point(diff_img, row, col, radius=radius)
    return info, marked


_DWT_RATE_MODE_MANUAL = "Manuel Δ"
_DWT_RATE_MODE_TARGET = "Hedef BPP"
# compare.py'deki _WAVELET_STEP_RANGE ile AYNI arama aralığı (bilinçli
# olarak tekrar tanımlanır — DWT Lab compare.py'ye bağımlı değildir,
# ama AYNI bisection deseni/aralığı kullanır: 0.15-2.5 bpp hedeflerinin
# TAMAMINDA zaten kanıtlanmış çalışan bir aralık).
_DWT_STEP_RANGE = (0.05, 512.0)


def toggle_dwt_rate_mode(mode: str):
    """[ Manuel Δ ] [ Hedef BPP ] — iki kontrol ASLA aynı anda aktif
    olmaz (mega-spec "DWT LAB sol kontrol panelini genişlet": "Aynı anda
    iki farklı kontrolün birbirini sessizce ezmesine izin verme")."""
    is_target = mode == _DWT_RATE_MODE_TARGET
    return gr.update(visible=is_target), gr.update(visible=not is_target)


def run_dwt_explorer(image, wavelet, levels, quant_step, inspect_level,
                     mode: str = wavelet_engine.DEFAULT_BOUNDARY_MODE,
                     rate_mode: str = _DWT_RATE_MODE_MANUAL, target_bpp: float = 0.5):
    if image is None:
        raise gr.Error("Önce üstteki AKTİF GÖRÜNTÜ çubuğundan bir görüntü seçin.")
    image = _prepare(image)
    gray = _to_gray(image)
    levels = min(int(levels), max_decomposition_level(gray.shape, wavelet))

    coeffs = decompose_for_viz(gray, wavelet, levels, mode)
    mosaic = subbands.pyramid_display_image(coeffs)
    # Kanonik, DÜZELTİLMİŞ seviye→dizi eşlemesi — TEK yerde hesaplanır,
    # seviye seçici (dwt_select_level) ve kart tıklamaları (dwt_click_*)
    # SADECE bunu okur, asla yeniden DWT çalıştırmaz (Part 19).
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)
    inspect_level = max(1, min(int(inspect_level), levels))
    (tree_img, bounds, ll_upd, lh_upd, hl_upd, hh_upd,
     band_details, selected_band, hist_fig) = dwt_select_level(levels_data, inspect_level, "LL")
    energy_fig = subbands.energy_distribution_figure(coeffs)

    filt_fig = subbands.filter_bank_figure(wavelet)
    filt_flow = cards.filter_bank_flow_html(wavelet)
    info = subbands.filter_bank_info(wavelet)

    stats = subbands.coeff_stats(coeffs)
    stats_rows = [[r["label"], r["shape"], r["count"], f"{r['min']:.1f}", f"{r['max']:.1f}",
                  f"{r['mean']:.2f}", f"{r['std']:.2f}", f"{r['energy_pct']:.1f}%",
                  f"{r['zero_pct']:.1f}%"] for r in stats]

    # --- Kayıpsız rekonstrüksiyon (kuantalama yok): teorik doğrulama ---
    lossless = pywt.waverec2(coeffs, wavelet, mode=mode)[:gray.shape[0], :gray.shape[1]] + 128.0
    lossless = np.clip(lossless, 0, 255)
    lossless_err = float(np.abs(lossless - gray).max())
    lossless_mse = mse(gray, lossless)
    validation_card = cards.validation_card_html(lossless_err, lossless_mse, tol=1e-3)

    # --- Kuantalamalı (lossy) rekonstrüksiyon ---
    # İki AYRI, birbirini SESSİZCE EZMEYEN mod (mega-spec "DWT LAB sol
    # kontrol panelini genişlet"):
    #  - Manuel Δ: kullanıcının verdiği quant_step DOĞRUDAN kullanılır
    #    (önceki davranışla BİREBİR aynı).
    #  - Hedef BPP: target_bpp'yi tutturan quant_step, compare.py'deki
    #    run_wavelet ile AYNI match_bpp bisection'ı kullanılarak GERÇEKTEN
    #    aranır — targetBpp → adım ara → kuantala → entropi/rate tahmini
    #    hesapla → achievedBpp. quant_step, arama SONUCUYLA
    #    DEĞİŞTİRİLİR (aşağıdaki her şey — seyreklik kartı, özet bar'daki
    #    "Δ" — TUTARLI olsun diye).
    n_pixels = gray.shape[0] * gray.shape[1]

    def _encode_for_bpp_search(step: float) -> tuple[np.ndarray, float]:
        # match_bpp encode(param) -> (recon, bpp) BEKLER; compress_channel
        # ise (recon, TOPLAM BİT) döner — GERÇEK BUG (bu görevde bulunup
        # düzeltildi): bit sayısı n_pixels'e bölünmeden match_bpp'e
        # verilirse bisection tamamen anlamsız bir büyüklükle (piksel
        # başına değil, TÜM görüntü başına "bpp") çalışır ve asla
        # yakınsamaz (compare.run_wavelet bu hataya düşmez çünkü
        # compress_image'ı kullanır, o zaten bpp'ye bölünmüş döner —
        # burada compress_channel kullanıldığından bölme AÇIKÇA gerekir).
        r, total_bits = wavelet_engine.compress_channel(gray, step, wavelet, levels, mode=mode)
        return r, total_bits / n_pixels

    if rate_mode == _DWT_RATE_MODE_TARGET:
        recon, achieved_bpp, quant_step = match_bpp(
            _encode_for_bpp_search, float(target_bpp), *_DWT_STEP_RANGE, False,
        )
    else:
        recon, total_bits = wavelet_engine.compress_channel(gray, float(quant_step), wavelet, levels, mode=mode)
        achieved_bpp = total_bits / n_pixels
    # Orijinal | rekonstrüksiyon | fark haritası — gerçek quantized→IDWT
    # rekonstrüksiyonu (mega-spec Part 29), lossless doğrulamayla
    # KARIŞTIRILMAZ.
    diff_img = subbands.reconstruction_diff_image(gray, recon)

    # --- Kuantalama sonrası seyreklik (ham vs kuantalanmış) ---
    quantized = quantize_for_viz(gray, float(quant_step), wavelet, levels, mode)
    sparsity = subbands.quantized_sparsity_stats(coeffs, quantized)
    sparsity_card = cards.sparsity_card_html(sparsity)

    max_level = max_decomposition_level(gray.shape, wavelet)
    # GERÇEK BUG DÜZELTMESİ (mega-spec "DWT LAB sol kontrol panelini
    # genişlet"): pywt'de `Wavelet.biorthogonal` TÜM dalgacıklar için
    # (ortogonal olanlar DAHİL — db2/db4/db8/db12/haar/sym4/coif1 hepsi
    # `biorthogonal=True` döner, çünkü ortogonallik biorthogonalliğin özel
    # bir durumudur) True'dur; önceki kod `info["biorthogonal"]`i BİRİNCİL
    # koşul olarak kullandığından TÜM dalgacıklar için yanlışlıkla
    # "Biortogonal" gösteriyordu. Doğru ayrım `orthogonal` bayrağıdır —
    # yalnız GERÇEKTEN ortogonal olmayan (bior2.2/bior4.4 gibi) dalgacıklar
    # "Biortogonal" olarak işaretlenir.
    transform_type = "Ortogonal" if info["orthogonal"] else "Biortogonal"
    rate_chips = [("TAHMİNİ BPP", f"{achieved_bpp:.3f}")]
    if rate_mode == _DWT_RATE_MODE_TARGET:
        rate_chips = [("HEDEF BPP", f"{float(target_bpp):.3f}"),
                     ("TAHMİNİ BPP", f"{achieved_bpp:.3f}"),
                     ("BULUNAN Δ", f"{float(quant_step):.2f}")]
    summary = cards.summary_bar_html([
        ("WAVELET", info["name"]),
        ("FİLTRE", f"{info['dec_len']} tap"),
        # "SEVİYE 4/5" belirsizdi — hangisi şu an gösteriliyor, hangisi
        # üst sınır belli değildi (mega-spec "FINAL PRE-PRESENTATION QA"
        # Part 7). İki AYRI etiket: gerçekten kullanılan seviye + yalnız
        # bilgi amaçlı matematiksel üst sınır.
        ("AYRIŞTIRMA SEVİYESİ", f"{levels}"),
        ("MAKS. GEÇERLİ SEVİYE", f"{max_level}"),
        ("SINIR MODU", mode),
        ("DÖNÜŞÜM", transform_type),
        ("REKONSTRÜKSİYON", "PASS" if lossless_err < 1e-3 else "FAIL"),
        # Renk kanalı kapsamı açıkça belirtilir (numerical-correctness audit
        # Part 36): bu ayrıştırma yalnız luminance üzerindedir; Karşılaştır
        # sekmesindeki gerçek DWT sıkıştırması RGB'yi YCbCr'ye çevirip her
        # kanalı ayrı işler — buradaki PSNR/BPP değerleri RGB-tüm-görüntü
        # metrikleriyle DOĞRUDAN kıyaslanamaz.
        ("KANAL", "Luminance (gri)"),
        *rate_chips,
    ])

    filt_text = (
        f"Analiz alçak-geçiren h[n]: {info['dec_lo']}\n\n"
        f"Analiz yüksek-geçiren g[n]: {info['dec_hi']}\n\n"
        f"Sentez alçak-geçiren: {info['rec_lo']}\n\n"
        f"Sentez yüksek-geçiren: {info['rec_hi']}"
    )
    return (
        summary, mosaic, tree_img, bounds,
        ll_upd, lh_upd, hl_upd, hh_upd, band_details, selected_band, hist_fig,
        np.clip(recon, 0, 255).astype(np.uint8), diff_img, energy_fig,
        filt_flow, filt_fig, filt_text, stats_rows, validation_card, sparsity_card,
        gr.update(maximum=levels, value=inspect_level),
        levels_data, levels,
        # Piramit/Rekonstrüksiyon/Fark Haritası nokta-inceleme bilgi
        # kutuları TEMİZLENİR (mega-spec "DWT LAB — TIKLAMA İLE NOKTA
        # İNCELEMEYİ 3 GÖRSELDE AKTİF ET" Part 7: wavelet/seviye/kuantalama
        # değişince eski nokta seçimi kalmamalı) + gray/recon durumunu
        # (float64, GERÇEK piksel değerleri) SAKLAR — her tıklamada DWT'yi
        # yeniden hesaplamamak için.
        "", "", gray, recon,
    )


# =============================================================================
# SEKME 4 — SEMANTİK ROI (mevcut sıkıştırma algoritması DEĞİŞMEDEN;
# ROI kaynağı OTOMATİK (YOLO, nesne bazında include/exclude) veya
# MANUEL (tıkla + boyut) olabilir — ikisi de AYNI mask_to_block_importance /
# importance_mask yoluna girer.)
# =============================================================================
def sem_detect_objects(image):
    """'Nesneleri Tespit Et' — YOLO'yu BİR KEZ çalıştırır, nesne başına
    include/exclude edilebilir bir liste üretir. Buton etiketi de
    döndürülür (mega-spec "FINAL PRE-PRESENTATION QA" Part 4): bir tespit
    ZATEN tamamlandıysa buton 'YENİDEN TESPİT ET' olur — kullanıcı dolu
    bir listeye bakarken hâlâ 'Nesneleri Tespit Et' görmez."""
    if image is None:
        raise gr.Error("Önce üstteki AKTİF GÖRÜNTÜ çubuğundan bir görüntü seçin.")
    image = _prepare(image)
    instances = get_importance_instances(image)
    btn_upd = gr.update(value="YENİDEN TESPİT ET")
    if not instances:
        return (instances, gr.update(choices=[], value=[]),
               cards.empty_state_html("Nesne bulunamadı",
                                      "İnsan/araç/hayvan gibi COCO nesneleri içeren bir görüntü deneyin."),
               image, btn_upd)
    choices = [f"{i}: {inst['label']} ({inst['confidence']:.0%})" for i, inst in enumerate(instances)]
    mask = fuse_instance_masks(instances, image.shape[:2])
    preview = overlay_mask(image, mask)
    info = cards.image_chip_html(f"{len(instances)} nesne tespit edildi", None)
    return instances, gr.update(choices=choices, value=choices), info, preview, btn_upd


def sem_update_auto_preview(image, instances, selected):
    """Onay kutuları değiştikçe YOLO'yu TEKRAR ÇALIŞTIRMADAN — yalnız
    seçili örneklerin maskelerini yeniden birleştirip önizlemeyi tazeler."""
    if image is None:
        return None
    image = _prepare(image)
    if not instances or not selected:
        return image
    idxs = {int(c.split(":")[0]) for c in selected}
    chosen = [inst for i, inst in enumerate(instances) if i in idxs]
    if not chosen:
        return image
    mask = fuse_instance_masks(chosen, image.shape[:2])
    return overlay_mask(image, mask)


def sem_toggle_mode(mode: str):
    is_auto = mode == "Otomatik (YOLO)"
    return gr.update(visible=is_auto), gr.update(visible=not is_auto)


def sem_roi_click(image, evt: gr.SelectData):
    """Görüntüye tıklayarak manuel ROI merkezini seçer."""
    x, y = evt.index
    return (x, y)


def sem_update_manual_overlay(image, center, w, h):
    if image is None:
        return None
    image = _prepare(image)
    if center is None:
        return image
    cx, cy = center
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = int(cx + w / 2), int(cy + h / 2)
    return draw_rect_overlay(image, x0, y0, x1, y1)


def sem_reset_state():
    """Aktif görüntü DEĞİŞTİĞİNDE çalışır — bu görevin en kritik hatasını
    düzeltir (mega-spec Part 2/6/26-29): önceki görüntüye ait YOLO tespit-
    leri, seçili nesne listesi, manuel ROI merkezi, maske ve TÜM sonuç
    kartları (baseline/semantic rekonstrüksiyon, FG/BG PSNR-SSIM, global
    PSNR, rate) temizlenmeden yeni görüntü altında görünmeye devam
    ediyordu — örn. motocross'ta tespit edilen "person/motorcycle" listesi,
    trafik fotoğrafına geçildiğinde ekranda KALIYORDU.

    Artık active_id her arttığında bu state'lerin TAMAMI sıfırlanır; UI
    başlangıç durumuna ("Nesneler henüz tespit edilmedi") döner. Otomatik
    tespit burada YENİDEN ÇALIŞTIRILMAZ (pahalı YOLO inference'ı gereksiz
    yere tekrarlamamak için) — kullanıcı yeni görüntü için 'Nesneleri
    Tespit Et'e tekrar basmalıdır; bu, spec'in "eğer otomatik tespit image
    change'de zaten çalışıyorsa yeniden çalıştır" koşulu bu uygulamada
    geçerli olmadığından (tespit hep manuel buton ile tetiklenir) doğru
    davranıştır."""
    return (
        [], gr.update(choices=[], value=[]),
        cards.empty_state_html("Nesneler henüz tespit edilmedi",
                               "'Nesneleri Tespit Et' ile başlayın."),
        None,                                              # sem_roi_center_state
        None, None, None,                                  # sem_mask, sem_base, sem_sem
        cards.empty_state_html("GÖRÜNTÜ YÜKLEYİN VE SIKIŞTIR'A BASIN"),
        "", "", "", "",                                     # sem_gain, sem_tradeoff, sem_global_card, sem_global
        gr.update(value="NESNELERİ TESPİT ET"),             # sem_detect_btn (Part 4)
    )


def sem_auto_detect(image, image_id, request: gr.Request):
    """Aktif görüntü seçilir seçilmez YOLO tespitini otomatik çalıştırır
    (mega-spec Part 13) — 'Nesneleri Tespit Et'e basmaya gerek YOKTUR.
    sem_detect_objects'in KENDİSİNİ çağırır (Part 29). Manuel butonun
    aksine (orada gr.Error kullanıcıya doğrudan gösterilir, kabul
    edilebilir), OTOMATİK yolda beklenmeyen bir hata (ör. YOLO ağırlığı
    indirilemedi) TÜM uygulamayı bloke etmemeli (Part 26) — bu yüzden
    burada yakalanıp zarifçe 'tespit başarısız' mesajına çevrilir."""
    if image is None or _is_stale(request.session_hash, image_id):
        return [], gr.update(), gr.update(), gr.update(), gr.update()
    try:
        return sem_detect_objects(image)
    except Exception as exc:  # noqa: BLE001 — otomatik yolda tam izolasyon gerekir
        return (
            [], gr.update(choices=[], value=[]),
            cards.empty_state_html("Otomatik nesne tespiti başarısız oldu",
                                   f"{exc} — 'Nesneleri Tespit Et' ile elle deneyebilir ya da "
                                   f"Manuel ROI moduna geçebilirsiniz."),
            image, gr.update(value="NESNELERİ TESPİT ET"),
        )


def sem_recompute(
    image, image_id, engine, target_bpp, bg_coarseness, mode,
    instances, selected, roi_center, roi_w, roi_h,
    request: gr.Request, progress: gr.Progress = gr.Progress(),
):
    """GENEL, hata-toleranslı semantik sıkıştırma sarmalayıcısı — hem ilk
    otomatik analiz (Part 4/13) HEM DE parametre değişikliği kaynaklı
    hedefli yeniden hesaplamalar (Part 5: 'Semantic object checkbox
    değişti → Semantic compression only', hedef bpp/motor/kabalık/manuel
    ROI slider bırakma) BUNU çağırır — run_semantic_pipeline'ın KENDİSİNİ
    sarar (Part 29: algoritma tekrarlanmaz), tek fark: kullanıcı buton
    BASMADAN (pasif bir slider/checkbox değişikliğiyle) tetiklendiği için
    gr.Error yerine zarif bir mesaj bırakır — ör. otomatik modda tüm
    nesneler check'ten çıkarılırsa veya manuel modda henüz ROI merkezi
    seçilmemişse, rahatsız edici bir hata TOAST'ı YERİNE dürüst bir 'N/A'
    kartı gösterilir. SIKIŞTIR butonu (kullanıcının doğrudan eylemi) hâlâ
    run_semantic_pipeline'ı DOĞRUDAN çağırır — orada gr.Error kabul
    edilebilir/beklenen bir geri bildirimdir."""
    if image is None or _is_stale(request.session_hash, image_id):
        return (gr.update(),) * 8  # bkz. sem_btn.click çıktı listesi
    if mode == "Otomatik (YOLO)" and not instances:
        msg = cards.empty_state_html(
            "Otomatik modda semantik nesne bulunamadı",
            "Bu görüntüde YOLO hiçbir COCO nesnesi tespit etmedi; anlamsız bir "
            "foreground/background metriği üretilmedi. 'Manuel ROI' moduna geçip "
            "kaynak görüntüye tıklayarak elle bir bölge seçebilirsiniz.",
        )
        return None, None, None, msg, "", "", "", ""
    if mode == "Otomatik (YOLO)" and instances and not selected:
        msg = cards.empty_state_html(
            "Hiçbir nesne seçili değil",
            "En az bir tespit edilen nesneyi işaretleyin.")
        return None, None, None, msg, "", "", "", ""
    if mode != "Otomatik (YOLO)" and roi_center is None:
        msg = cards.empty_state_html(
            "Manuel ROI merkezi seçilmedi",
            "Kaynak görüntüye tıklayarak bir ROI merkezi seçin.")
        return None, None, None, msg, "", "", "", ""
    try:
        return run_semantic_pipeline(
            image, engine, target_bpp, bg_coarseness, mode,
            instances, selected, roi_center, roi_w, roi_h, progress,
        )
    except Exception as exc:  # noqa: BLE001 — pasif tetiklenen yolda tam izolasyon gerekir
        msg = cards.empty_state_html("Semantik sıkıştırma başarısız oldu", str(exc))
        return None, None, None, msg, "", "", "", ""


def sem_auto_compress(
    image, image_id, instances, selected, engine, target_bpp, bg_coarseness,
    request: gr.Request, progress: gr.Progress = gr.Progress(),
):
    """İlk otomatik analiz (mega-spec Part 4/13) — tespitten SONRA VARSAYILAN
    motor/hedef bpp ile Otomatik (YOLO) modunda çalışır. sem_recompute'un
    ince bir sarmalayıcısıdır (Part 29: algoritma tekrarlanmaz)."""
    return sem_recompute(
        image, image_id, engine, target_bpp, bg_coarseness, "Otomatik (YOLO)",
        instances, selected, None, 128, 128, request, progress,
    )


def run_semantic_pipeline(
    image: np.ndarray | None,
    engine: str,
    target_bpp: float,
    bg_coarseness: float,
    mode: str,
    instances: list,
    selected: list,
    roi_center,
    roi_w: float,
    roi_h: float,
    progress: gr.Progress = gr.Progress(),
):
    if image is None:
        raise gr.Error("Önce üstteki AKTİF GÖRÜNTÜ çubuğundan bir görüntü seçin.")
    image = _prepare(image)
    h, w = image.shape[:2]

    progress(0.1, desc="ROI maskesi hazırlanıyor…")
    if mode == "Otomatik (YOLO)":
        if not instances:
            raise gr.Error("Önce 'Nesneleri Tespit Et' ile nesneleri bulun.")
        idxs = {int(c.split(":")[0]) for c in (selected or [])}
        chosen = [inst for i, inst in enumerate(instances) if i in idxs]
        if not chosen:
            raise gr.Error("En az bir tespit edilen nesne seçili olmalı.")
        mask = fuse_instance_masks(chosen, (h, w))
        labels = [inst["label"] for inst in chosen]
    else:
        if roi_center is None:
            raise gr.Error("Önce KAYNAK GÖRÜNTÜ üzerine tıklayarak ROI merkezini seçin.")
        cx, cy = roi_center
        mask = rectangle_mask((h, w), int(cx - roi_w / 2), int(cy - roi_h / 2),
                              int(cx + roi_w / 2), int(cy + roi_h / 2))
        labels = ["manuel ROI"]

    if not mask.any():
        raise gr.Error("Seçili ROI boş; farklı bir bölge/nesne seçin.")
    overlay = overlay_mask(image, mask)

    progress(0.35, desc="Baseline (uniform) hedef bpp'ye eşleniyor…")
    if engine == "DCT (JPEG mantığı)":
        block_imp = mask_to_block_importance(mask)
        base = match_bpp(lambda q: dct_engine.compress_image(image, q),
                         target_bpp, 1, 100, True)
        progress(0.65, desc="Semantic (ROI) hedef bpp'ye eşleniyor…")
        sem = match_bpp(
            lambda q: dct_engine.compress_image(image, q, block_imp, bg_coarseness),
            target_bpp, 1, 100, True,
        )
    else:
        wav = config.WAVELET_DEFAULT_FILTER
        base = match_bpp(lambda s: wavelet_engine.compress_image(image, s, wav),
                         target_bpp, 0.25, 512, False)
        progress(0.65, desc="Semantic (ROI) hedef bpp'ye eşleniyor…")
        sem = match_bpp(
            lambda s: wavelet_engine.compress_image(
                image, s, wav, importance_mask=mask, bg_coarseness=bg_coarseness
            ),
            target_bpp, 0.25, 512, False,
        )

    progress(0.9, desc="Metrikler hesaplanıyor…")
    mb = evaluate(image, base[0], mask)
    ms = evaluate(image, sem[0], mask)

    budget_card = cards.same_budget_badge_html(base[1], sem[1])
    # Terminoloji mega-spec "FINAL PRE-PRESENTATION QA" Part 6 ile birebir:
    # SEMANTİK BÖLGE KAZANCI / ARKA PLAN KALİTE KAYBI / GLOBAL KALİTE
    # DEĞİŞİMİ — belirsiz ifade YOK, ödünleşim açıkça "kayıp/değişim"
    # olarak adlandırılır (gizlenmez).
    gain_card = cards.multi_tradeoff_card_html(
        "SEMANTİK BÖLGE KAZANCI (foreground)", ACCENT_GREEN,
        [
            ("PSNR", mb["fg_psnr"], ms["fg_psnr"], " dB", True, 2),
            ("SSIM", mb["fg_ssim"], ms["fg_ssim"], "", True, 4),
        ],
    )
    tradeoff_card = cards.tradeoff_card_html(
        "ARKA PLAN KALİTE KAYBI (background PSNR)", "#F59E0B",
        mb["bg_psnr"], ms["bg_psnr"], " dB", True,
    )
    # Global PSNR artık yalnız paragraf içinde DEĞİL, ayrıca kompakt bir
    # metrik kart olarak da görünür (Part 5: "Do not hide Global PSNR
    # only inside a paragraph") — tam ödünleşim tek bakışta anlaşılır.
    global_card = cards.tradeoff_card_html(
        "GLOBAL KALİTE DEĞİŞİMİ (global PSNR)", "#94A3B8",
        mb["psnr"], ms["psnr"], " dB", True,
    )
    global_note = (
        f"**Global PSNR:** baseline {mb['psnr']:.2f} dB → semantic {ms['psnr']:.2f} dB "
        f"({ms['psnr']-mb['psnr']:+.2f} dB). Semantik tahsis, tüm görüntüde tekdüze "
        f"sadakat yerine tespit edilen önemli bölgelerin kalitesini önceliklendirir; "
        f"bu nedenle global PSNR artmak ZORUNDA değildir — bu bir hata değil, "
        f"tasarımın amaçladığı bir ödünleşimdir.\n\n"
        f"Tespit edilen nesneler: {', '.join(sorted(set(labels)))} "
        f"(maske kapsama: %{mask.mean()*100:.0f})."
    )
    return overlay, base[0], sem[0], budget_card, gain_card, tradeoff_card, global_card, global_note


# =============================================================================
# UI
# =============================================================================
_THEORY_SECTIONS = [
    ("JPEG / DCT", ACCENT_CYAN, """
```
GÖRÜNTÜ → BLOKLAR → DCT → KUANTALAMA → ENTROPİ KODLAMA → REKONSTRÜKSİYON
```
DCT, uzamsal piksel değerlerini frekans katsayılarına çevirir. Geleneksel
JPEG 8×8 blok kullanır. Kuantalama, daha az önemli yüksek frekans bilgisini
azaltır/sıfırlar — enerji az sayıda büyük katsayıda toplanır (güçlü sıkışma),
ama yüksek sıkıştırmada 8×8 sınırlarında blok artefaktları görülebilir.
DC katsayısı = bloğun ortalama parlaklığı; sol-üstten sağ-alta frekans artar.
"""),
    ("Wavelet / DWT", ACCENT_PURPLE, """
```
GÖRÜNTÜ → ALÇAK/YÜKSEK-GEÇİREN FİLTRELEME → ALT ÖRNEKLEME
        → LL / LH / HL / HH → LL'yi RECURSIVELY AYRIŞTIR → KUANTALAMA/KODLAMA
```
Çok-çözünürlüklü bir gösterim: blok sınırı yoktur, bu yüzden klasik 8×8
bloklaşma artefaktı görülmez. LL = yaklaşım (düşük frekans), LH/HL/HH =
yatay/dikey/köşegen detay. **Önemli ayrım:** bu projedeki motor JPEG2000'in
dayandığı dönüşüm ailesini kullanır ama JPEG2000'in KENDİSİ değildir —
gerçek bit akışı/EBCOT kodlaması yoktur (bkz. Sınırlamalar).
"""),
    ("PSNR", "#94A3B8", """
**PSNR = 10·log₁₀(255² / MSE)** — orijinale sayısal (piksel-bazlı) yakınlığı
ölçer. Yüksek PSNR genelde daha az bozulma demektir, ama PSNR mükemmel bir
algısal kalite metriği DEĞİLDİR; "40 dB = mükemmel kalite" gibi mutlak bir
eşik yoktur — yorum görüntüye ve yöntemin bozulma karakterine bağlıdır.
"""),
    ("SSIM", "#94A3B8", """
Yapısal benzerlik indeksi; parlaklık/kontrast/yapı bileşenlerini karşılaştırır
ve MSE/PSNR gibi saf piksel-hatası metriklerini algısal olarak tamamlar.
1.0 = birebir aynı yapı, 0'a yaklaştıkça yapısal fark artar.
"""),
    ("BPP ve Sıkıştırma Oranı", "#94A3B8", """
**BPP = sıkıştırılmış toplam bit / (genişlik × yükseklik)**. Özel DCT/DWT
motorları gerçek bir bit akışı yazmaz; bpp, order-0 Shannon entropi
tahminine dayanır (`src/engines/entropy.py`) — arayüzde **"entropi
tahmini"** olarak etiketlenir. Gerçek JPEG2000 (Pillow/OpenJPEG mevcutsa)
gerçek kodlanmış bayt sayısı kullanır — **"gerçek boyut"** etiketlenir.
**Sıkıştırma oranı = ham_bit / sıkıştırılmış_bit** (ham = genişlik×yükseklik×kanal×8).
"""),
    ("Rate–Distortion Analizi", "#94A3B8", """
Aynı görüntü için birden çok hedef bpp noktasında gerçek ölçümler alınıp
BPP–PSNR eğrisi çizilir. Eğriler ARASI kıyas, hangi yöntemin aynı bit
bütçesinde daha az bozulma ürettiğini gösterir; düşük bpp'de dalgacık
genelde DCT'nin önündedir (bloklaşma olmadığından).
"""),
    ("Semantik-Farkında Sıkıştırma", ACCENT_GREEN, """
YOLO-seg ile tespit edilen önemli bölgelere (insan, araç, ...) **aynı toplam
bit bütçesi içinde** daha fazla bit yönlendirilir; arka plan buna karşılık
daha kaba kuantalanır. Foreground kalitesi artar, background ve genelde
global PSNR bir miktar düşer — bu, tasarımın amaçladığı bir ödünleşimdir,
hata değildir.
"""),
]


def _theory_html() -> str:
    blocks = "".join(
        f'<div class="lab-card" style="border-top:3px solid {color}">'
        f'<div class="metric-title" style="margin-bottom:8px">{title}</div>'
        f'<div style="font-size:13px;color:var(--lab-text-secondary);line-height:1.55">{body}</div>'
        f'</div>'
        for title, color, body in _THEORY_SECTIONS
    )
    # Basit markdown -> HTML (kod bloğu + kalın metin); gr.HTML markdown yorumlamaz
    import re
    def _code(m):
        return (f'<pre style="background:var(--lab-bg-elevated);border:1px solid var(--lab-border);'
               f'border-radius:6px;padding:8px 10px;font-family:var(--lab-mono);font-size:11px;'
               f'color:var(--lab-text-primary);overflow-x:auto">{m.group(1)}</pre>')
    blocks = re.sub(r"```\n(.*?)\n```", _code, blocks, flags=re.S)
    blocks = re.sub(r"\*\*(.+?)\*\*",
                    r'<strong style="color:var(--lab-text-primary)">\1</strong>', blocks, flags=re.S)
    blocks = blocks.replace("\n", "<br>")
    return f'<div class="theory-grid">{blocks}</div>'


def build_ui() -> gr.Blocks:
    header_html = """
<div class="lab-header">
  <p class="lab-title">SEMANTIC COMPRESSION LAB</p>
  <p class="lab-subtitle">DCT · Wavelet · Semantic Rate–Distortion Analysis</p>
</div>"""

    with gr.Blocks(title="Semantic Compression Lab", theme=LAB_THEME, css=LAB_CSS,
                   js=FORCE_DARK_JS) as demo:
        gr.HTML(header_html)

        # ---- TEK GLOBAL AKTİF GÖRÜNTÜ (tüm sekmelerin ÜSTÜNDE, paylaşılan) ----
        active_img, active_id, active_meta, home_btn = build_active_image_bar()

        # ---- OTOMATİK TAM ANALİZ DURUMU (mega-spec "AUTOMATIC FULL ANALYSIS
        # PIPELINE" Part 3) — "Analyzing image... ✓ DCT ✓ DWT ..." kompakt
        # şerit, bitince "ANALİZ HAZIR"a çöker. status_state (hangi aşamalar
        # gerçekten bitti) ile status_html (görünen metin) HER auto-analiz
        # zincirinin son adımında birlikte güncellenir (bkz. build_ui sonu).
        status_state = gr.State(set())
        status_html = gr.HTML(visible=False)
        active_id.change(_status_reset, None, [status_state, status_html])

        with gr.Tabs():
            # ---------------- SEKME 1: KARŞILAŞTIR ----------------
            with gr.Tab("Karşılaştır"):
                with gr.Row(visible=False, elem_classes=["experiment-bar"]) as toolbar:
                    with gr.Column(scale=0, min_width=140):
                        target_mode = gr.Radio([_TARGET_MODE_BPP, _TARGET_MODE_RATIO],
                                               value=_TARGET_MODE_BPP, show_label=False)
                    with gr.Column(scale=2, min_width=220):
                        # Etiket/aralık modla BİRLİKTE değişir (bkz.
                        # target_mode_ui) — "0.5" değerinin bpp mi
                        # sıkıştırma oranı mı olduğu ASLA belirsiz kalmaz.
                        target_value = gr.Slider(0.1, 4.0, value=0.5, step=0.05,
                                                 label="Hedef bpp")
                    with gr.Column(scale=0, min_width=170):
                        with gr.Accordion("⚙ Deney Ayarları", open=False):
                            gr.Markdown('<div class="section-label">JPEG / DCT</div>')
                            dct_block_size = gr.Dropdown(
                                [str(b) for b in config.DCT_BLOCK_SIZE_OPTIONS],
                                value=str(config.DCT_BLOCK_SIZE), label="Blok boyutu")
                            gr.Markdown(
                                '<div class="section-label">JPEG2000 / DWT</div>'
                                '<span style="font-size:var(--text-sm);color:var(--lab-text-muted)">'
                                "OpenJPEG tabanlı gerçek wavelet codec — kendi rate-control'ünü "
                                "kullanır, seçilebilir bir parametresi yoktur.</span>"
                                if JPEG2000_AVAILABLE else
                                '<div class="section-label">JPEG2000 / DWT</div>'
                                '<span style="font-size:var(--text-sm);color:var(--lab-text-muted)">'
                                "Bu ortamda gerçek bir OpenJPEG kurulumu yok.</span>"
                            )
                            gr.Markdown('<div class="section-label">GERÇEK JPEG (libjpeg)</div>')
                            include_real_jpeg = gr.Checkbox(
                                value=False,
                                label=f"Gerçek JPEG (libjpeg) ile de karşılaştır "
                                     f"({'mevcut' if REAL_JPEG_AVAILABLE else 'bu ortamda yok'})",
                                interactive=REAL_JPEG_AVAILABLE)
                    with gr.Column(scale=0, min_width=140):
                        cmp_btn = gr.Button("KARŞILAŞTIR", variant="primary")

                # --- iş alanı — SAME-RATE COMPARISON (yalnız görüntüler) ---
                with gr.Column(visible=False) as workspace:
                    gr.Markdown(
                        '<div class="page-section-label section-gap-sm">SAME-RATE COMPARISON</div>'
                        '<span style="font-size:var(--text-sm);color:var(--lab-text-muted)">'
                        'JPEG/DCT vs JPEG2000/DWT — GERÇEK codec vs GERÇEK codec, '
                        'yaklaşık aynı bit oranında</span>'
                    )
                    with gr.Row(elem_classes=["view-toggle-row"]):
                        cmp_view_mode = gr.Radio(["Yan Yana", "Önce / Sonra"], value="Yan Yana",
                                                 show_label=False, elem_id="cmp-view-mode")
                    with gr.Column(visible=True) as same_side_col:
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('<div class="frame-caption" style="text-align:center">ORİJİNAL</div>')
                                out_orig = gr.Image(show_label=False, height=340, elem_classes=["frame-neutral"])
                            with gr.Column():
                                gr.Markdown(f'<div class="frame-caption" style="text-align:center;color:{ACCENT_CYAN}">JPEG / DCT</div>')
                                out_dct = gr.Image(show_label=False, height=340, elem_classes=["frame-jpeg"])
                            with gr.Column():
                                gr.Markdown(f'<div class="frame-caption" style="text-align:center;color:{ACCENT_PURPLE}">JPEG2000 / DWT</div>')
                                out_wav = gr.Image(show_label=False, height=340, elem_classes=["frame-wavelet"])
                    with gr.Column(visible=False) as before_after_col:
                        cmp_ba_method = gr.Radio([_BA_METHOD_JPEG, _BA_METHOD_WAVELET],
                                                 value=_BA_METHOD_JPEG, show_label=False)
                        cmp_ba_slider = gr.HTML(cards.empty_state_html(
                            "ÖNCE/SONRA İÇİN SONUÇ YOK", "Önce KARŞILAŞTIR'a basın."))

                    # --- sonuçlar (yalnız KARŞILAŞTIR sonrası) ---
                    with gr.Column(visible=False) as results_group:
                        rate_fairness = gr.HTML()
                        gr.Markdown('<div class="page-section-label section-gap-lg">KALİTE KARŞILAŞTIRMASI</div>')
                        quality_strip = gr.HTML()

                        gr.Markdown('<div class="page-section-label section-gap-lg">SIKIŞTIRMA ÖZETİ</div>')
                        with gr.Row():
                            dct_summary_kpi = gr.HTML()
                            wav_summary_kpi = gr.HTML()

                        with gr.Column(visible=False) as real_jpeg_col:
                            gr.Markdown(f'<div class="page-section-label section-gap-lg" style="color:{METHOD_REAL_JPEG}">GERÇEK JPEG (libjpeg) — çapraz doğrulama (opsiyonel)</div>'
                                       '<span style="font-size:var(--text-sm);color:var(--lab-text-muted)">'
                                       'JPEG / DCT kartındaki entropi tahmininin aksine GERÇEK kodlanmış bayt sayısından hesaplanır.</span>')
                            with gr.Row():
                                out_real_jpeg = gr.Image(show_label=False, height=220, elem_classes=["frame-real-jpeg"])
                                real_jpeg_compact = gr.HTML()

                        gr.Markdown('<div class="page-section-label section-gap-lg">RATE–DISTORTION ANALYSIS</div>')
                        cmp_plot = gr.Plot(label=None, show_label=False)

                        with gr.Accordion("TEKNİK DETAYLAR", open=False):
                            target_full = gr.HTML()
                            with gr.Row():
                                dct_full = gr.HTML()
                                wav_full = gr.HTML()
                            real_jpeg_full = gr.HTML()

                # ANA SAYFA butonu (build_active_image_bar içinde tanımlı) —
                # Karşılaştır sekmesinin toolbar/workspace'i de gizlenir;
                # bu bileşenler yalnız BURADA (Compare tab kapsamında)
                # tanımlı olduğu için build_active_image_bar'ın KENDİ
                # .click() kaydına EK olarak, AYNI butona ikinci bir
                # dinleyici eklenir (Gradio bir bileşene birden fazla
                # olay dinleyicisi bağlanmasına izin verir).
                home_btn.click(
                    lambda: (gr.update(visible=False), gr.update(visible=False)),
                    None, [toolbar, workspace])

                include_real_jpeg.change(toggle_real_jpeg_column, include_real_jpeg, real_jpeg_col)
                active_id.change(
                    _reveal_workspace, [active_img, active_id],
                    [out_orig, toolbar, workspace, results_group,
                     out_dct, out_wav, rate_fairness, quality_strip, cmp_plot,
                     target_full, dct_full, wav_full,
                     dct_summary_kpi, wav_summary_kpi, cmp_ba_slider, cmp_view_mode,
                     out_real_jpeg, real_jpeg_compact, real_jpeg_full])

                # [ Yan Yana ] [ Önce / Sonra ] — mevcut üçlü görünüm KALMAYA
                # devam eder (mega-spec Part 16), Önce/Sonra bir ALTERNATİF.
                # Moda geçildiğinde (veya karşılaştırılacak yöntem
                # değiştiğinde) kaydırıcı, ZATEN hesaplanmış out_orig/out_dct/
                # out_wav değerlerinden okunur — YENİ bir hesaplama YAPILMAZ
                # (Part 12/14).
                cmp_view_mode.change(toggle_cmp_view_mode, cmp_view_mode, [same_side_col, before_after_col]
                                     ).then(render_before_after, [out_orig, out_dct, out_wav, cmp_ba_method],
                                           cmp_ba_slider)
                cmp_ba_method.change(render_before_after, [out_orig, out_dct, out_wav, cmp_ba_method],
                                     cmp_ba_slider)

                _cmp_inputs = [active_img, target_mode, target_value, dct_block_size,
                              include_real_jpeg]
                _cmp_outputs = [out_dct, out_wav, rate_fairness, quality_strip, cmp_plot,
                               target_full, dct_full, wav_full,
                               dct_summary_kpi, wav_summary_kpi,
                               out_real_jpeg, real_jpeg_compact, real_jpeg_full]
                _ba_refresh_inputs = [out_orig, out_dct, out_wav, cmp_ba_method]

                cmp_btn.click(run_main_comparison, _cmp_inputs, _cmp_outputs
                             ).then(lambda: gr.update(visible=True), None, results_group
                             ).then(render_before_after, _ba_refresh_inputs, cmp_ba_slider)
                # Hedef parametreler değişince Compare OTOMATİK yeniden
                # hesaplanır (mega-spec "FINAL INTEGRATION" Part 5: "Target
                # BPP değişti → Compare + matched rate related computations").
                # Sürekli sürüklenen slider (target_value)
                # yalnız BIRAKILDIĞINDA (.release) tetikler — her ara pikselde
                # ağır bir bisection taraması YENİDEN çalıştırmaz; ayrık
                # seçimler (Radio/Dropdown/Checkbox) .change ile anında.
                # Mod değişince ÖNCE slider'ın etiketi/aralığı/varsayılanı
                # yeni moda göre güncellenir (target_mode_ui), SONRA o
                # GÜNCEL değerle karşılaştırma çalışır. Her zincir SONUNDA
                # Önce/Sonra kaydırıcısını da tazeler (Part 12: parametre
                # değişince before/after GÜNCEL sonucu göstermeli) — slider
                # sürüklemesinin KENDİSİ hâlâ backend'e dokunmaz (Part 14).
                target_mode.change(target_mode_ui, target_mode, target_value
                                   ).then(run_main_comparison, _cmp_inputs, _cmp_outputs
                                   ).then(lambda: gr.update(visible=True), None, results_group
                                   ).then(render_before_after, _ba_refresh_inputs, cmp_ba_slider)
                target_value.release(run_main_comparison, _cmp_inputs, _cmp_outputs
                                     ).then(lambda: gr.update(visible=True), None, results_group
                                     ).then(render_before_after, _ba_refresh_inputs, cmp_ba_slider)
                dct_block_size.change(run_main_comparison, _cmp_inputs, _cmp_outputs
                                      ).then(lambda: gr.update(visible=True), None, results_group
                                      ).then(render_before_after, _ba_refresh_inputs, cmp_ba_slider)
                include_real_jpeg.change(run_main_comparison, _cmp_inputs, _cmp_outputs
                                         ).then(lambda: gr.update(visible=True), None, results_group
                                         ).then(render_before_after, _ba_refresh_inputs, cmp_ba_slider)

            # ---------------- SEKME 2: DCT LAB ----------------
            with gr.Tab("DCT Lab"):
                gr.Markdown('<div class="page-section-label">DCT TRANSFORM EXPLORER</div>')
                with gr.Row():
                    # SOL: kaynak görüntü + blok/kalite ayarları (~%32)
                    with gr.Column(scale=1, min_width=360):
                        gr.Markdown('<div class="section-label">KAYNAK GÖRÜNTÜ</div>')
                        dct_source_img = gr.Image(show_label=False, height=340,
                                                  elem_classes=["source-image-frame"])
                        dct_block_line = gr.HTML()

                        gr.Markdown('<div class="section-label section-gap-lg">BLOK AYARLARI</div>')
                        dct_bs = gr.Dropdown([str(b) for b in config.DCT_BLOCK_SIZE_OPTIONS],
                                             value="8", label="Blok boyutu")
                        dct_row = gr.Slider(0, 63, value=0, step=1, label="Satır")
                        dct_col = gr.Slider(0, 63, value=0, step=1, label="Sütun")
                        dct_q = gr.Slider(1, 100, value=50, step=1, label="Quality")
                        dct_btn = gr.Button("BLOĞU İNCELE", variant="primary")

                    # SAĞ: pipeline + ölçüm şeridi + büyük DCT görselleştirmesi (~%68)
                    with gr.Column(scale=2, min_width=640):
                        gr.Markdown(
                            '<div class="section-label">DCT TRANSFORM PIPELINE</div>'
                            '01 PİKSEL → 02 SEVİYE KAYDIRMA → 03 DCT → 04 KUANTALAMA → '
                            '05 ZİGZAG → 06 REKONSTRÜKSİYON'
                        )
                        dct_summary = gr.HTML(cards.empty_state_html(
                            "DCT ANALİZİ İÇİN GÖRÜNTÜ SEÇİN",
                            "Üstteki AKTİF GÖRÜNTÜ çubuğundan bir örnek seçin veya yükleyin."))
                        dct_plot = gr.Plot(show_label=False)
                        dct_zigzag_plot = gr.Plot(show_label=False)
                        with gr.Accordion("Nasıl okunur?", open=False):
                            dct_info = gr.Markdown()

                _dct_outputs = [dct_source_img, dct_block_line, dct_plot, dct_zigzag_plot,
                                dct_summary, dct_info]

                # Görüntüye TIKLAYARAK blok seçimi — satır/sütun slider'larını
                # günceller; onların KENDİ .change() olayı update_dct_analysis'i
                # zaten TAM olarak tetikler (iki yönlü bağlama).
                dct_source_img.select(select_block_from_click, dct_bs, [dct_row, dct_col])

                # Aktif görüntü değiştiğinde: grid'i (satır/sütun aralığı) yeni
                # görüntüye göre yeniden hesapla (varsayılan: merkez blok —
                # mega-spec Part 16) → SONRA TAM analizi yeniden çalıştır. Eski
                # görüntünün blok sonucu yeni görüntü altında ASLA görünmez.
                # Bu zincir aynı zamanda "AUTOMATIC FULL ANALYSIS PIPELINE"nın
                # DCT aşamasıdır (Part 9) — BLOĞU İNCELE'ye basmaya gerek
                # YOKTUR; son adım global durum şeridinde "DCT analizi"ni
                # işaretler.
                active_id.change(
                    dct_grid_shape, [active_img, dct_bs], [dct_row, dct_col]
                ).then(update_dct_analysis, [active_img, dct_bs, dct_row, dct_col, dct_q], _dct_outputs
                ).then(lambda s: _mark_done(s, "dct"), status_state, [status_state, status_html])

                # Blok boyutu değişince de grid yeniden hesaplanır (satır/sütun
                # menzili değişir), SONRA tam analiz.
                dct_bs.change(
                    dct_grid_shape, [active_img, dct_bs], [dct_row, dct_col]
                ).then(update_dct_analysis, [active_img, dct_bs, dct_row, dct_col, dct_q], _dct_outputs)

                # Satır/Sütun/Quality: HERHANGİ biri değiştiğinde TAM yeniden
                # hesaplama (canlı güncelleme — mega-spec Part 11 seçenek A;
                # inspect_block tek bir blok üzerinde çalıştığından maliyeti
                # ihmal edilebilir, bu yüzden buton beklemeye gerek yoktur).
                dct_row.change(update_dct_analysis, [active_img, dct_bs, dct_row, dct_col, dct_q], _dct_outputs)
                dct_col.change(update_dct_analysis, [active_img, dct_bs, dct_row, dct_col, dct_q], _dct_outputs)
                dct_q.change(update_dct_analysis, [active_img, dct_bs, dct_row, dct_col, dct_q], _dct_outputs)
                # Buton: aynı canonical fonksiyon — açık/manuel tetikleme için
                # (ör. görüntü hâlâ None iken kullanıcı deneyip anlamlı bir
                # boş-durum mesajı görmek isterse).
                dct_btn.click(update_dct_analysis, [active_img, dct_bs, dct_row, dct_col, dct_q], _dct_outputs)

            # ---------------- SEKME 3: DWT LAB ----------------
            with gr.Tab("DWT Lab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('<div class="section-label">KAYNAK GÖRÜNTÜ</div>')
                        dwt_source_img = gr.Image(show_label=False, height=220,
                                                  elem_classes=["source-image-frame"])
                        dwt_wav = gr.Dropdown(config.WAVELET_UI_OPTIONS,
                                              value=config.WAVELET_DEFAULT_FILTER,
                                              label="Dalgacık ailesi")
                        dwt_lvl = gr.Slider(1, config.WAVELET_LEVEL_UI_MAX,
                                            value=config.WAVELET_LEVELS, step=1,
                                            label="Ayrıştırma seviyesi")
                        # pywt'nin GERÇEKTEN desteklediği sınır uzatma modları
                        # (mega-spec "FINAL INTEGRATION" Part 27) — fake bir
                        # seçenek eklenmez, wavelet_engine.BOUNDARY_MODES
                        # doğrudan pywt.Modes.modes'tan gelir.
                        dwt_mode = gr.Dropdown(
                            list(wavelet_engine.BOUNDARY_MODES),
                            value=wavelet_engine.DEFAULT_BOUNDARY_MODE,
                            label="Sınır uzatma modu (boundary mode)")
                        # [ Manuel Δ ] [ Hedef BPP ] — iki kontrol ASLA aynı
                        # anda aktif olmaz (mega-spec "DWT LAB sol kontrol
                        # panelini genişlet"). Hedef BPP seçilirse quant_step
                        # GERÇEK bir bisection aramasıyla (match_bpp — Karşı-
                        # laştır sekmesindeki run_wavelet ile AYNI mekanizma)
                        # bulunur; UI'da Hedef BPP ile Tahmini BPP AYRI
                        # gösterilir (özet bar'da), biri diğerini gizlice
                        # ezmez.
                        dwt_rate_mode = gr.Radio(
                            [_DWT_RATE_MODE_MANUAL, _DWT_RATE_MODE_TARGET],
                            value=_DWT_RATE_MODE_MANUAL, show_label=False)
                        dwt_target_bpp = gr.Slider(
                            0.25, 4.00, value=0.5, step=0.05,
                            label="Hedef BPP", visible=False)
                        dwt_step = gr.Slider(0.5, 64.0, value=8.0, step=0.5,
                                             label="Kuantalama adımı (Δ)", visible=True)
                        dwt_btn = gr.Button("AYRIŞTIR", variant="primary")

                    with gr.Column(scale=2):
                        # --- A. TRANSFORM OVERVIEW (birincil) ---
                        dwt_summary = gr.HTML(cards.empty_state_html("BİR GÖRÜNTÜ YÜKLEYİN VE AYRIŞTIR'A BASIN"))
                        gr.Markdown(
                            '<div class="section-label">PIRAMİT KATSAYI HARİTASI</div>'
                            '<span style="font-size:var(--text-sm);color:var(--lab-text-muted)">'
                            'Görüntüleme için kontrast-gerilmiştir (normalize edilmiştir); '
                            'gerçek katsayı değerleri DEĞİŞMEZ.</span>'
                        )
                        dwt_mosaic = gr.Image(show_label=False)
                        # Tıklama ile nokta inceleme (mega-spec "DWT LAB —
                        # TIKLAMA İLE NOKTA İNCELEMEYİ 3 GÖRSELDE AKTİF ET"):
                        # yalnız TIKLAMA (hover değil) küçük, sabit kalan bir
                        # bilgi kutusu açar; tıklanan hücre görselde işaretlenir.
                        dwt_mosaic_info = gr.HTML()
                        # Kuantalamalı Rekonstrüksiyon metrik kartı (PSNR/
                        # SSIM/BPP) kaldırıldı — bu ölçümler zaten Karşılaştır
                        # sekmesinde var; DWT Lab yalnız dönüşümün KENDİSİNİ
                        # (rekonstrüksiyon görüntüsü + fark haritası) gösterir.
                        # İki sütun eşit genişlikte alanı doldurur.
                        with gr.Row():
                            with gr.Column(scale=1):
                                dwt_recon = gr.Image(label="Kuantalamalı rekonstrüksiyon (gri)")
                            with gr.Column(scale=1):
                                dwt_diff = gr.Image(label="Fark haritası (orijinal − rekonstrüksiyon)")
                        # Rekonstrüksiyon VE Fark Haritası PAYLAŞILAN tek bilgi
                        # kutusu kullanır (ikisi de aynı X/Y/Orijinal/
                        # Rekonstrüksiyon/Fark üçlüsünü gösterir).
                        dwt_recon_info = gr.HTML()

                        # --- B. DECOMPOSITION EXPLORER (birincil) — mega-spec
                        # "DWT LAB FIX AND UPGRADE": ağaç + 4 subband kartı
                        # GERÇEK, seviyeye göre doğru eşlenmiş katsayılardan
                        # beslenir (bkz. dwt_select_level / subbands.
                        # decompose_levels). Ağaç artık gr.Image (tıklanabilir
                        # — Part 11); kartlar da tıklanınca seçili bant/piksel
                        # detayını gösterir (Part 12/14).
                        gr.Markdown('<div class="section-label">AYRIŞTIRMA GEZGİNİ</div>')
                        gr.HTML(cards.dwt_band_legend_html())
                        dwt_levels_state = gr.State({})
                        dwt_max_level_state = gr.State(1)
                        dwt_tree_bounds_state = gr.State([])
                        dwt_selected_band_state = gr.State("LL")
                        # Piramit/Rekonstrüksiyon/Fark Haritası tıklama ile
                        # nokta inceleme (mega-spec "DWT LAB — TIKLAMA İLE
                        # NOKTA İNCELEMEYİ 3 GÖRSELDE AKTİF ET") gray/recon
                        # dizilerini SAKLAR — her tıklamada DWT'yi yeniden
                        # HESAPLAMAMAK için (Part 20 ruhu: mevcut "yeniden
                        # hesaplama yapma" ilkesiyle tutarlı).
                        dwt_gray_state = gr.State(None)
                        dwt_recon_state = gr.State(None)
                        with gr.Row():
                            with gr.Column(scale=3):
                                dwt_tree = gr.Image(show_label=False, elem_classes=["dwt-tree-image"])
                            with gr.Column(scale=2):
                                dwt_inspect_lvl = gr.Slider(1, config.WAVELET_LEVELS, value=1, step=1,
                                                            label="İncelenecek seviye")
                                with gr.Row():
                                    dwt_ll = gr.Image(label="LL", height=190)
                                    dwt_lh = gr.Image(label="LH", height=190)
                                with gr.Row():
                                    dwt_hl = gr.Image(label="HL", height=190)
                                    dwt_hh = gr.Image(label="HH", height=190)
                        gr.Markdown('<div class="section-label section-gap-sm">SEÇİLİ BANT DETAYLARI</div>')
                        with gr.Row():
                            with gr.Column(scale=1):
                                dwt_band_details = gr.HTML(cards.empty_state_html(
                                    "Bant seçilmedi", "AYRIŞTIR'a basın veya bir subband kartına tıklayın."))
                            with gr.Column(scale=1):
                                dwt_hist = gr.Plot(show_label=False)

                        # --- C. TEKNİK DETAYLAR (kapalı) ---
                        with gr.Accordion("Teknik Detaylar (filtre katsayıları, ham/kuantalanmış istatistikler, rekonstrüksiyon hatası)", open=False):
                            gr.Markdown('<div class="section-label">2D AYRILABİLİR FİLTRE BANKASI (satır → sütun)</div>')
                            dwt_filt_flow = gr.HTML()
                            dwt_filt = gr.Plot(label="Analiz Filtre Bankası — gerçek dürtü yanıtı")
                            dwt_filt_text = gr.Textbox(label="Filtre katsayıları (gerçek)", lines=4)
                            gr.Markdown(
                                '<div class="section-label">SUBBAND İSTATİSTİKLERİ (HAM DWT katsayıları — '
                                'kuantalama ÖNCESİ; kuantalanmış seyreklik aşağıdaki karttadır)</div>'
                            )
                            dwt_stats = gr.Dataframe(
                                headers=["Bant", "Boyut", "Katsayı Sayısı", "Min", "Max",
                                        "Ortalama", "Std Sapma", "Enerji %", "Sıfır oranı"],
                                interactive=False)
                            dwt_energy = gr.Plot(label="Subband Enerji Dağılımı", show_label=False)
                            with gr.Row():
                                dwt_validation = gr.HTML()
                                dwt_sparsity = gr.HTML()

                # Sıra dwt_reset_results()'ın dönüş sırasıyla BİREBİR eşleşir.
                _dwt_result_outputs = [
                    dwt_summary, dwt_mosaic, dwt_tree, dwt_tree_bounds_state,
                    dwt_ll, dwt_lh, dwt_hl, dwt_hh, dwt_band_details, dwt_selected_band_state, dwt_hist,
                    dwt_recon, dwt_diff, dwt_energy,
                    dwt_stats, dwt_validation, dwt_sparsity,
                    dwt_levels_state, dwt_max_level_state,
                    dwt_mosaic_info, dwt_recon_info, dwt_gray_state, dwt_recon_state,
                ]
                # Sıra run_dwt_explorer()'ın dönüş sırasıyla BİREBİR eşleşir.
                _dwt_full_outputs = [
                    dwt_summary, dwt_mosaic, dwt_tree, dwt_tree_bounds_state,
                    dwt_ll, dwt_lh, dwt_hl, dwt_hh, dwt_band_details, dwt_selected_band_state, dwt_hist,
                    dwt_recon, dwt_diff, dwt_energy,
                    dwt_filt_flow, dwt_filt, dwt_filt_text, dwt_stats,
                    dwt_validation, dwt_sparsity, dwt_inspect_lvl,
                    dwt_levels_state, dwt_max_level_state,
                    dwt_mosaic_info, dwt_recon_info, dwt_gray_state, dwt_recon_state,
                ]
                _dwt_level_outputs = [
                    dwt_tree, dwt_tree_bounds_state, dwt_ll, dwt_lh, dwt_hl, dwt_hh,
                    dwt_band_details, dwt_selected_band_state, dwt_hist,
                ]

                dwt_wav.change(dwt_level_range, [active_img, dwt_wav], dwt_lvl)
                dwt_lvl.change(lambda lvl: gr.update(maximum=int(lvl), value=1), dwt_lvl, dwt_inspect_lvl)

                # "İncelenecek seviye" değişince (slider VEYA ağaç tıklaması
                # ile) DWT YENİDEN HESAPLANMAZ — yalnız önbellekten (
                # dwt_levels_state) okunur (mega-spec Part 5/19).
                dwt_inspect_lvl.change(
                    dwt_select_level, [dwt_levels_state, dwt_inspect_lvl, dwt_selected_band_state],
                    _dwt_level_outputs)
                # Ağaç görselindeki HERHANGİ bir düğüme (LL/LH/HL/HH, her
                # seviye) tıklamak hem o seviyeyi hem o bandı seçer (mega-
                # spec "TÜM SUBBAND NODE'LARINI TAM INTERAKTİF HALE GETİR")
                # — piksel sınırları ax.transData'dan kesin hesaplanır.
                # dwt_tree_click, dwt_select_level'i DOĞRUDAN çağırdığından
                # (yeniden yazılmaz) çıktı listesi de aynı sırayla, başına
                # dwt_inspect_lvl eklenerek kullanılır.
                dwt_tree.select(
                    dwt_tree_click,
                    [dwt_levels_state, dwt_tree_bounds_state, dwt_max_level_state],
                    [dwt_inspect_lvl] + _dwt_level_outputs)
                # Subband görüntülerine (LL/LH/HL/HH) tıklamak (mega-spec
                # "DWT LAB — subband görüntülerine tıklama ile katsayı
                # inceleme", Part 12/14) o bandı seçili yapar, tıklanan
                # pikseldeki GERÇEK ham katsayıyı gösterir, tıklanan hücreyi
                # görüntü üzerinde işaretler (yalnız TIKLAMA — hover HİÇBİR
                # ŞEY yapmaz, Gradio .select() zaten yalnız tıklamada
                # tetiklenir) ve o işaret kullanıcı BAŞKA bir noktaya
                # tıklayana kadar sabit kalır. Diğer 3 önizleme kasıtlı
                # olarak işaretsiz kalır (yeni bant seçilince eski hücre
                # seçimi kaybolur). Ağaç da AYNI seçimle yeniden çizilir
                # (Part 6: tree ↔ kartlar senkron kalsın).
                _dwt_click_outputs = [
                    dwt_selected_band_state, dwt_band_details, dwt_hist,
                    dwt_tree, dwt_tree_bounds_state,
                    dwt_ll, dwt_lh, dwt_hl, dwt_hh,
                ]
                dwt_ll.select(dwt_click_ll, [dwt_levels_state, dwt_inspect_lvl], _dwt_click_outputs)
                dwt_lh.select(dwt_click_lh, [dwt_levels_state, dwt_inspect_lvl], _dwt_click_outputs)
                dwt_hl.select(dwt_click_hl, [dwt_levels_state, dwt_inspect_lvl], _dwt_click_outputs)
                dwt_hh.select(dwt_click_hh, [dwt_levels_state, dwt_inspect_lvl], _dwt_click_outputs)

                # Piramit Katsayı Haritası / Kuantalamalı Rekonstrüksiyon /
                # Fark Haritası'na tıklama ile nokta inceleme (mega-spec
                # "DWT LAB — TIKLAMA İLE NOKTA İNCELEMEYİ 3 GÖRSELDE AKTİF
                # ET") — yalnız TIKLAMA (Gradio .select() zaten hover'da
                # tetiklenmez), seçim başka bir noktaya tıklayana kadar
                # sabit kalır. DWT YENİDEN HESAPLANMAZ — mozaik zaten
                # ekrandaki dwt_mosaic'ten, gray/recon ise run_dwt_
                # explorer'ın doldurduğu dwt_gray_state/dwt_recon_state'ten
                # okunur.
                dwt_mosaic.select(
                    dwt_mosaic_click, [dwt_mosaic, dwt_levels_state, dwt_max_level_state],
                    [dwt_mosaic_info, dwt_mosaic])
                dwt_recon.select(
                    dwt_recon_click, [dwt_gray_state, dwt_recon_state],
                    [dwt_recon_info, dwt_recon])
                dwt_diff.select(
                    dwt_diff_click, [dwt_gray_state, dwt_recon_state],
                    [dwt_recon_info, dwt_diff])

                # Aktif görüntü değiştiğinde: kaynak önizlemeyi tazele, seviye
                # menzilini yeni görüntüye clamp et, VE önceki görüntünün
                # mozaik/ağaç/subband/rekonstrüksiyon/istatistik/önbellek
                # çıktılarını temizle (mega-spec Part 17 — "DWT old result
                # yeni image altında kalmıyor"). Dalgacık seçimi/sınır modu/
                # filtre bankası bilgisi görüntüden bağımsız olduğu için
                # KORUNUR.
                active_id.change(lambda img: img, active_img, dwt_source_img)
                active_id.change(dwt_level_range, [active_img, dwt_wav], dwt_lvl)
                active_id.change(dwt_reset_results, None, _dwt_result_outputs)

                _dwt_inputs = [active_img, dwt_wav, dwt_lvl, dwt_step, dwt_inspect_lvl, dwt_mode,
                              dwt_rate_mode, dwt_target_bpp]
                dwt_btn.click(run_dwt_explorer, _dwt_inputs, _dwt_full_outputs)
                # Dalgacık ailesi/sınır modu/hız modu değişince DWT OTOMATİK
                # yeniden hesaplanır (mega-spec Part 5/26/27) — ayrık
                # Dropdown/Radio seçimleri olduğundan .change ile anında;
                # seviye/kuantalama adımı/hedef bpp SÜREKLİ slider'lardır,
                # yalnız BIRAKILDIĞINDA (.release) tetiklenir (her ara
                # pikselde ağır decomposition YENİDEN çalıştırmaz).
                dwt_wav.change(run_dwt_explorer, _dwt_inputs, _dwt_full_outputs)
                dwt_mode.change(run_dwt_explorer, _dwt_inputs, _dwt_full_outputs)
                dwt_lvl.release(run_dwt_explorer, _dwt_inputs, _dwt_full_outputs)
                dwt_step.release(run_dwt_explorer, _dwt_inputs, _dwt_full_outputs)
                # [ Manuel Δ ] [ Hedef BPP ] görünürlüğü ÖNCE güncellenir,
                # SONRA güncel moda göre yeniden hesaplanır (target_mode_ui
                # ile AYNI zincirleme desen — Compare sekmesindeki hedef
                # modu değişince kullanılan desenin BİREBİR aynısı).
                dwt_rate_mode.change(toggle_dwt_rate_mode, dwt_rate_mode, [dwt_target_bpp, dwt_step]
                                     ).then(run_dwt_explorer, _dwt_inputs, _dwt_full_outputs)
                dwt_target_bpp.release(run_dwt_explorer, _dwt_inputs, _dwt_full_outputs)

            # ---------------- SEKME 4: SEMANTİK ROI ----------------
            with gr.Tab("Semantik ROI"):
                gr.HTML(
                    '<div class="page-section-label">SEMANTİK-FARKINDA BİT TAHSİSİ</div>'
                    '<p style="font-size:var(--text-md);color:var(--lab-text-secondary);margin-top:-4px">'
                    'Aynı toplam bit bütçesi, farklı uzamsal kalite tahsisi.</p>'
                )
                sem_instances_state = gr.State([])
                sem_roi_center_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('<div class="section-label">KAYNAK GÖRÜNTÜ</div>')
                        sem_source_img = gr.Image(show_label=False, height=300,
                                                  elem_classes=["source-image-frame"])

                        gr.Markdown('<div class="section-label section-gap-lg">ROI SEÇİMİ</div>')
                        sem_mode = gr.Radio(["Otomatik (YOLO)", "Manuel ROI"],
                                           value="Otomatik (YOLO)", label="Mod")
                        with gr.Group(visible=True) as sem_auto_group:
                            sem_detect_btn = gr.Button("Nesneleri Tespit Et", variant="secondary")
                            sem_detect_info = gr.HTML(cards.empty_state_html(
                                "Nesneler henüz tespit edilmedi", "'Nesneleri Tespit Et' ile başlayın."))
                            sem_object_checklist = gr.CheckboxGroup(
                                choices=[], value=[], label="Tespit edilen nesneler (dahil/hariç)")
                        with gr.Group(visible=False) as sem_manual_group:
                            gr.Markdown(
                                '<span style="font-size:var(--text-sm);color:var(--lab-text-muted)">'
                                'Kaynak görüntüye tıklayarak ROI merkezini seçin.</span>')
                            sem_roi_w = gr.Slider(16, 512, value=128, step=8, label="ROI genişliği (px)")
                            sem_roi_h = gr.Slider(16, 512, value=128, step=8, label="ROI yüksekliği (px)")

                        gr.Markdown('<div class="section-label section-gap-lg">HEDEF ORAN</div>')
                        sem_bpp = gr.Slider(0.15, 1.5, value=0.40, step=0.05, label="Hedef bpp")
                        with gr.Accordion("Gelişmiş Ayarlar", open=False):
                            sem_engine = gr.Radio(
                                ["DCT (JPEG mantığı)", "Wavelet (JPEG2000 mantığı)"],
                                value="DCT (JPEG mantığı)", label="Sıkıştırma motoru",
                            )
                            sem_coarse = gr.Slider(2, 12, value=config.ROI_BG_COARSENESS, step=1,
                                                   label="Arka plan kabalık çarpanı")
                        sem_btn = gr.Button("SIKIŞTIR", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('<div class="frame-caption" style="text-align:center">ROI MASKESİ</div>')
                                sem_mask = gr.Image(show_label=False, elem_classes=["frame-neutral"])
                            with gr.Column():
                                gr.Markdown('<div class="frame-caption" style="text-align:center">BASELINE (UNIFORM)</div>')
                                sem_base = gr.Image(show_label=False, elem_classes=["frame-neutral"])
                            with gr.Column():
                                gr.Markdown(f'<div class="frame-caption" style="text-align:center;color:{ACCENT_GREEN}">SEMANTİK ROI</div>')
                                sem_sem = gr.Image(show_label=False, elem_classes=["frame-wavelet"])
                        sem_budget = gr.HTML(cards.empty_state_html("GÖRÜNTÜ YÜKLEYİN VE SIKIŞTIR'A BASIN"))
                        with gr.Row():
                            sem_gain = gr.HTML()
                            sem_tradeoff = gr.HTML()
                            # Global PSNR artık paragrafın içine gizli
                            # değil, ayrı kompakt bir kart (Part 5).
                            sem_global_card = gr.HTML()
                        sem_global = gr.Markdown()

                sem_mode.change(sem_toggle_mode, sem_mode, [sem_auto_group, sem_manual_group])
                # KAYNAK GÖRÜNTÜ HER ZAMAN ham/işlenmemiş aktif görüntüdür
                # (mega-spec "FINAL PRE-PRESENTATION QA" Part 3) — YOLO
                # overlay'i BURAYA değil ROI MASKESİ (sem_mask) kartına
                # yazılır. Önceden sem_detect_objects'in 4. dönüş değeri
                # (overlay preview) yanlışlıkla sem_source_img'e bağlıydı.
                sem_detect_btn.click(
                    sem_detect_objects, active_img,
                    [sem_instances_state, sem_object_checklist, sem_detect_info, sem_mask, sem_detect_btn])
                sem_roi_w.change(sem_update_manual_overlay,
                                 [active_img, sem_roi_center_state, sem_roi_w, sem_roi_h], sem_source_img)
                sem_roi_h.change(sem_update_manual_overlay,
                                 [active_img, sem_roi_center_state, sem_roi_w, sem_roi_h], sem_source_img)

                # sem_recompute (hata-toleranslı sarmalayıcı — bkz. tanımı)
                # PASİF tetiklenen tüm yeniden hesaplamalarda kullanılır;
                # active_id ile stale-image korumalıdır (Part 3/17).
                _sem_inputs = [active_img, active_id, sem_engine, sem_bpp, sem_coarse, sem_mode,
                              sem_instances_state, sem_object_checklist,
                              sem_roi_center_state, sem_roi_w, sem_roi_h]
                _sem_outputs = [sem_mask, sem_base, sem_sem, sem_budget, sem_gain, sem_tradeoff,
                               sem_global_card, sem_global]

                # Nesne checkbox'ları / hedef bpp / motor / kabalık değişince
                # semantik sıkıştırma OTOMATİK yeniden hesaplanır (mega-spec
                # "FINAL INTEGRATION" Part 5: "Semantic object checkbox
                # değişti → Semantic compression only") — önce önizleme
                # maskesi tazelenir (sem_update_auto_preview, ucuz), SONRA
                # gerçek sıkıştırma. Sürekli slider'lar (.release).
                sem_object_checklist.change(
                    sem_update_auto_preview, [active_img, sem_instances_state, sem_object_checklist],
                    sem_mask
                ).then(sem_recompute, _sem_inputs, _sem_outputs)
                sem_engine.change(sem_recompute, _sem_inputs, _sem_outputs)
                sem_bpp.release(sem_recompute, _sem_inputs, _sem_outputs)
                sem_coarse.release(sem_recompute, _sem_inputs, _sem_outputs)
                # Manuel ROI: tıklayarak merkez seçmek + boyut slider'larını
                # BIRAKMAK da gerçek sıkıştırmayı tetikler (yalnız görsel
                # dikdörtgen değil — mega-spec Part 39).
                sem_source_img.select(sem_roi_click, active_img, sem_roi_center_state).then(
                    sem_update_manual_overlay, [active_img, sem_roi_center_state, sem_roi_w, sem_roi_h],
                    sem_source_img
                ).then(sem_recompute, _sem_inputs, _sem_outputs)
                sem_roi_w.release(sem_recompute, _sem_inputs, _sem_outputs)
                sem_roi_h.release(sem_recompute, _sem_inputs, _sem_outputs)

                # Aktif görüntü değiştiğinde: kaynak önizlemeyi tazele VE
                # kritik stale-state hatasını düzelt — önceki görüntünün YOLO
                # tespitleri/nesne listesi/manuel ROI/maske/tüm sonuç kartları
                # TAMAMEN temizlenir (mega-spec Part 2/6/26-29).
                active_id.change(lambda img: img, active_img, sem_source_img)
                active_id.change(
                    sem_reset_state, None,
                    [sem_instances_state, sem_object_checklist, sem_detect_info, sem_roi_center_state,
                     sem_mask, sem_base, sem_sem, sem_budget, sem_gain, sem_tradeoff, sem_global_card,
                     sem_global, sem_detect_btn])

                sem_btn.click(
                    run_semantic_pipeline,
                    [active_img, sem_engine, sem_bpp, sem_coarse, sem_mode, sem_instances_state,
                     sem_object_checklist, sem_roi_center_state, sem_roi_w, sem_roi_h],
                    [sem_mask, sem_base, sem_sem, sem_budget, sem_gain, sem_tradeoff, sem_global_card,
                     sem_global])

            # ---------------- SEKME 5: TEORİ ----------------
            with gr.Tab("Teori"):
                gr.HTML(_theory_html())

        # =====================================================================
        # OTOMATİK TAM ANALİZ ORKESTRASYONU (mega-spec "AUTOMATIC FULL
        # ANALYSIS PIPELINE") — TÜM sekmeler tanımlandıktan SONRA, tek bir
        # yerde kayıt edilir ki Gradio'nun aynı tetikleyicideki (active_id)
        # dinleyicileri KAYIT SIRASINA göre ÇALIŞTIRDIĞI garanti edilsin:
        #
        #   1) _status_reset            (yukarıda, en önce kayıt edildi)
        #   2) Compare workspace reveal + temizleme (Karşılaştır sekmesinde)
        #   3) DCT canlı analiz (DCT Lab sekmesinde) → "dct" işaretlenir
        #   4) DWT/Semantic RESET (kendi sekmelerinde)
        #   5) DWT otomatik analiz          → "dwt" işaretlenir
        #   6) Compare otomatik analiz + RD → "compare"+"rd" işaretlenir
        #   7) Semantic otomatik tespit     → "semantic_detect" işaretlenir
        #   8) Semantic otomatik sıkıştırma → "semantic_compress" işaretlenir,
        #      durum şeridi "ANALİZ HAZIR"a çöker.
        #
        # Hiçbir algoritma burada YENİDEN YAZILMAZ (Part 29) — yalnız MEVCUT
        # run_dwt_explorer/run_main_comparison/sem_detect_objects/
        # run_semantic_pipeline fonksiyonları image_id-güvenli sarmalayıcılar
        # (dwt_auto_analysis/compare_auto_analysis/sem_auto_detect/
        # sem_auto_compress) üzerinden çağrılır.
        # =====================================================================
        active_id.change(
            dwt_auto_analysis,
            [active_img, active_id, dwt_wav, dwt_lvl, dwt_step, dwt_mode,
             dwt_rate_mode, dwt_target_bpp],
            _dwt_full_outputs,
        ).then(lambda s: _mark_done(s, "dwt"), status_state, [status_state, status_html])

        active_id.change(
            compare_auto_analysis,
            [active_img, active_id, target_mode, target_value, dct_block_size,
             include_real_jpeg],
            [out_dct, out_wav, rate_fairness, quality_strip, cmp_plot,
             target_full, dct_full, wav_full,
             dct_summary_kpi, wav_summary_kpi,
             out_real_jpeg, real_jpeg_compact, real_jpeg_full],
        ).then(lambda: gr.update(visible=True), None, results_group
        ).then(render_before_after, [out_orig, out_dct, out_wav, cmp_ba_method], cmp_ba_slider
        ).then(lambda s: _mark_done(s, "compare", "rd"), status_state, [status_state, status_html])

        active_id.change(
            sem_auto_detect, [active_img, active_id],
            [sem_instances_state, sem_object_checklist, sem_detect_info, sem_mask, sem_detect_btn],
        ).then(lambda s: _mark_done(s, "semantic_detect"), status_state, [status_state, status_html]
        ).then(
            sem_auto_compress,
            [active_img, active_id, sem_instances_state, sem_object_checklist,
             sem_engine, sem_bpp, sem_coarse],
            [sem_mask, sem_base, sem_sem, sem_budget, sem_gain, sem_tradeoff, sem_global_card, sem_global],
        ).then(lambda s: _mark_done(s, "semantic_compress"),
              status_state, [status_state, status_html])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
