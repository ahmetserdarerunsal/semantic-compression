# -*- coding: utf-8 -*-
"""Web arayüzü: semantik-farkında sıkıştırma + JPEG/DCT vs Wavelet/DWT
karşılaştırma + DCT/DWT Explorer'lar (tek Gradio uygulaması, çoklu sekme).

Kullanım:
    python app.py
Sonra tarayıcıda http://127.0.0.1:7860 açılır.

Sekmeler:
  1) JPEG vs Wavelet Karşılaştırma — aynı hedef bpp'de üç (+ opsiyonel
     gerçek JPEG2000) motor, yan yana, gerçek zamanlı RD grafiğiyle.
  2) DCT Explorer — tek bir bloğun orijinal->DCT->kuantalama->rekonstrüksiyon
     adımlarını gerçek görüntü verisinden gösterir.
  3) DWT Explorer — gerçek katsayılardan piramit mozaik, dyadic ağaç,
     filtre bankası ve kayıpsız-rekonstrüksiyon testi.
  4) Semantik-Farkında Sıkıştırma — YOLO-seg ROI'li bit yönlendirme
     (projenin özgün özelliği; değiştirilmeden korunmuştur).
  5) Teori / Nasıl Çalışır — kısa DCT/DWT açıklaması.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import gradio as gr
import numpy as np

import config
import src.compare as compare
from src.engines import dct_engine, wavelet_engine
from src.engines.dct_engine import JPEG_LUMA_QTABLE
from src.engines.jpeg2000_engine import JPEG2000_AVAILABLE
from src.engines.wavelet_engine import decompose_for_viz, max_decomposition_level
from src.metrics.quality import evaluate, mse, psnr, ratio_to_bpp
from src.roi.bit_allocation import mask_to_block_importance, match_bpp
from src.semantic.importance_map import get_importance_mask
from src.viz import dct_block, subbands
from src.viz.plots import overlay_mask, rd_curve_figure

MAX_SIDE = 768  # büyük yüklemeler hız için küçültülür


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


def _metrics_rows(m: dict, is_real_codec: bool = False) -> list[list[str]]:
    psnr_str = "∞ (kayıpsız)" if not np.isfinite(m["psnr"]) else f"{m['psnr']:.2f}"
    size_label = "Sıkıştırılmış boyut (gerçek)" if is_real_codec else "Sıkıştırılmış boyut (entropi tahmini)"
    return [
        ["PSNR (dB)", psnr_str],
        ["SSIM", f"{m['ssim']:.4f}"],
        ["MSE", f"{m['mse']:.3f}"],
        ["BPP (gerçekleşen)", f"{m['bpp']:.3f}"],
        ["Sıkıştırma oranı", f"{m['compression_ratio']:.2f} : 1"],
        [size_label, f"{m['compressed_size_bytes'] / 1024.0:.2f} KB"],
    ]


# =============================================================================
# SEKME 1 — JPEG vs Wavelet karşılaştırma (aynı hedef bpp)
# =============================================================================
_RD_TARGETS = [0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.8, 2.5]


def _target_to_bpp(mode: str, value: float) -> float:
    return float(value) if mode == "Hedef bpp" else ratio_to_bpp(float(value))


def update_wavelet_level_range(image, wavelet):
    if image is None:
        return gr.update(maximum=config.WAVELET_LEVEL_UI_MAX)
    image = _prepare(image)
    cap = min(max_decomposition_level(image.shape[:2], wavelet), config.WAVELET_LEVEL_UI_MAX)
    return gr.update(maximum=cap, value=min(config.WAVELET_LEVELS, cap))


def run_main_comparison(
    image, target_mode, target_value, dct_block_size, wavelet, wavelet_level,
    include_jp2k, progress=gr.Progress(),
):
    if image is None:
        raise gr.Error("Önce bir görüntü yükleyin veya örneklerden seçin.")
    image = _prepare(image)
    target_bpp = _target_to_bpp(target_mode, target_value)
    dct_block_size = int(dct_block_size)
    wavelet_level = int(wavelet_level)

    progress(0.10, desc="JPEG/DCT hedef bpp'ye eşleniyor (bisection)…")
    dct_res = compare.run_dct(image, target_bpp, dct_block_size)

    progress(0.45, desc="Wavelet/DWT hedef bpp'ye eşleniyor (bisection)…")
    wav_res = compare.run_wavelet(image, target_bpp, wavelet, wavelet_level)

    jp2k_img, jp2k_rows, jp2k_md = None, [], ""
    if include_jp2k:
        if JPEG2000_AVAILABLE:
            progress(0.65, desc="Gerçek JPEG2000 (OpenJPEG) kodlanıyor…")
            jp2k_res = compare.run_jpeg2000(image, target_bpp)
            jp2k_img = jp2k_res.recon
            jp2k_rows = _metrics_rows(jp2k_res.metrics, is_real_codec=True)
            jp2k_md = f"**{jp2k_res.label}**\n\n{jp2k_res.param_label}\n\n{jp2k_res.note}"
        else:
            jp2k_md = ("Bu ortamda gerçek bir JPEG2000 kodeği (OpenJPEG) bulunamadı; "
                       "bu sütun atlandı. Sahte/simüle bir sonuç GÖSTERİLMEZ.")

    progress(0.85, desc="Rate-Distortion taraması (gerçek ölçümler)…")
    targets = sorted(set(_RD_TARGETS + [round(target_bpp, 3)]))
    curves = compare.rd_sweep(image, targets, wavelet, wavelet_level, dct_block_size)
    fig = rd_curve_figure(
        curves,
        f"Rate-Distortion — {image.shape[1]}x{image.shape[0]} görüntü "
        f"(yıldız = güncel hedef {target_bpp:.2f} bpp)",
        highlight={
            "JPEG / DCT": (dct_res.metrics["bpp"], dct_res.metrics["psnr"]),
            f"Wavelet / DWT ({wavelet})": (wav_res.metrics["bpp"], wav_res.metrics["psnr"]),
        },
    )

    verdict = (
        f"**Hedef: {target_bpp:.3f} bpp.** Gerçekleşen (bisection ile eşlenen, "
        f"tam hedefe eşit OLMAK ZORUNDA değildir) — "
        f"JPEG/DCT: **{dct_res.metrics['bpp']:.3f} bpp**, {dct_res.metrics['psnr']:.2f} dB PSNR "
        f"({dct_res.param_label})  ·  "
        f"Wavelet/DWT: **{wav_res.metrics['bpp']:.3f} bpp**, {wav_res.metrics['psnr']:.2f} dB PSNR "
        f"({wav_res.param_label})."
    )

    return (
        image, dct_res.recon, wav_res.recon,
        _metrics_rows(dct_res.metrics), _metrics_rows(wav_res.metrics),
        fig, verdict, jp2k_img, jp2k_rows, jp2k_md,
    )


# =============================================================================
# SEKME 2 — DCT Explorer
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


def run_dct_explorer(image, block_size, row, col, quality):
    if image is None:
        raise gr.Error("Önce bir görüntü yükleyin.")
    image = _prepare(image)
    gray = _to_gray(image)
    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    result = dct_block.inspect_block(gray, int(block_size), int(row), int(col),
                                     float(quality), base_table)
    fig = dct_block.block_pipeline_figure(result)
    nh, nw = result["grid_shape"]
    info = (
        f"**Blok ızgarası:** {nh} x {nw} blok (görüntü {gray.shape[1]}x{gray.shape[0]}, "
        f"blok boyutu {int(block_size)}x{int(block_size)})  ·  "
        f"seçili blok konumu (satır={result['position'][0]}, sütun={result['position'][1]})\n\n"
        f"**DC katsayısı** = {result['dc']:.1f} (bloğun ortalama parlaklığı, "
        f"seviye-kaydırmalı ölçekte). Sol-üst = düşük uzamsal frekans, "
        f"sağ-alt yönü = artan frekans. Kuantalama sonrası {result['n_total'] - result['n_nonzero']}/"
        f"{result['n_total']} katsayı sıfırlandı (yüksek frekanslar küçük/sıfır oluyor)."
    )
    note = ("*(Not: 8x8 dışındaki blok boyutları standart JPEG'in bir parçası değildir; "
            "bu projede eğitim amaçlı bir DCT-boyutu deneyidir — kuantalama tablosu "
            "8x8 standart JPEG tablosundan enterpolasyonla türetilir.)*"
            if int(block_size) != 8 else
            "*(8x8: baseline JPEG standardının kullandığı gerçek blok boyutu ve "
            "gerçek luminance kuantalama tablosu.)*")
    return fig, info + "\n\n" + note


# =============================================================================
# SEKME 3 — DWT Explorer
# =============================================================================
def dwt_level_range(image, wavelet):
    if image is None:
        return gr.update(maximum=config.WAVELET_LEVEL_UI_MAX)
    image = _prepare(image)
    cap = min(max_decomposition_level(image.shape[:2], wavelet), config.WAVELET_LEVEL_UI_MAX)
    return gr.update(maximum=cap, value=min(config.WAVELET_LEVELS, cap))


def run_dwt_explorer(image, wavelet, levels, quant_step):
    if image is None:
        raise gr.Error("Önce bir görüntü yükleyin.")
    image = _prepare(image)
    gray = _to_gray(image)
    levels = min(int(levels), max_decomposition_level(gray.shape, wavelet))

    coeffs = decompose_for_viz(gray, wavelet, levels)
    mosaic = subbands.pyramid_display_image(coeffs)
    grid = subbands.single_level_grid(coeffs, 1)
    tree_fig = subbands.dyadic_tree_figure(levels)
    filt_fig = subbands.filter_bank_figure(wavelet)
    info = subbands.filter_bank_info(wavelet)

    stats = subbands.coeff_stats(coeffs)
    stats_rows = [[r["label"], r["shape"], f"{r['min']:.1f}", f"{r['max']:.1f}",
                  f"{r['mean']:.2f}", f"{r['zero_pct']:.1f}%"] for r in stats]

    # Kayıpsız rekonstrüksiyon (kuantalama yok): teorik doğrulama
    import pywt
    lossless = pywt.waverec2(coeffs, wavelet)[:gray.shape[0], :gray.shape[1]] + 128.0
    lossless = np.clip(lossless, 0, 255)
    lossless_err = float(np.abs(lossless - gray).max())
    lossless_mse = mse(gray, lossless)

    # Kuantalamalı (lossy) rekonstrüksiyon: karşılaştırma motoruyla aynı yol
    recon, bpp = wavelet_engine.compress_channel(gray, float(quant_step), wavelet, levels)
    lossy_psnr = psnr(gray, recon)
    lossy_mse = mse(gray, recon)

    verdict = (
        f"**Dalgacık:** {info['name']} ({info['family']}), "
        f"{'ortogonal' if info['orthogonal'] else 'biortogonal'}, "
        f"analiz filtre uzunluğu = **{info['dec_len']} tap**  ·  **Seviye:** {levels} "
        f"(bu görüntü/dalgacık için azami geçerli seviye: "
        f"{max_decomposition_level(gray.shape, wavelet)})\n\n"
        f"**Kayıpsız rekonstrüksiyon testi** (kuantalama YOK — yalnız DWT→IDWT): "
        f"azami mutlak hata = {lossless_err:.2e}, MSE = {lossless_mse:.2e} "
        f"→ sayısal hassasiyet düzeyinde, dönüşüm tersinirliği doğrulanmıştır.\n\n"
        f"**Kuantalamalı rekonstrüksiyon** (base_step={float(quant_step):.2f}): "
        f"bpp={bpp:.3f}, PSNR={lossy_psnr:.2f} dB, MSE={lossy_mse:.3f}"
    )
    filt_text = (
        f"Analiz alçak-geçiren (h): {info['dec_lo']}\n\n"
        f"Analiz yüksek-geçiren (g): {info['dec_hi']}"
    )
    return (mosaic, grid, tree_fig, filt_fig, stats_rows, verdict, filt_text,
           np.clip(recon, 0, 255).astype(np.uint8))


# =============================================================================
# SEKME 4 — Semantik-Farkında Sıkıştırma (mevcut özellik, değiştirilmeden)
# =============================================================================
def run_semantic_pipeline(
    image: np.ndarray | None,
    engine: str,
    target_bpp: float,
    bg_coarseness: float,
    progress: gr.Progress = gr.Progress(),
):
    """UI'nin tek giriş noktası: maske + baseline + semantic + metrik tablosu."""
    if image is None:
        raise gr.Error("Önce bir görüntü yükleyin veya örneklerden seçin.")
    image = _prepare(image)

    progress(0.1, desc="YOLO-seg maske çıkarılıyor…")
    mask, labels = get_importance_mask(image)
    if not mask.any():
        raise gr.Error(
            "YOLO-seg bu görüntüde nesne bulamadı; ROI uygulanamıyor. "
            "İnsan/araç/hayvan gibi COCO nesneleri içeren bir görüntü deneyin."
        )
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

    table = [
        ["bpp (gerçek)", f"{base[1]:.3f}", f"{sem[1]:.3f}"],
        ["Foreground PSNR (dB)", f"{mb['fg_psnr']:.2f}", f"{ms['fg_psnr']:.2f}"],
        ["Foreground SSIM", f"{mb['fg_ssim']:.4f}", f"{ms['fg_ssim']:.4f}"],
        ["Background PSNR (dB)", f"{mb['bg_psnr']:.2f}", f"{ms['bg_psnr']:.2f}"],
        ["Global PSNR (dB)", f"{mb['psnr']:.2f}", f"{ms['psnr']:.2f}"],
    ]
    verdict = (
        f"Tespit: {', '.join(sorted(set(labels)))} — maske kapsama %{mask.mean()*100:.0f}. "
        f"Aynı bütçede foreground kazancı: **{ms['fg_psnr']-mb['fg_psnr']:+.2f} dB** "
        f"(background bedeli {ms['bg_psnr']-mb['bg_psnr']:+.2f} dB)."
    )
    return overlay, base[0], sem[0], table, verdict


# =============================================================================
# UI
# =============================================================================
_THEORY_MD = """
### DCT nasıl çalışır (JPEG mantığı)

```
Görüntü → 8x8 (veya seçilen boyutta) bloklara böl → seviye kaydır (-128)
        → 2B DCT → kuantalama (yüksek frekanslar kabalaşır/sıfırlanır)
        → zigzag + DPCM/RLE sembolleri → entropi kodlama → sıkıştırılmış veri
```

- **Blok tabanlı** dönüşüm; güçlü enerji sıkışması (az sayıda büyük katsayı).
- Yüksek sıkıştırmada **blok artefaktları** (8x8 sınırlarında görünür kenarlar) oluşabilir.
- DC katsayısı = bloğun ortalama parlaklığı; sol-üstten sağ-alta uzamsal frekans artar.

### DWT nasıl çalışır (JPEG2000 mantığı)

```
Görüntü → alçak-geçiren + yüksek-geçiren filtreleme (satır, sonra sütun)
        → 2 kat alt örnekleme → LL / LH / HL / HH
        → LL'yi recursively tekrar ayrıştır (dyadic ağaç)
        → katsayı kuantalama/eşikleme → kodlama → sıkıştırılmış veri
```

- **Çok-çözünürlüklü**: tek bir global dönüşüm, blok sınırı yok → klasik 8x8
  bloklaşma artefaktı görülmez.
- LL = yaklaşım (düşük frekans), LH/HL/HH = yatay/dikey/köşegen detay.
- JPEG2000'in temel prensibiyle aynı dönüşüm ailesini kullanır (bu projedeki
  motor JPEG2000'in KENDİSİ değildir — bkz. Sınırlamalar).

### Önemli ayrımlar

- **DCT ≠ JPEG**, **DWT ≠ JPEG2000**: transform, sıkıştırmanın yalnız bir
  adımıdır; kuantalama + entropi kodlama olmadan "sıkıştırma" tamamlanmaz.
- **PSNR ≠ algısal kalite**: sayısal yakınlığı ölçer, mükemmel bir insan
  kalite algısı modeli değildir.
- **Hedef bpp ≠ gerçekleşen bpp**: özel motorlarda bisection ile YAKLAŞIK
  tutturulur (bkz. karşılaştırma ekranındaki "gerçekleşen" değerler).

### Sınırlamalar (dürüstçe belirtilmeli)

- `src/engines/dct_engine.py` ve `src/engines/wavelet_engine.py` **gerçek bir
  bit akışı yazmaz**; bpp, order-0 Shannon entropi tahminine dayanır
  (`src/engines/entropy.py`). Bu nedenle mutlak bpp değerleri gerçek bir
  kodlayıcınınkinden hafifçe sapabilir, ama motor içi kıyaslamalar tutarlıdır.
- Karşılaştırma ekranındaki "gerçek JPEG2000" seçeneği ayrıdır: Pillow'un
  OpenJPEG kodeği üzerinden gerçek bir bitstream üretir/çözer (bu ortamda
  mevcutsa).
"""


def build_ui() -> gr.Blocks:
    examples = sorted(str(p) for cat in config.IMAGE_CATEGORIES
                      for p in (config.DATA_DIR / cat).glob("*.png"))

    with gr.Blocks(title="Semantik-Farkında Görüntü Sıkıştırma") as demo:
        gr.Markdown(
            "# Görüntü Sıkıştırma: DCT / JPEG vs Wavelet / DWT\n"
            "Dijital Sinyal İşleme — sıkıştırma dönüşümlerinin karşılaştırmalı, "
            "interaktif incelemesi."
        )

        with gr.Tabs():
            # ---------------- SEKME 1 ----------------
            with gr.Tab("JPEG vs Wavelet Karşılaştırma"):
                with gr.Row():
                    with gr.Column(scale=1):
                        cmp_inp = gr.Image(label="Girdi görüntüsü", type="numpy")
                        target_mode = gr.Radio(["Hedef bpp", "Hedef Sıkıştırma Oranı"],
                                               value="Hedef bpp", label="Hedef tür")
                        target_value = gr.Slider(0.1, 4.0, value=0.5, step=0.05,
                                                 label="Hedef değer (bpp veya oran)")
                        gr.Markdown("**JPEG/DCT ayarları**")
                        dct_block_size = gr.Dropdown(
                            [str(b) for b in config.DCT_BLOCK_SIZE_OPTIONS],
                            value=str(config.DCT_BLOCK_SIZE), label="DCT blok boyutu")
                        gr.Markdown("**Wavelet/DWT ayarları**")
                        wavelet_dd = gr.Dropdown(
                            config.WAVELET_UI_OPTIONS, value=config.WAVELET_DEFAULT_FILTER,
                            label="Dalgacık ailesi")
                        wavelet_level = gr.Slider(1, config.WAVELET_LEVEL_UI_MAX,
                                                  value=config.WAVELET_LEVELS, step=1,
                                                  label="Ayrıştırma seviyesi")
                        include_jp2k = gr.Checkbox(
                            value=False,
                            label=f"Gerçek JPEG2000 kodekle de karşılaştır "
                                 f"({'mevcut' if JPEG2000_AVAILABLE else 'BU ORTAMDA YOK'})",
                            interactive=JPEG2000_AVAILABLE)
                        cmp_btn = gr.Button("Karşılaştır", variant="primary")
                        if examples:
                            gr.Examples(examples=examples, inputs=cmp_inp, label="Örnek görüntüler")
                    with gr.Column(scale=2):
                        cmp_verdict = gr.Markdown()
                        with gr.Row():
                            out_orig = gr.Image(label="Orijinal")
                            out_dct = gr.Image(label="JPEG / DCT")
                            out_wav = gr.Image(label="Wavelet / DWT")
                        with gr.Row():
                            gr.Markdown("")
                            dct_table = gr.Dataframe(headers=["Metrik", "Değer"],
                                                     label="JPEG/DCT metrikleri", interactive=False)
                            wav_table = gr.Dataframe(headers=["Metrik", "Değer"],
                                                     label="Wavelet/DWT metrikleri", interactive=False)
                        with gr.Row(visible=True):
                            out_jp2k = gr.Image(label="JPEG2000 (gerçek kodek)")
                            jp2k_table = gr.Dataframe(headers=["Metrik", "Değer"],
                                                      label="JPEG2000 metrikleri", interactive=False)
                        jp2k_note = gr.Markdown()
                        cmp_plot = gr.Plot(label="Rate-Distortion Grafiği (gerçek ölçümler)")

                wavelet_dd.change(update_wavelet_level_range, [cmp_inp, wavelet_dd], wavelet_level)
                cmp_inp.change(update_wavelet_level_range, [cmp_inp, wavelet_dd], wavelet_level)
                cmp_btn.click(
                    run_main_comparison,
                    [cmp_inp, target_mode, target_value, dct_block_size, wavelet_dd,
                     wavelet_level, include_jp2k],
                    [out_orig, out_dct, out_wav, dct_table, wav_table, cmp_plot,
                     cmp_verdict, out_jp2k, jp2k_table, jp2k_note],
                )

            # ---------------- SEKME 2 ----------------
            with gr.Tab("DCT Explorer"):
                gr.Markdown(
                    "Gerçek görüntüden seçilen TEK bir bloğun DCT boru hattı: "
                    "orijinal piksel → DCT katsayıları → kuantalama → rekonstrüksiyon."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        dct_inp = gr.Image(label="Girdi görüntüsü", type="numpy")
                        dct_bs = gr.Dropdown([str(b) for b in config.DCT_BLOCK_SIZE_OPTIONS],
                                             value="8", label="Blok boyutu")
                        dct_row = gr.Slider(0, 63, value=0, step=1, label="Blok satırı")
                        dct_col = gr.Slider(0, 63, value=0, step=1, label="Blok sütunu")
                        dct_q = gr.Slider(1, 100, value=50, step=1, label="Quality (1-100)")
                        dct_btn = gr.Button("Bloğu incele", variant="primary")
                    with gr.Column(scale=2):
                        dct_plot = gr.Plot(label="DCT boru hattı")
                        dct_info = gr.Markdown()

                dct_inp.change(dct_grid_shape, [dct_inp, dct_bs], [dct_row, dct_col])
                dct_bs.change(dct_grid_shape, [dct_inp, dct_bs], [dct_row, dct_col])
                dct_btn.click(run_dct_explorer, [dct_inp, dct_bs, dct_row, dct_col, dct_q],
                             [dct_plot, dct_info])

            # ---------------- SEKME 3 ----------------
            with gr.Tab("DWT Explorer"):
                gr.Markdown(
                    "Gerçek DWT katsayılarından piramit mozaik, dyadic ayrıştırma "
                    "ağacı, filtre bankası ve kayıpsız-rekonstrüksiyon testi."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        dwt_inp = gr.Image(label="Girdi görüntüsü", type="numpy")
                        dwt_wav = gr.Dropdown(config.WAVELET_UI_OPTIONS,
                                              value=config.WAVELET_DEFAULT_FILTER,
                                              label="Dalgacık ailesi")
                        dwt_lvl = gr.Slider(1, config.WAVELET_LEVEL_UI_MAX,
                                            value=config.WAVELET_LEVELS, step=1,
                                            label="Ayrıştırma seviyesi")
                        dwt_step = gr.Slider(0.5, 64.0, value=8.0, step=0.5,
                                             label="Kuantalama adımı (rekonstrüksiyon demosu)")
                        dwt_btn = gr.Button("Ayrıştır", variant="primary")
                    with gr.Column(scale=2):
                        dwt_verdict = gr.Markdown()
                        with gr.Row():
                            dwt_mosaic = gr.Image(label="Piramit mozaik (tüm seviyeler, gerçek katsayılar)")
                            dwt_grid = gr.Image(label="Seviye 1: LL | LH / HL | HH")
                        with gr.Row():
                            dwt_tree = gr.Plot(label="Dyadic Ayrıştırma Ağacı")
                            dwt_filt = gr.Plot(label="Filtre Bankası (darbe cevabı)")
                        dwt_filt_text = gr.Textbox(label="Filtre katsayıları (gerçek)", lines=4)
                        dwt_stats = gr.Dataframe(
                            headers=["Bant", "Boyut", "Min", "Max", "Ortalama", "Sıfır oranı"],
                            label="Subband istatistikleri (gerçek katsayılardan)", interactive=False)
                        dwt_recon = gr.Image(label="Kuantalamalı rekonstrüksiyon (gri)")

                dwt_wav.change(dwt_level_range, [dwt_inp, dwt_wav], dwt_lvl)
                dwt_inp.change(dwt_level_range, [dwt_inp, dwt_wav], dwt_lvl)
                dwt_btn.click(
                    run_dwt_explorer, [dwt_inp, dwt_wav, dwt_lvl, dwt_step],
                    [dwt_mosaic, dwt_grid, dwt_tree, dwt_filt, dwt_stats, dwt_verdict,
                     dwt_filt_text, dwt_recon],
                )

            # ---------------- SEKME 4 ----------------
            with gr.Tab("Semantik-Farkında Sıkıştırma"):
                gr.Markdown(
                    "YOLO-seg önemli bölgeleri bulur → aynı bit bütçesinde "
                    "**uniform** ve **ROI'li** sıkıştırma yan yana (projenin özgün özelliği)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        sem_inp = gr.Image(label="Girdi görüntüsü", type="numpy")
                        sem_engine = gr.Radio(
                            ["DCT (JPEG mantığı)", "Wavelet (JPEG2000 mantığı)"],
                            value="DCT (JPEG mantığı)", label="Sıkıştırma motoru",
                        )
                        sem_bpp = gr.Slider(0.15, 1.5, value=0.40, step=0.05,
                                            label="Hedef bpp (bit / piksel)")
                        sem_coarse = gr.Slider(2, 12, value=config.ROI_BG_COARSENESS, step=1,
                                               label="Arka plan kabalık çarpanı")
                        sem_btn = gr.Button("Sıkıştır", variant="primary")
                        if examples:
                            gr.Examples(examples=examples, inputs=sem_inp, label="Örnek görüntüler")
                    with gr.Column(scale=2):
                        sem_verdict = gr.Markdown()
                        with gr.Row():
                            sem_mask = gr.Image(label="YOLO-seg önem maskesi")
                            sem_base = gr.Image(label="Baseline (uniform)")
                            sem_sem = gr.Image(label="Semantic (ROI)")
                        sem_table = gr.Dataframe(
                            headers=["Metrik", "Baseline", "Semantic"],
                            label="Aynı bit bütçesinde kıyas", interactive=False,
                        )
                sem_btn.click(run_semantic_pipeline, [sem_inp, sem_engine, sem_bpp, sem_coarse],
                              [sem_mask, sem_base, sem_sem, sem_table, sem_verdict])

            # ---------------- SEKME 5 ----------------
            with gr.Tab("Teori / Nasıl Çalışır"):
                gr.Markdown(_THEORY_MD)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
