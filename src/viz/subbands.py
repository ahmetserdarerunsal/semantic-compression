# -*- coding: utf-8 -*-
"""DWT Explorer görselleştirmeleri: gerçek katsayılardan piramit mozaik,
dyadic ayrıştırma ağacı ve filtre bankası bilgisi.

Tüm görseller `wavelet_engine.decompose_for_viz` ile üretilen GERÇEK
pywt.wavedec2 katsayılarından hesaplanır. Normalizasyon yalnızca ekranda
gösterim içindir; ham katsayı değerleri değiştirilmez (bkz. mega-spec:
"Do NOT use decorative/static images").
"""
from __future__ import annotations

from typing import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt


def _stretch(band: np.ndarray) -> np.ndarray:
    """Tek bir subband'i ekran için 0-255'e gerer (1.-99. persentil kırpma).

    Sadece GÖSTERİM amaçlı; döndürülen dizi ayrı bir kopyadır, orijinal
    katsayı dizisine dokunulmaz.
    """
    lo, hi = np.percentile(band, [1, 99])
    if hi <= lo:
        return np.full(band.shape, 128, dtype=np.uint8)
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def pyramid_display_image(coeffs: Sequence) -> np.ndarray:
    """Çok seviyeli DWT katsayılarından TAM piramit mozaiği (uint8, gri).

    coeffs: pywt.wavedec2 çıktısı [cA_n, (cH_n,cV_n,cD_n), ..., (cH_1,cV_1,cD_1)].
    Her seviyede LL bölgesi bir sonraki (daha ince) seviyenin kendi
    LL/LH/HL/HH mozaiğiyle recursively değiştirilir — klasik multiresolution
    piramit görünümü. Her bant kendi içinde bağımsız kontrast-gerilir (LL
    yaklaşık görüntü gibi, detay bantları kenar/doku deseni gibi görünür).
    """
    mosaic = _stretch(coeffs[0])  # her seviyede kendi içinde kontrast-gerilir
    for li in range(1, len(coeffs)):
        mosaic = _combine_quadrants(mosaic, *(_stretch(b) for b in coeffs[li]))
    return mosaic


def _combine_quadrants(ll: np.ndarray, cH: np.ndarray, cV: np.ndarray,
                        cD: np.ndarray) -> np.ndarray:
    """LL (veya alt-piramit) ile aynı seviyenin 3 detay bandını 2x2 mozaikte
    birleştirir. pywt'nin sınır (symmetric) uzatması nedeniyle bantların
    boyutları birebir eşleşmeyebilir; ortak minimum boyuta kırpılır (yalnız
    GÖSTERİM mozaiği için — gerçek katsayı dizileri değişmez)."""
    th = min(ll.shape[0], cH.shape[0], cV.shape[0], cD.shape[0])
    tw = min(ll.shape[1], cH.shape[1], cV.shape[1], cD.shape[1])
    ll, cH, cV, cD = (a[:th, :tw] for a in (ll, cH, cV, cD))
    top = np.hstack([ll, cH])
    bottom = np.hstack([cV, cD])
    return np.vstack([top, bottom])


def pyramid_regions(levels_data: dict[int, dict[str, np.ndarray]], n_levels: int) -> list[dict]:
    """pyramid_display_image()'ın çizdiği İÇ İÇE 2x2 mozaik geometrisini
    (aynı ortak-minimum kırpma kuralı dahil, bkz. _combine_quadrants)
    piksel ÇİZMEDEN simüle ederek, her subband'in mozaiktaki NİHAİ piksel
    dikdörtgenini hesaplar (mega-spec "DWT LAB — TIKLAMA İLE NOKTA
    İNCELEMEYİ 3 GÖRSELDE AKTİF ET" Part 1) — tıklanan pikselden hangi
    (level, band) bölgesine denk geldiğini çözmek için.

    `levels_data[k]["LH"/"HL"/"HH"]` dizileri pyramid_display_image'ın
    kullandığı `coeffs[li]` bantlarıyla LİTERAL AYNI nesnelerdir (bkz.
    decompose_levels) — bu yüzden buradan hesaplanan bölgeler GERÇEKTEN
    çizilen pikselle piksel-kesin eşleşir; `levels_data["LL"]` de yalnız
    en kaba seviyede (`k == n_levels`) coeffs[0] ile aynıdır (mozaik ara
    seviye LL_k'lerini hiç AYRI göstermez — onlar sonraki iterasyonlarda
    daha ince mozaikle YER DEĞİŞTİRİR).

    Dönüş: [{"level", "band", "y0", "x0", "h", "w"}, ...] — piksel
    dikdörtgenleri, EKLENME sırasıyla (en kaba/en iç önce)."""
    ll = levels_data[n_levels]["LL"]
    cur_h, cur_w = ll.shape
    regions: list[dict] = [{"level": n_levels, "band": "LL", "y0": 0, "x0": 0, "h": cur_h, "w": cur_w}]

    for k in range(n_levels, 0, -1):
        lh, hl, hh = levels_data[k]["LH"], levels_data[k]["HL"], levels_data[k]["HH"]
        th = min(cur_h, lh.shape[0], hl.shape[0], hh.shape[0])
        tw = min(cur_w, lh.shape[1], hl.shape[1], hh.shape[1])
        # _combine_quadrants ÖNCEKİ mozaiğin TAMAMINI (içindeki her
        # bölgeyle birlikte) [:th,:tw]'e kırpıp değişmeden (0,0)'a
        # yerleştirir — buradaki tüm önceki bölgeler AYNI şekilde kırpılır.
        for r in regions:
            r["h"] = max(0, min(r["h"], th - r["y0"])) if r["y0"] < th else 0
            r["w"] = max(0, min(r["w"], tw - r["x0"])) if r["x0"] < tw else 0
        regions.append({"level": k, "band": "LH", "y0": 0, "x0": tw, "h": th, "w": tw})
        regions.append({"level": k, "band": "HL", "y0": th, "x0": 0, "h": th, "w": tw})
        regions.append({"level": k, "band": "HH", "y0": th, "x0": tw, "h": th, "w": tw})
        cur_h, cur_w = th * 2, tw * 2

    return regions


def subband_from_mosaic_click(regions: list[dict], x_px: int, y_px: int) -> dict | None:
    """Piramit mozaiğinde tıklanan pikselden hangi (level, band) bölgesine
    ve o bölge İÇİNDEKİ yerel (satır, sütun) katsayı konumuna denk
    geldiğini çözer. Bölgeler kesişmez (strict quadrant partition), bu
    yüzden en fazla BİR eşleşme döner; hiçbiri kapsamıyorsa None."""
    for r in regions:
        if r["h"] <= 0 or r["w"] <= 0:
            continue
        if r["y0"] <= y_px < r["y0"] + r["h"] and r["x0"] <= x_px < r["x0"] + r["w"]:
            return {"level": r["level"], "band": r["band"],
                   "row": y_px - r["y0"], "col": x_px - r["x0"]}
    return None


def mark_point(gray_or_rgb: np.ndarray, row: int | None, col: int | None,
               radius: int = 4) -> np.ndarray:
    """Herhangi bir gri/renkli görüntü üzerinde TEK bir pikseli küçük bir
    camgöbeği çerçeveyle işaretler (mega-spec "TIKLANAN NOKTAYI İŞARETLE",
    Part 4) — Piramit/Rekonstrüksiyon/Fark Haritası'nın ÜÇÜ için de aynı
    paylaşılan çizim yardımcısı (kopya kod yazılmaz). Yalnız GÖSTERİM
    amaçlıdır; girdi dizisi değiştirilmez, RGB bir kopya döner."""
    a = np.clip(gray_or_rgb, 0, 255).astype(np.uint8)
    rgb = a if a.ndim == 3 else np.stack([a] * 3, axis=-1)
    rgb = rgb.copy()
    h, w = rgb.shape[:2]
    if row is None or col is None or not (0 <= row < h and 0 <= col < w):
        return rgb
    x0, y0 = max(0, col - radius), max(0, row - radius)
    x1, y1 = min(w - 1, col + radius), min(h - 1, row + radius)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), (49, 200, 255), 2)
    return rgb


def decompose_levels(
    coeffs: Sequence, n_levels: int, wavelet: str, mode: str = "symmetric",
) -> dict[int, dict[str, np.ndarray]]:
    """pywt.wavedec2 çıktısını KULLANICI-YÜZÜ seviye numarasına göre eşler —
    1 = ilk/en ince ayrıştırma (görüntüye en yakın, en büyük diziler),
    n_levels = son/en kaba ayrıştırma (en küçük diziler).

    KRİTİK DÜZELTME (audit'te bulunan gerçek bug): pywt'nin KENDİ coeffs
    listesi bunun TERSİ sırada döner —
        coeffs = [cA_n, D_n, D_{n-1}, ..., D_1]   (D_i = (cH_i,cV_i,cD_i))
    yani coeffs[1] EN KABA seviyenin (n_levels) detayını taşır, coeffs[-1]
    EN İNCE seviyenin (1) detayını taşır. Önceki kod `coeffs[level_index]`
    ile DOĞRUDAN indexliyordu — kullanıcı "Seviye 1" seçtiğinde aslında en
    kaba seviyenin verisini gösteriyordu (ve etiket hiç seviye numarası
    içermediğinden bu fark UI'da görünmüyordu). Doğru eşleme:
        li = n_levels - k + 1   (k: kullanıcı seviyesi, li: coeffs indeksi)
    Bu fonksiyon TEK yerde, açıkça bu eşlemeyi uygular.

    LL_k (k < n_levels): pywt'nin LİTERAL çıktısı DEĞİLDİR — yalnız
    coeffs[0] (k == n_levels) literaldir. Ara seviyelerin LL'i, AYNI coeffs
    listesinin coeffs[0 : n_levels-k+1] kısmının TERS DÖNÜŞÜMÜ (waverec2)
    ile hesaplanır; bu, tamamen bağımsız bir `wavedec2(..., level=k)`
    çağrısından FARKLI (ve bu ekran için doğru) sonuç verir, çünkü aynı
    decomposition zincirinin İÇİNDEN türer — pywt'nin tek-sayı boyutlarda
    simetrik sınır uzatması nedeniyle bağımsız decompozisyon farklı bir
    piksel boyutu üretebilir (doğrulanmıştır). Sonuç olarak LL_k'nin boyutu
    bazen eşlik ettiği LH_k/HL_k/HH_k'den 1 piksel farklı olabilir — bu bir
    hata değildir (mega-spec Part 7); her bandın GERÇEK shape'i ayrı ayrı
    gösterilir, birbirine zorla eşitlenmez.

    Dönüş: {k: {"LL": ndarray, "LH": ndarray, "HL": ndarray, "HH": ndarray}}
    Tüm diziler HAM float64 katsayılardır (görüntüleme normalizasyonu YOK).
    """
    out: dict[int, dict[str, np.ndarray]] = {}
    for k in range(1, n_levels + 1):
        li = n_levels - k + 1
        cH, cV, cD = coeffs[li]
        if k == n_levels:
            ll = coeffs[0]
        else:
            # Kısmi ters dönüşüm AYNI sınır moduyla yapılmalı — mode
            # forward decomposition'la uyuşmazsa LL_k tutarsız olur.
            ll = pywt.waverec2(coeffs[0: n_levels - k + 1], wavelet, mode=mode)
        out[k] = {"LL": ll, "LH": cH, "HL": cV, "HH": cD}
    return out


def band_preview(band: np.ndarray) -> np.ndarray:
    """Tek bir bandın (LL/LH/HL/HH, herhangi bir seviye) ekran-normalize
    önizleme görüntüsü — dışa açık ince sarmalayıcı (bkz. _stretch)."""
    return _stretch(band)


def band_preview_with_marker(band: np.ndarray, row: int | None, col: int | None) -> np.ndarray:
    """band_preview() + TIKLANAN hücrenin görsel işareti (mega-spec "DWT
    LAB — subband görüntülerine tıklama ile katsayı inceleme").

    row/col, GERÇEK ham katsayı dizisinin (band) satır/sütunudur —
    band_preview() diziyi YENİDEN BOYUTLANDIRMAZ (yalnız değer aralığını
    0-255'e gerer), bu yüzden işaret KESİNLİKLE band.shape ile aynı piksel
    uzayına çizilir; ek bir koordinat dönüşümüne gerek yoktur. Yalnız
    GÖSTERİM amaçlı bir overlay'dir — döndürülen dizi ayrı bir kopyadır,
    istatistik/histogram hesaplarına (RAW `band` üzerinden çalışır)
    hiçbir etkisi yoktur."""
    stretched = _stretch(band)
    rgb = np.stack([stretched] * 3, axis=-1).copy()
    h, w = band.shape
    if row is None or col is None or not (0 <= row < h and 0 <= col < w):
        return rgb
    # İşaret boyutu diziye oranlı (çok küçük bantlarda bile görünür kalsın,
    # büyük bantlarda tek pikselden daha belirgin olsun).
    half = max(1, min(h, w) // 40)
    x0, y0 = max(0, col - half), max(0, row - half)
    x1, y1 = min(w - 1, col + half), min(h - 1, row + half)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), (49, 200, 255), 1)
    return rgb


def selected_band_stats(band: np.ndarray) -> dict:
    """Seçilen TEK bandın ham istatistiği (mega-spec Part 12/20).

    'Enerji %' (tüm decomposition'a oranla) buraya BİLEREK dahil edilmez:
    ara seviyelerin LL'i (k < n_levels) kısmi-rekonstrüksiyon olduğundan
    "toplam enerjiye oranı" onun için iyi tanımlı bir büyüklük değildir —
    yalnız MUTLAK enerji (Σkatsayı²) gösterilir, her koşulda doğru ve
    anlamlıdır. Tüm decomposition'a göre normalize enerji yüzdesi, yalnız
    coeff_stats()'ın ürettiği TAM tabloda (Teknik Detaylar) gösterilir —
    orada LL yalnız gerçek/literal coeffs[0] için bir kez görünür."""
    total = band.size
    zeros = int(np.sum(np.isclose(band, 0.0, atol=1e-6)))
    return dict(
        shape=f"{band.shape[0]}×{band.shape[1]}",
        count=total,
        min=float(band.min()), max=float(band.max()),
        mean=float(band.mean()), std=float(band.std()),
        energy=float(np.sum(band.astype(np.float64) ** 2)),
        zero_pct=100.0 * zeros / total if total else 0.0,
    )


def coeff_stats(coeffs: Sequence) -> list[dict]:
    """Her bant için (etiket, shape, min, max, mean, std, enerji%, sıfır-
    oranı, katsayı sayısı) istatistiği.

    UI'da sayısal olarak göstermek için — piramitteki/ağaçtaki her düğümün
    ARDINDA gerçek sayılar olduğunu kanıtlar (yalnızca dekoratif değildir;
    mega-spec: "Node hover: Level, Subband, Dimensions, Energy, Mean, Std,
    Coefficient count"). Enerji, TÜM bantların toplam enerjisine göre
    yüzde olarak normalize edilir (Σ katsayı² / toplam Σ katsayı²).
    """
    all_bands = [coeffs[0]]
    for li in range(1, len(coeffs)):
        all_bands.extend(coeffs[li])
    total_energy = sum(float(np.sum(b.astype(np.float64) ** 2)) for b in all_bands)

    rows = [dict(label="LL (yaklaşım)", **_band_stats(coeffs[0], total_energy))]
    n_levels = len(coeffs) - 1
    for li in range(1, len(coeffs)):
        level_no = n_levels - li + 1  # coeffs[1]=en kaba=en yüksek seviye no
        cH, cV, cD = coeffs[li]
        for name, band in [("LH", cH), ("HL", cV), ("HH", cD)]:
            rows.append(dict(label=f"{name}{level_no}", **_band_stats(band, total_energy)))
    return rows


def quantized_sparsity_stats(raw_coeffs: Sequence, quantized_coeffs: Sequence) -> dict:
    """Kuantalama ÖNCESİ (ham, hemen hiç sıfır olmayan float katsayılar) ve
    SONRASI (tam sayı kuantalanmış) seyreklik karşılaştırması.

    mega-spec Part 7: ham katsayılarda zero_pct≈%0 olması hata değildir —
    sıkıştırmayı açıklayan asıl istatistik KUANTALAMA SONRASI seyrekliktir.
    Değerler her zaman güncel görüntü/parametrelerden hesaplanır.
    """
    def _flatten(coeffs):
        arrs = [coeffs[0]]
        for li in range(1, len(coeffs)):
            arrs.extend(coeffs[li])
        return np.concatenate([a.ravel() for a in arrs])

    raw = _flatten(raw_coeffs)
    q = _flatten(quantized_coeffs)
    raw_total = raw.size
    raw_nonzero = int(np.sum(~np.isclose(raw, 0.0, atol=1e-6)))
    q_total = q.size
    q_nonzero = int(np.count_nonzero(q))
    q_zero = q_total - q_nonzero
    return dict(
        raw_total=raw_total,
        raw_nonzero_pct=100.0 * raw_nonzero / raw_total if raw_total else 0.0,
        total=q_total,
        nonzero=q_nonzero,
        zero=q_zero,
        nonzero_pct=100.0 * q_nonzero / q_total if q_total else 0.0,
        sparsity_pct=100.0 * q_zero / q_total if q_total else 0.0,
    )


def _band_stats(band: np.ndarray, total_energy: float | None = None) -> dict:
    total = band.size
    zeros = int(np.sum(np.isclose(band, 0.0, atol=1e-6)))
    energy = float(np.sum(band.astype(np.float64) ** 2))
    return dict(
        shape=f"{band.shape[0]}x{band.shape[1]}",
        count=total,
        min=float(band.min()),
        max=float(band.max()),
        mean=float(band.mean()),
        std=float(band.std()),
        energy_pct=100.0 * energy / total_energy if total_energy else 0.0,
        zero_pct=100.0 * zeros / total if total else 0.0,
    )


def dyadic_tree_image(
    levels: int, active_level: int | None = None, active_band: str = "LL",
) -> tuple[np.ndarray, list[dict]]:
    """Multiresolution akış haritası — dosya-sistemi ağacı DEĞİL, sinyal akış
    diyagramı (bu projeye özgü, orijinal tasarım). `gr.Plot` yerine `gr.Image`
    ile kullanılmak üzere RGB uint8 dizi + her seviyenin/her BANDIN
    (mega-spec "DWT LAB — TÜM SUBBAND NODE'LARINI TAM INTERAKTİF HALE
    GETİR") PİKSEL-uzayı sınırlarını birlikte döner.

    Geometri: LL zinciri dikey MERKEZ hattı oluşturur (INPUT'tan başlayıp
    her seviyede bir sonraki LL'e düz iner); her seviyede LH/HL/HH detay
    katsayıları LL düğümünden YATAY olarak dallanan küçük, ayrı renkli
    çipler olarak gösterilir (LL=camgöbeği ana yol, LH=yeşilimsi,
    HL=amber, HH=mor). `active_level`+`active_band` verilirse YALNIZ o TEK
    düğüm (satırın TAMAMI değil) belirgin kenarlık/parlaklıkla vurgulanır —
    flashy animasyon değil, sade bir fark; INPUT'a kadarki yol da (hangi
    seviyeye kadar ayrıştırıldığı) ince bir çizgiyle işaretlenir.

    Piksel sınırları `ax.transData` ile HESAPLANIR (figürün gerçek çizim
    sonrası düzeninden — tight_layout dahil — okunur, sabit bir varsayım
    DEĞİLDİR) — hem dikey (seviye) hem yatay (bant: LL/LH/HL/HH) eksende;
    bu yüzden click-to-select ataması matematiksel olarak kesindir,
    yaklaşık/tahmini değildir. Dönüş: (image, bounds) — bounds her seviye
    için {"level", "y0", "y1", "LL": (x0,x1), "LH": (x0,x1), "HL": (x0,x1),
    "HH": (x0,x1)} sözlüğü (bkz. subband_from_click).

    Huffman ağacıyla KARIŞTIRILMAMALI: bu, katsayı uzayının kendisinin
    ayrıştırma hiyerarşisidir, olasılık/kod ağacı değildir.
    """
    from src.viz.style import BG_PANEL, BORDER_LIGHT, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY

    LL_COLOR = "#31C8FF"   # ana yol (cyan)
    LH_COLOR = "#4ADE80"   # yeşilimsi
    HL_COLOR = "#FBBF24"   # amber
    HH_COLOR = "#A78BFA"   # mor
    INPUT_COLOR = BORDER_LIGHT
    ACTIVE_GLOW = "#F3F6FA"  # seçili seviyeyi çevreleyen ince parlak kenarlık

    ll_w, ll_h = 1.1, 0.58
    chip_w, chip_h = 0.62, 0.34
    chip_gap = 0.14
    row_gap = 1.5
    ll_x = 0.0

    fig, ax = plt.subplots(figsize=(6.0, 0.9 + row_gap * levels), dpi=120)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)
    ax.axis("off")

    def _node(x, y, w, h, text, color, textcolor="#0B1120", fontsize=9.5, active=False):
        if active:
            ax.add_patch(plt.Rectangle((x - w / 2 - 0.05, y - h / 2 - 0.05), w + 0.1, h + 0.1,
                                       facecolor="none", edgecolor=ACTIVE_GLOW, linewidth=2.0, zorder=2))
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color,
                                   edgecolor=(ACTIVE_GLOW if active else BORDER_LIGHT),
                                   linewidth=(1.6 if active else 0.9), zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=textcolor, zorder=3, fontweight="bold")

    def _line(x1, y1, x2, y2, color=TEXT_MUTED, lw=1.3):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)

    y0 = 0.0
    _node(ll_x, y0, ll_w, ll_h, "INPUT", INPUT_COLOR, TEXT_PRIMARY, 9)

    y = y0
    chip_x0 = ll_x + ll_w / 2 + 0.35
    level_y: dict[int, float] = {}
    chip_cx: dict[int, dict[str, float]] = {}  # {level: {"LH": cx, "HL": cx, "HH": cx}}
    for lvl in range(1, levels + 1):
        y_next = y - row_gap
        level_y[lvl] = y_next
        on_active_path = active_level is not None and lvl <= active_level
        _line(ll_x, y - ll_h / 2, ll_x, y_next + ll_h / 2,
             color=(ACTIVE_GLOW if on_active_path else TEXT_MUTED), lw=(2.0 if on_active_path else 1.3))
        is_active_row = lvl == active_level
        ax.text(ll_x - ll_w / 2 - 0.18, y_next, f"LEVEL {lvl}", ha="right", va="center",
                fontsize=9, color=(TEXT_PRIMARY if is_active_row else TEXT_SECONDARY),
                fontweight="bold")
        # Yalnız GERÇEKTEN seçili TEK düğüm (satırın tamamı değil) belirgin
        # vurguya sahip olur (mega-spec Part 7: "belirgin border + hafif
        # glow" — LL/LH/HL/HH birbirinden BAĞIMSIZ seçilebilir düğümlerdir).
        _node(ll_x, y_next, ll_w, ll_h, f"LL{lvl}", LL_COLOR,
             active=(is_active_row and active_band == "LL"))

        # LH/HL/HH: LL düğümünden yatay dallanan küçük çip kümesi
        branch_y = y_next
        _line(ll_x + ll_w / 2, branch_y, chip_x0 - chip_w / 2 - chip_gap, branch_y)
        chips = [("LH", LH_COLOR), ("HL", HL_COLOR), ("HH", HH_COLOR)]
        chip_cx[lvl] = {}
        for i, (label, color) in enumerate(chips):
            cx = chip_x0 + i * (chip_w + chip_gap)
            chip_cx[lvl][label] = cx
            _line(chip_x0 - chip_w / 2 - chip_gap if i == 0 else cx - chip_w - chip_gap,
                 branch_y, cx - chip_w / 2, branch_y)
            _node(cx, branch_y, chip_w, chip_h, f"{label}{lvl}", color, "#0B1120", 7.5,
                 active=(is_active_row and active_band == label))

        y = y_next

    chips_right_edge = chip_x0 + 2 * (chip_w + chip_gap) + chip_w / 2
    ax.set_xlim(ll_x - ll_w / 2 - 1.5, chips_right_edge + 0.2)
    ax.set_ylim(y - ll_h, ll_h * 1.6)
    fig.tight_layout()
    fig.canvas.draw()  # tight_layout dahil NİHAİ yerleşimi sabitler (transData bundan SONRA doğrudur)

    canvas_w, canvas_h = fig.canvas.get_width_height()

    def _x_range(cx_data: float, half_w_data: float) -> tuple[int, int]:
        x_lo_disp, _ = ax.transData.transform((cx_data - half_w_data, 0.0))
        x_hi_disp, _ = ax.transData.transform((cx_data + half_w_data, 0.0))
        return int(x_lo_disp), int(x_hi_disp)

    bounds: list[dict] = []
    for lvl in range(1, levels + 1):
        # transData: veri-uzayı -> EKRAN pikseli (sol-alt orijin); dizi
        # satırına çevirmek için dikey eksen ters çevrilir (üst-sol orijin).
        _, y_disp = ax.transData.transform((ll_x, level_y[lvl]))
        row_center = canvas_h - y_disp
        half = row_gap * (canvas_h / (ax.get_ylim()[1] - ax.get_ylim()[0])) / 2.0
        entry = {
            "level": lvl, "y0": int(row_center - half), "y1": int(row_center + half),
            "LL": _x_range(ll_x, ll_w / 2),
        }
        for label, cx in chip_cx[lvl].items():
            entry[label] = _x_range(cx, chip_w / 2)
        bounds.append(entry)

    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()  # RGBA -> RGB, kendi kopyası (canvas serbest bırakılınca bozulmasın)
    plt.close(fig)
    return image, bounds


def subband_from_click(bounds: list[dict], x_px: int, y_px: int, max_level: int) -> tuple[int, str]:
    """dyadic_tree_image()'ın döndürdüğü piksel sınırlarını kullanarak
    tıklanan (x,y) piksel konumundan GERÇEK (seviye, bant) çiftini çözer
    (mega-spec "DWT LAB — TÜM SUBBAND NODE'LARINI TAM INTERAKTİF HALE
    GETİR"). Önceki `level_from_click_y` yalnız DİKEY (Y) ekseni okuyordu
    — bu, LH/HL/HH çiplerinin görsel olarak ayrı düğümler gibi görünüp
    tıklandığında HİÇBİR ŞEY yapmaması (her zaman aynı satırdaki mevcut
    bant korunuyordu) bug'ının KÖK NEDENIYDI. Artık her düğümün (LL/LH/
    HL/HH) kendi GERÇEK yatay piksel aralığı da kontrol edilir.

    Bir satıra (seviyeye) isabet edip HİÇBİR düğümün TAM içine denk
    gelmeyen tıklamalar (örn. çipler arasındaki boşluk) o satırın LL'ine
    düşer — belirsiz/tanımsız bir bant seçmek yerine deterministik,
    öngörülebilir bir varsayılan. Sınırların tamamen dışına (INPUT'un
    üstüne / son seviyenin altına) tıklamak en yakın geçerli seviyenin
    LL'ine clamp edilir."""
    if not bounds:
        return 1, "LL"
    for entry in bounds:
        if entry["y0"] <= y_px <= entry["y1"]:
            for band in ("LL", "LH", "HL", "HH"):
                x0, x1 = entry[band]
                if x0 <= x_px <= x1:
                    return entry["level"], band
            return entry["level"], "LL"
    # En yakın seviyeye clamp (sınırlar arasındaki boşluklara tıklanırsa)
    closest = min(bounds, key=lambda e: min(abs(y_px - e["y0"]), abs(y_px - e["y1"])))
    return max(1, min(closest["level"], max_level)), "LL"


def energy_distribution_figure(coeffs: Sequence) -> plt.Figure:
    """Her bandın GERÇEK enerjisinin (Σkatsayı²) toplam decomposition
    enerjisine oranını bar chart olarak gösterir (mega-spec "FINAL
    INTEGRATION" Part 30) — coeff_stats() ile AYNI enerji hesaplamasını
    kullanır (tek kaynak, iki farklı yerde iki farklı sayı üretilmez)."""
    from src.viz.style import (BORDER_LIGHT, TEXT_PRIMARY, TEXT_SECONDARY,
                               apply_lab_style)

    stats = coeff_stats(coeffs)
    labels = [r["label"] for r in stats]
    values = [r["energy_pct"] for r in stats]
    band_colors = {"LL": "#31C8FF", "LH": "#4ADE80", "HL": "#FBBF24", "HH": "#A78BFA"}
    colors = [band_colors.get(lbl[:2] if lbl[:2] in band_colors else lbl[:1], BORDER_LIGHT)
             for lbl in labels]

    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.bar(range(len(labels)), values, color=colors, edgecolor="none")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color=TEXT_SECONDARY)
    ax.set_ylabel("Enerji %")
    ax.set_title("Subband Enerji Dağılımı — Σkatsayı² / toplam enerji", fontsize=10, color=TEXT_PRIMARY)
    apply_lab_style(fig, ax)
    fig.tight_layout()
    return fig


def band_histogram_figure(band: np.ndarray, title: str) -> plt.Figure:
    """Seçili TEK bandın (herhangi bir seviye/tür) ham katsayı dağılımı
    histogramı — GERÇEK katsayılardan (mega-spec Part 31: 'HH3 seçildiyse
    HH3 raw coefficients')."""
    from src.viz.style import ACCENT_PURPLE, TEXT_PRIMARY, apply_lab_style

    # 16 subband düğümünün TAMAMI artık tıklanabilir olduğundan (mega-spec
    # "TÜM SUBBAND NODE'LARINI TAM INTERAKTİF HALE GETİR") bu fonksiyon
    # eskisinden çok daha sık çağrılıyor — dct_block.block_pipeline_figure
    # ile AYNI desen (bkz. orada): önceki çağrılardan kalan, zaten Gradio'ya
    # gönderilmiş figürleri kapatmadan bırakmak matplotlib'in global figür
    # kaydında sızıntıya yol açardı.
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.hist(band.ravel(), bins=64, color=ACCENT_PURPLE, alpha=0.85)
    ax.axvline(0, color="#64748B", linewidth=0.8)
    ax.set_xlabel("katsayı değeri")
    ax.set_ylabel("frekans")
    ax.set_title(f"{title} — katsayı dağılımı ({band.size:,} katsayı)", fontsize=9.5, color=TEXT_PRIMARY)
    apply_lab_style(fig, ax)
    fig.tight_layout()
    return fig


def reconstruction_diff_image(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Orijinal - rekonstrüksiyon farkının GERÇEK piksel-uzayı görüntüsü
    (mega-spec Part 29) — işaretli fark simetrik normalize edilir (0 = orta
    gri, pozitif/negatif fark simetrik haritalanır); yalnız GÖSTERİM
    normalizasyonudur, döndürülen sayısal fark (dışarıda ayrıca
    hesaplanan MSE/PSNR) buradan ETKİLENMEZ."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    m = float(np.max(np.abs(diff)))
    if m < 1e-9:
        return np.full(diff.shape, 128, dtype=np.uint8)
    out = (diff / m) * 127.5 + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def filter_bank_info(wavelet: str) -> dict:
    """Seçili dalgacığın gerçek analiz/sentez filtre katsayıları ve uzunluğu.

    pywt.Wavelet(...).filter_bank -> (dec_lo, dec_hi, rec_lo, rec_hi).
    UI bu sayıları doğrudan gösterir; icat edilmiş bir "tap sayısı" değildir.
    """
    w = pywt.Wavelet(wavelet)
    dec_lo, dec_hi, rec_lo, rec_hi = w.filter_bank
    return dict(
        name=w.name,
        family=w.family_name,
        orthogonal=w.orthogonal,
        biorthogonal=w.biorthogonal,
        dec_len=w.dec_len,
        rec_len=w.rec_len,
        dec_lo=[round(c, 6) for c in dec_lo],
        dec_hi=[round(c, 6) for c in dec_hi],
        rec_lo=[round(c, 6) for c in rec_lo],
        rec_hi=[round(c, 6) for c in rec_hi],
    )


def filter_bank_figure(wavelet: str) -> plt.Figure:
    """Analiz alçak/yüksek geçiren filtre darbe cevaplarının çubuk grafiği."""
    from src.viz.style import ACCENT_CYAN, ACCENT_PURPLE, TEXT_PRIMARY, apply_lab_style

    info = filter_bank_info(wavelet)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    markerline, stemlines, base = axes[0].stem(info["dec_lo"])
    plt.setp(markerline, color=ACCENT_CYAN)
    plt.setp(stemlines, color=ACCENT_CYAN)
    axes[0].set_title(f"Analiz alçak-geçiren (h) — {info['dec_len']} tap", fontsize=9)
    markerline2, stemlines2, base2 = axes[1].stem(info["dec_hi"])
    plt.setp(markerline2, color=ACCENT_PURPLE)
    plt.setp(stemlines2, color=ACCENT_PURPLE)
    axes[1].set_title(f"Analiz yüksek-geçiren (g) — {info['dec_len']} tap", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="#64748B", linewidth=0.6)
        ax.set_xlabel("katsayı indeksi n")
    apply_lab_style(fig, axes)
    fig.suptitle(f"Filtre Bankası — {wavelet}", fontsize=10, color=TEXT_PRIMARY)
    fig.tight_layout()
    return fig
