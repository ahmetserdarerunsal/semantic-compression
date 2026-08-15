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


def single_level_grid(coeffs: Sequence, level_index: int) -> np.ndarray:
    """Belirli bir ayrıştırma seviyesinin LL|LH / HL|HH 2x2 gridini döner.

    level_index=1 en kaba (son) seviye, level_index=len(coeffs)-1 en ince
    (ilk) seviyedir — pywt indeksiyle birebir aynı yön.
    LL paneli: o seviyeye kadar recursively ayrıştırılmış gerçek yaklaşım
    (coeffs[0] değil; her zaman "bir önceki" aşamanın piramidi).
    """
    level_index = max(1, min(level_index, len(coeffs) - 1))
    # LL paneli: level_index'e kadarki piramit (kendi içinde recursive)
    ll_panel = pyramid_display_image(coeffs[: level_index])
    cH, cV, cD = coeffs[level_index]
    return _combine_quadrants(ll_panel, _stretch(cH), _stretch(cV), _stretch(cD))


def coeff_stats(coeffs: Sequence) -> list[dict]:
    """Her bant için (etiket, shape, min, max, mean, sıfır-oranı) istatistiği.

    UI'da sayısal olarak göstermek için — piramitteki her karonun ARDINDA
    gerçek sayılar olduğunu kanıtlar (yalnızca dekoratif değildir).
    """
    rows = [dict(label="LL (yaklaşım)", **_band_stats(coeffs[0]))]
    n_levels = len(coeffs) - 1
    for li in range(1, len(coeffs)):
        level_no = n_levels - li + 1  # coeffs[1]=en kaba=en yüksek seviye no
        cH, cV, cD = coeffs[li]
        for name, band in [("LH", cH), ("HL", cV), ("HH", cD)]:
            rows.append(dict(label=f"{name}{level_no}", **_band_stats(band)))
    return rows


def _band_stats(band: np.ndarray) -> dict:
    total = band.size
    zeros = int(np.sum(np.isclose(band, 0.0, atol=1e-6)))
    return dict(
        shape=f"{band.shape[0]}x{band.shape[1]}",
        min=float(band.min()),
        max=float(band.max()),
        mean=float(band.mean()),
        zero_pct=100.0 * zeros / total if total else 0.0,
    )


def dyadic_tree_figure(levels: int) -> plt.Figure:
    """Dyadic (yalnız LL dalının recursively ayrıştırıldığı) ağaç şeması.

    Huffman ağacıyla KARIŞTIRILMAMALI: bu, katsayı uzayının kendisinin
    ayrıştırma hiyerarşisidir, olasılık/kod ağacı değildir.
    """
    fig, ax = plt.subplots(figsize=(2.6 * levels + 2, 3.2))
    ax.axis("off")
    x0, y0 = 0.0, 0.0
    box_w, box_h = 1.1, 0.6
    gap_y = 1.1

    def _box(x, y, text, color="#0072B2", textcolor="white"):
        ax.add_patch(
            plt.Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h,
                          facecolor=color, edgecolor="#333333", linewidth=1.0, zorder=2)
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                color=textcolor, zorder=3, fontweight="bold")

    def _line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="#888888", linewidth=1.2, zorder=1)

    _box(x0, y0, "Görüntü", color="#454247")
    cur_x, cur_y = x0, y0
    detail_color = "#E69F00"
    ll_color = "#0072B2"
    span = 1.6
    for lvl in range(1, levels + 1):
        y = cur_y - gap_y
        xs = [cur_x - 1.5 * span, cur_x - 0.5 * span, cur_x + 0.5 * span, cur_x + 1.5 * span]
        labels = [f"LL{lvl}", f"LH{lvl}", f"HL{lvl}", f"HH{lvl}"]
        colors = [ll_color, detail_color, detail_color, detail_color]
        for x, label, color in zip(xs, labels, colors):
            _line(cur_x, cur_y - box_h / 2, x, y + box_h / 2)
            _box(x, y, label, color=color)
        cur_x, cur_y = xs[0], y  # yalnız LL dalı recursively devam eder
        span *= 0.55

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(cur_y - 1.0, 1.0)
    ax.set_title(
        f"Dyadic Ayrıştırma Ağacı — {levels} seviye "
        "(mavi=LL dalı recursively bölünür, turuncu=detay yaprakları)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


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
    info = filter_bank_info(wavelet)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    axes[0].stem(info["dec_lo"])
    axes[0].set_title(f"Analiz alçak-geçiren (h) — {info['dec_len']} tap", fontsize=9)
    axes[1].stem(info["dec_hi"], linefmt="C1-", markerfmt="C1o")
    axes[1].set_title(f"Analiz yüksek-geçiren (g) — {info['dec_len']} tap", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="#888888", linewidth=0.6)
        ax.set_xlabel("katsayı indeksi n")
    fig.suptitle(f"Filtre Bankası — {wavelet}", fontsize=10)
    fig.tight_layout()
    return fig
