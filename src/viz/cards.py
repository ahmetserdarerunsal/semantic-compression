# -*- coding: utf-8 -*-
"""HTML metrik kartı üretimi: laboratuvar arayüzünün Gradio Dataframe yerine
kullandığı özel kartlar (mega-spec Part 31 — varsayılan Dataframe hem görsel
olarak zayıf hem de etiketlerin kırpıldığı bir düzen hatasına sahipti).

Tüm sayılar İÇERİDE yuvarlanmaz — yalnız burada, GÖSTERİM anında (Part 70).
Tutarlı hassasiyet: bpp 3 ondalık, PSNR 2, SSIM 4, MSE 3 (adaptif), oran 2,
boyut 2, çok küçük hatalar bilimsel gösterim.
"""
from __future__ import annotations

import base64
import html as _html

import cv2
import numpy as np

from src.viz.style import ACCENT_CYAN, ACCENT_PURPLE, CRITICAL, POSITIVE, TEXT_MUTED, WARNING

# ---------------------------------------------------------------------------
# Biçimlendirme (Part 70) — yuvarlama YALNIZ burada yapılır
# ---------------------------------------------------------------------------
def fmt_psnr(v: float) -> str:
    """NaN ve +Inf FARKLI anlamlara gelir, aynı gösterime karıştırılmaz
    (numerical-correctness audit): +Inf = MSE=0, gerçek kayıpsız
    rekonstrüksiyon ("∞ (kayıpsız)"); NaN = metrik bu bölgede TANIMSIZ
    (ör. ROI tüm görüntüyü kaplayınca arka plan boş kalır) — "N/A", asla
    "kayıpsız" gibi yanlış bir anlam taşımaz."""
    if np.isnan(v):
        return "N/A"
    if np.isinf(v):
        return "∞ (kayıpsız)"
    return f"{v:.2f} dB"


def fmt_ssim(v: float) -> str:
    return "N/A" if np.isnan(v) else f"{v:.4f}"


def fmt_mse(v: float) -> str:
    return f"{v:.3e}" if (0 < abs(v) < 0.001) else f"{v:.3f}"


def fmt_bpp(v: float) -> str:
    return f"{v:.3f}"


def fmt_ratio(v: float) -> str:
    return "∞" if not np.isfinite(v) else f"{v:.2f} : 1"


def fmt_size_kb(bytes_: float) -> str:
    kb = bytes_ / 1024.0
    return f"{kb:.2f} KB" if kb < 1024 else f"{kb / 1024.0:.2f} MB"


def fmt_reduction(v: float) -> str:
    """Boyut azalması yüzdesi — quality.calculate_metrics()'in TEK
    kaynaklı size_reduction_pct alanından gelir (mega-spec "FINAL FEATURE
    PASS" Part 7/18: ikinci bir formülle yeniden hesaplanmaz)."""
    return "N/A" if np.isnan(v) else f"%{v:.1f}"


def fmt_sci(v: float) -> str:
    return f"{v:.2e}"


def _esc(s: str) -> str:
    return _html.escape(str(s))


# ---------------------------------------------------------------------------
# Delta hesabı — yön anlamlıysa renklendir, değilse nötr (Part 32)
# ---------------------------------------------------------------------------
def _delta_span(delta: float, unit: str, higher_is_better: bool | None,
                decimals: int = 2) -> str:
    if higher_is_better is None:
        color = TEXT_MUTED
    elif (delta > 0) == higher_is_better:
        color = POSITIVE
    elif delta == 0:
        color = TEXT_MUTED
    else:
        color = WARNING
    sign = "+" if delta >= 0 else ""
    return (f'<span style="color:{color};font-size:11px;font-family:var(--lab-mono)">'
           f'{sign}{delta:.{decimals}f}{unit}</span>')


# ---------------------------------------------------------------------------
# Metrik kartı: JPEG/DCT, Wavelet/DWT, JPEG2000 için ortak
# ---------------------------------------------------------------------------
def _metric_row(label: str, value: str, delta: str | None = None) -> str:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return (f'<div class="metric-row"><span class="metric-label">{label}</span>'
           f'<span class="metric-value-wrap"><span class="metric-value">{value}</span>'
           f'{delta_html}</span></div>')


def compact_metric_card(
    title: str, accent: str, metrics: dict, *, compare_to: dict | None = None,
    param_label: str = "",
) -> str:
    """Birincil (Level-1) görünüm için: yalnız PSNR/SSIM/BPP (mega-spec
    Part 5 — "Show only PSNR/SSIM/BPP on the primary screen"). MSE/oran/
    boyut/entropi-etiketi gibi ayrıntılar compact değil metric_card()'a,
    yani "Detaylar" panosuna aittir — aynı `metrics` sözlüğünden, çift
    hesaplama olmadan üretilir."""
    d_psnr = d_ssim = d_bpp = None
    if compare_to is not None and np.isfinite(metrics["psnr"]) and np.isfinite(compare_to["psnr"]):
        d_psnr = _delta_span(metrics["psnr"] - compare_to["psnr"], " dB", True)
        d_ssim = _delta_span(metrics["ssim"] - compare_to["ssim"], "", True, 4)
        d_bpp = _delta_span(metrics["bpp"] - compare_to["bpp"], "", None, 3)
    param_html = f'<div class="metric-param">{_esc(param_label)}</div>' if param_label else ""
    rows = "".join([
        _metric_row("PSNR", fmt_psnr(metrics["psnr"]), d_psnr),
        _metric_row("SSIM", fmt_ssim(metrics["ssim"]), d_ssim),
        _metric_row("BPP", fmt_bpp(metrics["bpp"]), d_bpp),
    ])
    return f"""
<div class="lab-card metric-card" style="border-top:3px solid {accent}">
  <div class="metric-card-head">
    <span class="method-dot" style="background:{accent}"></span>
    <span class="metric-title">{_esc(title)}</span>
  </div>
  {param_html}
  {rows}
</div>"""


def target_rate_header_html(target_bpp: float) -> str:
    """Birincil ekranda hedef oranı TEK büyük değer olarak gösterir; bisection
    ayrıntısı/eşlenen noktalar target_summary_html() ile Detaylar'a taşınır."""
    return f"""
<div class="target-rate-header">
  <span class="target-rate-label">HEDEF ORAN</span>
  <span class="target-rate-value">{fmt_bpp(target_bpp)}<span class="unit"> bpp</span></span>
</div>"""


def category_badge_html(label: str) -> str:
    return f'<span class="category-badge">Kategori: {_esc(label)}</span>'


def metric_card(
    title: str, accent: str, metrics: dict, *, compare_to: dict | None = None,
    size_badge: str | None = None, param_label: str = "",
) -> str:
    """metrics: calculate_metrics() çıktısı. compare_to verilirse Δ satırları
    eklenir (örn. Wavelet kartında JPEG'e göre fark). Bu, "Detaylar" panosu
    içindir — MSE/sıkıştırma oranı/boyut gibi ikincil metrikleri içerir."""
    row = _metric_row
    d_psnr = d_ssim = d_bpp = None
    if compare_to is not None and np.isfinite(metrics["psnr"]) and np.isfinite(compare_to["psnr"]):
        d_psnr = _delta_span(metrics["psnr"] - compare_to["psnr"], " dB", True)
        d_ssim = _delta_span(metrics["ssim"] - compare_to["ssim"], "", True, 4)
        d_bpp = _delta_span(metrics["bpp"] - compare_to["bpp"], "", None, 3)

    badge_html = f'<span class="size-badge">{_esc(size_badge)}</span>' if size_badge else ""
    param_html = f'<div class="metric-param">{_esc(param_label)}</div>' if param_label else ""

    rows = "".join([
        row("PSNR", fmt_psnr(metrics["psnr"]), d_psnr),
        row("SSIM", fmt_ssim(metrics["ssim"]), d_ssim),
        row("MSE", fmt_mse(metrics["mse"])),
        row("BPP", fmt_bpp(metrics["bpp"]), d_bpp),
        row("SIKIŞTIRMA ORANI", fmt_ratio(metrics["compression_ratio"])),
        row("AZALMA", fmt_reduction(metrics["size_reduction_pct"])),
        row("BOYUT", fmt_size_kb(metrics["compressed_size_bytes"])),
    ])
    return f"""
<div class="lab-card metric-card" style="border-top:3px solid {accent}">
  <div class="metric-card-head">
    <span class="method-dot" style="background:{accent}"></span>
    <span class="metric-title">{_esc(title)}</span>
    {badge_html}
  </div>
  {param_html}
  {rows}
</div>"""


def target_summary_html(target_bpp: float, points: list[tuple[str, str, float]]) -> str:
    """points: [(label, color, actual_bpp), ...] — eşlenen çalışma noktaları."""
    rows = "".join(
        f'<div class="target-row"><span class="method-dot" style="background:{color}"></span>'
        f'<span class="target-label">{_esc(label)}</span>'
        f'<span class="target-value">{fmt_bpp(bpp)} bpp</span></div>'
        for label, color, bpp in points
    )
    diff = max(bpp for _, _, bpp in points) - min(bpp for _, _, bpp in points) if points else 0.0
    return f"""
<div class="lab-card target-card">
  <div class="section-label">HEDEF ORAN</div>
  <div class="target-value-hero">{fmt_bpp(target_bpp)} <span class="unit">bpp</span></div>
  <div class="target-caption">bisection ile YAKLAŞIK eşlenir — gerçekleşen değer hedefe eşit olmak zorunda değildir</div>
  {rows}
  <div class="target-row target-diff"><span class="target-label">Fark</span>
  <span class="target-value">{fmt_bpp(diff)} bpp</span></div>
</div>"""


def status_badge(text: str, kind: str = "neutral") -> str:
    colors = {"positive": POSITIVE, "warning": WARNING, "critical": CRITICAL, "neutral": TEXT_MUTED}
    color = colors.get(kind, TEXT_MUTED)
    return (f'<span class="status-badge" style="color:{color};border-color:{color}">'
           f'{_esc(text)}</span>')


def validation_card_html(max_err: float, mse_val: float, tol: float = 1e-3) -> str:
    passed = max_err < tol
    badge = status_badge("PASS" if passed else "FAIL", "positive" if passed else "critical")
    return f"""
<div class="lab-card validation-card">
  <div class="section-label">TRANSFORM DOĞRULAMA — DWT → IDWT (kuantalama yok)</div>
  <div class="validation-head">{badge}</div>
  <div class="metric-row"><span class="metric-label">Azami Mutlak Hata</span>
  <span class="metric-value mono">{fmt_sci(max_err)}</span></div>
  <div class="metric-row"><span class="metric-label">MSE</span>
  <span class="metric-value mono">{fmt_sci(mse_val)}</span></div>
  <div class="metric-param">Tolerans: {fmt_sci(tol)} — sayısal hassasiyet düzeyinde hata, dönüşümün tersinir olduğunu doğrular.</div>
</div>"""


def sparsity_card_html(stats: dict) -> str:
    return f"""
<div class="lab-card sparsity-card">
  <div class="section-label">KATSAYI SEYREKLİĞİ</div>
  <div class="sparsity-compare">
    <div class="sparsity-col">
      <div class="sparsity-col-title">KUANTALAMA ÖNCESİ (ham)</div>
      <div class="sparsity-big">{stats['raw_nonzero_pct']:.1f}%</div>
      <div class="metric-label">sıfır olmayan</div>
    </div>
    <div class="sparsity-arrow">→</div>
    <div class="sparsity-col">
      <div class="sparsity-col-title">KUANTALAMA SONRASI</div>
      <div class="sparsity-big" style="color:{ACCENT_PURPLE}">{stats['nonzero_pct']:.1f}%</div>
      <div class="metric-label">sıfır olmayan</div>
    </div>
  </div>
  <div class="metric-row"><span class="metric-label">Sparsity (sıfır oranı)</span>
  <span class="metric-value mono">{stats['sparsity_pct']:.1f}%</span></div>
  <div class="metric-row"><span class="metric-label">Toplam katsayı</span>
  <span class="metric-value mono">{stats['total']:,}</span></div>
  <div class="metric-param">Ham float katsayılar neredeyse hiç tam sıfır olmaz; sıkıştırmayı
  açıklayan seyreklik KUANTALAMA SONRASI ortaya çıkar.</div>
</div>"""


def _fmt_num(v: float, decimals: int) -> str:
    """NaN'ı 'N/A' olarak gösterir (numerical-correctness audit) — Python'un
    varsayılan `{nan:.2f}` davranışı ham "nan" metni üretir, bu da bir
    ölçüm gibi görünüp yanıltıcı olabilir. Bu yardımcı tradeoff kartlarının
    (foreground kazancı / background ödünü) TÜMÜ tarafından kullanılır."""
    return "N/A" if (isinstance(v, float) and np.isnan(v)) else f"{v:.{decimals}f}"


def tradeoff_card_html(title: str, accent: str, baseline: float, semantic: float,
                       unit: str, higher_is_better: bool, decimals: int = 2) -> str:
    delta = semantic - baseline
    undefined = np.isnan(delta)
    good = (not undefined) and (delta > 0) == higher_is_better
    color = TEXT_MUTED if undefined else (POSITIVE if good else WARNING)
    sign = "" if undefined or delta < 0 else "+"
    delta_str = "N/A" if undefined else f"{sign}{delta:.{decimals}f}{unit}"
    return f"""
<div class="lab-card tradeoff-card" style="border-top:3px solid {accent}">
  <div class="section-label">{_esc(title)}</div>
  <div class="tradeoff-row"><span class="metric-label">Baseline</span>
  <span class="metric-value mono">{_fmt_num(baseline, decimals)}{unit}</span></div>
  <div class="tradeoff-row"><span class="metric-label">Semantic</span>
  <span class="metric-value mono">{_fmt_num(semantic, decimals)}{unit}</span></div>
  <div class="tradeoff-delta" style="color:{color}">{delta_str}</div>
</div>"""


def multi_tradeoff_card_html(title: str, accent: str,
                             rows: list[tuple[str, float, float, str, bool, int]]) -> str:
    """rows: [(etiket, baseline, semantic, birim, higher_is_better, ondalık), ...]
    Birden çok metriği (örn. FG PSNR + FG SSIM) tek kartta gösterir."""
    body = ""
    for label, base, sem, unit, higher_is_better, dec in rows:
        delta = sem - base
        undefined = np.isnan(delta)
        good = (not undefined) and (delta > 0) == higher_is_better
        color = TEXT_MUTED if undefined else (POSITIVE if good else WARNING)
        sign = "" if undefined or delta < 0 else "+"
        delta_str = "N/A" if undefined else f"{sign}{delta:.{dec}f}{unit}"
        body += f"""
  <div class="tradeoff-metric">
    <div class="metric-label">{_esc(label)}</div>
    <div class="tradeoff-row"><span class="metric-label">Baseline</span>
    <span class="metric-value mono">{_fmt_num(base, dec)}{unit}</span></div>
    <div class="tradeoff-row"><span class="metric-label">Semantic</span>
    <span class="metric-value mono">{_fmt_num(sem, dec)}{unit}</span></div>
    <div class="tradeoff-delta" style="color:{color};font-size:15px">{delta_str}</div>
  </div>"""
    return f"""
<div class="lab-card tradeoff-card" style="border-top:3px solid {accent}">
  <div class="section-label">{_esc(title)}</div>
  {body}
</div>"""


def same_budget_badge_html(baseline_bpp: float, semantic_bpp: float) -> str:
    diff = abs(semantic_bpp - baseline_bpp)
    return f"""
<div class="lab-card same-budget-card">
  <div class="section-label">AYNI BİT BÜTÇESİ</div>
  <div class="budget-row">
    <div class="budget-col"><div class="metric-label">Baseline</div>
    <div class="metric-value mono">{fmt_bpp(baseline_bpp)} bpp</div></div>
    <div class="budget-col"><div class="metric-label">Semantic</div>
    <div class="metric-value mono">{fmt_bpp(semantic_bpp)} bpp</div></div>
    <div class="budget-col"><div class="metric-label">Fark</div>
    <div class="metric-value mono">{fmt_bpp(diff)} bpp</div></div>
  </div>
</div>"""


def empty_state_html(text: str, caption: str = "") -> str:
    caption_html = f'<div class="empty-caption">{_esc(caption)}</div>' if caption else ""
    return f"""
<div class="lab-card empty-state">
  <div class="empty-title">{_esc(text)}</div>
  {caption_html}
</div>"""


def measurement_strip_html(items: list[tuple[str, str]]) -> str:
    """Bilimsel cihaz okuması gibi TEK yatay şerit (kart yığını değil) —
    mega-spec Part 16. items: [(etiket, değer), ...]."""
    cells = "".join(
        f'<div class="measurement-item"><span class="measurement-label">{_esc(l)}</span>'
        f'<span class="measurement-value">{_esc(v)}</span></div>'
        for l, v in items
    )
    return f'<div class="measurement-strip">{cells}</div>'


def selected_block_line_html(block_size: int, row: int, col: int,
                             region: tuple[int, int, int, int] | None = None) -> str:
    """'BLOK 8×8 · Satır 48 · Sütun 43' — mega-spec Part 14."""
    txt = f"{block_size}×{block_size} · Satır {row} · Sütun {col}"
    if region:
        x0, y0, x1, y1 = region
        txt += f" · piksel x{x0}–{x1}, y{y0}–{y1}"
    return f'<div class="selected-block-line"><span class="sb-label">Seçili Blok</span>{_esc(txt)}</div>'


def filter_bank_flow_html(wavelet_name: str) -> str:
    """DWT analiz filtre bankasının 2D ayrılabilir (separable) yapısını
    gösteren ÖZGÜN HTML/CSS akış şeması — satır filtreleme → alt örnekleme →
    sütun filtreleme → alt örnekleme → LL/LH/HL/HH (mega-spec Part 13:
    "statik resim olmasın... hover kısa açıklama versin"; CSS-only hover
    tooltip kullanılır, JS gerekmez). Arkadaş projesindeki mimari diyagram
    KOPYALANMAMIŞTIR — kendi kart/ok dilimiz kullanılır."""
    def _node(label, tip, extra_cls=""):
        return (f'<div class="flow-node {extra_cls}">{_esc(label)}'
               f'<span class="flow-tip">{_esc(tip)}</span></div>')

    def _arrow():
        return '<div class="flow-arrow">↓</div>'

    return f"""
<div class="filter-flow">
  {_node("KAYNAK GÖRÜNTÜ", "Analiz edilen görüntü/kanal")}
  {_arrow()}
  <div class="flow-row">
    {_node(f"Satır: alçak-geçiren h", f"{wavelet_name} analiz filtresi — satırlar boyunca", "flow-lo")}
    {_node(f"Satır: yüksek-geçiren g", f"{wavelet_name} analiz filtresi — satırlar boyunca", "flow-hi")}
  </div>
  {_arrow()}
  <div class="flow-row">
    {_node("↓2 (sütun)", "Sütunlarda 2 kat alt örnekleme")}
    {_node("↓2 (sütun)", "Sütunlarda 2 kat alt örnekleme")}
  </div>
  {_arrow()}
  <div class="flow-row flow-row-4">
    {_node("Sütun: alçak → LL", "Yaklaşım (düşük frekans)", "flow-ll")}
    {_node("Sütun: yüksek → LH", "Yatay detay", "flow-lh")}
    {_node("Sütun: alçak → HL", "Dikey detay", "flow-hl")}
    {_node("Sütun: yüksek → HH", "Köşegen detay", "flow-hh")}
  </div>
  {_arrow()}
  <div class="flow-row flow-row-4">
    <span class="flow-band flow-ll">LL</span>
    <span class="flow-band flow-lh">LH</span>
    <span class="flow-band flow-hl">HL</span>
    <span class="flow-band flow-hh">HH</span>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# DWT Ayrıştırma Gezgini — seçili bant kartı + sabit açıklama şeridi
# (mega-spec "DWT LAB — FIX AND UPGRADE THE DECOMPOSITION EXPLORER")
# ---------------------------------------------------------------------------
_DWT_BAND_TITLES = {
    "LL": "Approximation — düşük frekans / genel görüntü yapısı",
    "LH": "Yatay detay (yatay-yönelimli kenarlar) — pywt 'horizontal detail'",
    "HL": "Dikey detay (dikey-yönelimli kenarlar) — pywt 'vertical detail'",
    "HH": "Köşegen / yüksek-frekans detay",
}


def dwt_band_legend_html() -> str:
    """LL/LH/HL/HH'nin ne temsil ettiğini AÇIKÇA (hover'a bağlı olmadan)
    gösteren sabit şerit — mega-spec Part 13: yön kuralı (LH=yatay,
    HL=dikey) pywt'nin kendi cH/cV semantiğine göre ampirik olarak
    doğrulanmıştır (bkz. tests/test_correctness.py
    test_dwt_lh_hl_orientation_matches_pywt_semantics)."""
    chips = "".join(
        f'<span class="dwt-legend-chip"><b>{name}</b>: {title}</span>'
        for name, title in _DWT_BAND_TITLES.items()
    )
    return f'<div class="dwt-legend">{chips}</div>'


def dwt_band_details_html(
    level: int, band_name: str, stats: dict, pixel: tuple[int, int, float] | None = None,
) -> str:
    """Seçili TEK bandın (mega-spec Part 12/14/20) gerçek istatistikleri —
    tıklanan bir piksel varsa (Part 14) onun ham katsayı değeri de
    gösterilir. `stats`, subbands.selected_band_stats()'ın GERÇEK
    çıktısıdır; hiçbir değer burada uydurulmaz."""
    pixel_html = ""
    if pixel is not None:
        row, col, val = pixel
        # Küçük, okunabilir bilgi kutusu (mega-spec "DWT LAB — subband
        # görüntülerine tıklama ile katsayı inceleme"): X=sütun, Y=satır,
        # görüntü koordinat kuralıyla (evt.index) tutarlı; val DOĞRUDAN
        # ham katsayı dizisinden (band[row][col]) — önizlemenin
        # normalize edilmiş piksel değeri DEĞİL.
        pixel_html = f"""
<div class="dwt-pixel-box">
  <div class="dwt-pixel-title">{_esc(band_name)}{level}</div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">X</span><span class="dwt-pixel-val">{col}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Y</span><span class="dwt-pixel-val">{row}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Katsayı</span><span class="dwt-pixel-val">{val:+.4f}</span></div>
</div>"""
    return f"""
<div class="lab-card">
  <div class="metric-card-head">
    <span class="metric-title">{_esc(band_name)}{level} — {_esc(_DWT_BAND_TITLES.get(band_name, ""))}</span>
  </div>
  <div class="metric-row"><span class="metric-label">Boyut</span>
  <span class="metric-value mono">{_esc(stats["shape"])}</span></div>
  <div class="metric-row"><span class="metric-label">Katsayı sayısı</span>
  <span class="metric-value mono">{stats["count"]:,}</span></div>
  <div class="metric-row"><span class="metric-label">Min / Max</span>
  <span class="metric-value mono">{stats["min"]:.2f} / {stats["max"]:.2f}</span></div>
  <div class="metric-row"><span class="metric-label">Ortalama</span>
  <span class="metric-value mono">{stats["mean"]:.3f}</span></div>
  <div class="metric-row"><span class="metric-label">Std sapma</span>
  <span class="metric-value mono">{stats["std"]:.3f}</span></div>
  <div class="metric-row"><span class="metric-label">Enerji (Σ katsayı²)</span>
  <span class="metric-value mono">{stats["energy"]:.2f}</span></div>
  <div class="metric-row"><span class="metric-label">Sıfıra-yakın oranı</span>
  <span class="metric-value mono">{stats["zero_pct"]:.1f}%</span></div>
  {pixel_html}
</div>"""


def mosaic_pixel_html(level: int, band: str, row: int, col: int, value: float) -> str:
    """Piramit Katsayı Haritası'nda tıklanan noktanın bilgi kutusu (mega-
    spec "DWT LAB — TIKLAMA İLE NOKTA İNCELEMEYİ 3 GÖRSELDE AKTİF ET"
    Part 1) — `value` DOĞRUDAN ham DWT katsayı dizisinden gelir, mozaiğin
    normalize edilmiş piksel değeri DEĞİL."""
    return f"""
<div class="dwt-pixel-box">
  <div class="dwt-pixel-title">{_esc(band)}{level}</div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Subband</span><span class="dwt-pixel-val">{_esc(band)}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Level</span><span class="dwt-pixel-val">{level}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">X</span><span class="dwt-pixel-val">{col}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Y</span><span class="dwt-pixel-val">{row}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Katsayı</span><span class="dwt-pixel-val">{value:+.4f}</span></div>
</div>"""


def recon_pixel_html(row: int, col: int, original: float, reconstructed: float) -> str:
    """Kuantalamalı Rekonstrüksiyon / Fark Haritası için PAYLAŞILAN bilgi
    kutusu (mega-spec Part 2/3) — original/reconstructed DOĞRUDAN gerçek
    (gray, recon) dizilerinden okunur; fark = original - reconstructed
    (subbands.reconstruction_diff_image ile AYNI işaret kuralı — UI'da
    kullanılan tanım burada da BİREBİR aynıdır)."""
    diff = original - reconstructed
    sign = "+" if diff >= 0 else ""
    return f"""
<div class="dwt-pixel-box">
  <div class="dwt-pixel-title">Piksel ({col}, {row})</div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">X</span><span class="dwt-pixel-val">{col}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Y</span><span class="dwt-pixel-val">{row}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Orijinal</span><span class="dwt-pixel-val">{original:.2f}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Rekonstrüksiyon</span><span class="dwt-pixel-val">{reconstructed:.2f}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Fark (orijinal − rekonstrüksiyon)</span><span class="dwt-pixel-val">{sign}{diff:.2f}</span></div>
</div>"""


def summary_bar_html(items: list[tuple[str, str]]) -> str:
    """Kompakt tek satır özet: [(etiket, değer), ...] — DCT/DWT Lab üst bar."""
    chips = "".join(
        f'<div class="summary-chip"><span class="summary-chip-label">{_esc(l)}</span>'
        f'<span class="summary-chip-value mono">{_esc(v)}</span></div>'
        for l, v in items
    )
    return f'<div class="summary-bar">{chips}</div>'


# ---------------------------------------------------------------------------
# Yapısal redesign bileşenleri: boş durum / görüntü çipi / oran adaleti /
# kalite şeridi — kart-içinde-kart yerine (mega-spec "structural redesign").
# ---------------------------------------------------------------------------
def hero_empty_html(title: str, caption: str) -> str:
    """Karşılaştır sekmesinin İLK açılış durumu: tek ortalanmış blok, büyük
    boş sonuç panelleri YOK. Seç/Yükle butonları bu HTML'in dışında, gerçek
    Gradio Button bileşenleri olarak eklenir (etkileşim gerektiği için)."""
    return f"""
<div class="hero-empty">
  <div class="hero-title">{_esc(title)}</div>
  <div class="hero-caption">{_esc(caption)}</div>
</div>"""


def active_image_bar_html(
    name: str, category_label: str | None, width: int, height: int,
    original_width: int | None = None, original_height: int | None = None,
    file_size_bytes: int | None = None,
) -> str:
    """Uygulama genelinde TEK, paylaşılan 'AKTİF GÖRÜNTÜ' çubuğu içeriği
    (mega-spec 'FINAL STATE/CALLBACK FIX' Part 21) — Compare/DCT/DWT/
    Semantic sekmelerinin tümü aynı görüntüyü kullandığını burada görür.

    original_width/height: `_prepare()` (app.py) büyük görüntüleri hıza
    karşı küçültür — bu SESSİZCE gizlenmiyordu (mega-spec "FINAL
    MATHEMATICAL VALIDATION & AUDIT" Part 3). (width,height) HER ZAMAN
    gerçek analiz çözünürlüğüdür; orijinal farklıysa (yalnız o zaman,
    gereksiz UI kalabalığı olmadan — Part 41) ek bir not gösterilir.

    file_size_bytes: kullanıcı isteğiyle (bkz. çağrı yeri) artık HİÇBİR
    yerde GÖSTERİLMİYOR — parametre yalnız geriye dönük uyumluluk için
    duruyor, kasıtlı olarak yok sayılıyor."""
    cat_html = f'<span class="chip-category">{_esc(category_label)}</span>' if category_label else ""
    resized_note = ""
    if original_width and original_height and (original_width, original_height) != (width, height):
        resized_note = (
            f'<span class="active-image-resized" '
            f'title="Analiz için hıza karşı küçültüldü — orijinal dosya {original_width}×{original_height}">'
            f'(orijinal {original_width}×{original_height})</span>'
        )
    file_size_html = ""
    return f"""
<div class="active-image-bar-content">
  <span class="active-image-tag">AKTİF GÖRÜNTÜ</span>
  <span class="chip-name">{_esc(name)}</span>
  {cat_html}
  <span class="active-image-dims">{width}×{height}</span>
  {resized_note}
  {file_size_html}
</div>"""


def image_chip_html(name: str, category_label: str | None) -> str:
    cat_html = f'<span class="chip-category">{_esc(category_label)}</span>' if category_label else ""
    return f"""
<div class="image-chip">
  <span class="chip-name">{_esc(name)}</span>
  {cat_html}
</div>"""


def rate_fairness_html(target_bpp: float, dct_bpp: float, wav_bpp: float) -> str:
    """Tek satırlık oran-adaleti özeti: TARGET/JPEG/JPEG2000/Δ — mega-spec
    Part 10, bisection ayrıntısı olmadan (o Detaylar'a taşınmıştır).

    Terminoloji notu (mega-spec "ANA COMPARE TERMİNOLOJİ GÜNCELLEMESİ"):
    bu fonksiyon YALNIZ ana Karşılaştır ekranında (app.run_main_comparison)
    kullanılır — DWT Lab/Semantik ROI bunu ÇAĞIRMAZ; ikinci taraf artık
    gerçek JPEG2000/OpenJPEG kodeği olduğu için etiketler "Wavelet" değil
    "JPEG2000". Sayısal hesap (target_bpp/dct_bpp/wav_bpp/diff) DEĞİŞMEDİ."""
    diff = abs(wav_bpp - dct_bpp)
    def item(label, value, cls=""):
        return (f'<div class="rf-item"><span class="rf-label">{_esc(label)}</span>'
               f'<span class="rf-value {cls}">{fmt_bpp(value)} bpp</span></div>')
    return f"""
<div class="rate-fairness">
  {item("Hedef", target_bpp)}
  {item("JPEG", dct_bpp, "accent-jpeg")}
  {item("JPEG2000", wav_bpp, "accent-wavelet")}
  {item("Fark |JPEG2000−JPEG|", diff)}
</div>"""


def quality_strip_html(dct_metrics: dict, wav_metrics: dict, wavelet_name: str = "JPEG2000 / DWT") -> str:
    """Kart yığını yerine TEK karşılaştırma şeridi: Metrik | JPEG | JPEG2000 | Δ.
    Yalnız PSNR/SSIM/BPP (birincil ekran) — MSE/oran/boyut Detaylar'dadır.
    Δ = JPEG2000 − JPEG işaret kuralı; başlıkta AÇIKÇA belirtilir (mega-spec
    "FINAL PRE-PRESENTATION QA" Part 18) — önceden yalnız "Δ" yazıyordu,
    işaretin yönü belirsizdi. (Terminoloji notu: bu fonksiyon YALNIZ ana
    Karşılaştır ekranında kullanılır — DWT Lab/Semantik ROI'yi etkilemez.)"""
    def delta(key, decimals, unit, higher_better):
        d = wav_metrics[key] - dct_metrics[key]
        if higher_better is None:
            color = TEXT_MUTED
        elif d == 0:
            color = TEXT_MUTED
        else:
            color = POSITIVE if (d > 0) == higher_better else WARNING
        sign = "+" if d >= 0 else ""
        return f'<td class="delta" style="color:{color}">{sign}{d:.{decimals}f}{unit}</td>'

    def row(label, dct_val, wav_val, delta_html):
        return f'<tr><td>{label}</td><td>{dct_val}</td><td>{wav_val}</td>{delta_html}</tr>'

    rows = "".join([
        row("PSNR", fmt_psnr(dct_metrics["psnr"]), fmt_psnr(wav_metrics["psnr"]),
            delta("psnr", 2, " dB", True)),
        row("SSIM", fmt_ssim(dct_metrics["ssim"]), fmt_ssim(wav_metrics["ssim"]),
            delta("ssim", 4, "", True)),
        row("BPP", fmt_bpp(dct_metrics["bpp"]), fmt_bpp(wav_metrics["bpp"]),
            delta("bpp", 3, "", None)),
    ])
    return f"""
<table class="quality-strip">
  <thead><tr><th></th><th class="th-jpeg">JPEG / DCT</th>
  <th class="th-wavelet">{_esc(wavelet_name)}</th><th title="JPEG2000 − JPEG">Δ (JPEG2000 − JPEG)</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


# =============================================================================
# FINAL FEATURE PASS — Sıkıştırma Özeti (Part 2-9) ve Önce/Sonra kaydırıcısı
# (Part 10-16). Yeni sayısal değer/formül YOK: her ikisi de
# quality.calculate_metrics()'in (TEK kaynak) çıktısını okur veya zaten
# hesaplanmış rekonstrüksiyon dizilerini gösterir.
# =============================================================================
def compression_summary_card_html(
    title: str, accent: str, metrics: dict, size_badge: str,
) -> str:
    """Kompakt SIKIŞTIRMA ÖZETİ kartı: PSNR/SSIM/MSE/BPP/SIKIŞTIRMA/AZALMA
    KPI satırı + ORİJİNAL → SIKIŞTIRILMIŞ boyut oku. `metrics`,
    calculate_metrics()'in TEK kaynaklı çıktısıdır — burada hiçbir değer
    yeniden hesaplanmaz; metric_card (Teknik Detaylar) ile BİREBİR aynı
    sayıları taşır. `size_badge`: çağıran taraf EngineResult.is_real_codec'e
    göre 'GERÇEK BOYUT' / 'ENTROPİ TAHMİNİ' seçer — gerçek bitstream'den mi
    yoksa entropi tahmininden mi geldiği HER ZAMAN açıkça işaretlenir."""
    def kpi(label: str, value: str) -> str:
        return (f'<div class="kpi-box"><div class="kpi-label">{_esc(label)}</div>'
               f'<div class="kpi-value">{value}</div></div>')

    kpis = "".join([
        kpi("PSNR", fmt_psnr(metrics["psnr"])),
        kpi("SSIM", fmt_ssim(metrics["ssim"])),
        kpi("MSE", fmt_mse(metrics["mse"])),
        kpi("BPP", fmt_bpp(metrics["bpp"])),
        kpi("SIKIŞTIRMA", fmt_ratio(metrics["compression_ratio"])),
        kpi("AZALMA", fmt_reduction(metrics["size_reduction_pct"])),
    ])
    return f"""
<div class="lab-card kpi-summary-card" style="border-top:3px solid {accent}">
  <div class="metric-card-head">
    <span class="method-dot" style="background:{accent}"></span>
    <span class="metric-title">{_esc(title)}</span>
    <span class="size-badge">{_esc(size_badge)}</span>
  </div>
  <div class="kpi-row">{kpis}</div>
  <div class="size-arrow-row">
    <span class="size-arrow-value">{fmt_size_kb(metrics["original_size_bytes"])}</span>
    <span class="size-arrow-sep">→</span>
    <span class="size-arrow-value" style="color:{accent}">{fmt_size_kb(metrics["compressed_size_bytes"])}</span>
    <span class="size-reduction-pct">{fmt_reduction(metrics["size_reduction_pct"])} azalma</span>
  </div>
</div>"""


def _image_to_data_uri(arr: np.ndarray) -> str:
    """Bir (RGB veya gri) numpy görüntüsünü PNG data URI'sine kodlar —
    Before/After kaydırıcısı BUNU tarayıcıya doğrudan gömer; ek bir HTTP
    isteği veya disk dosyası olmadığından slider hareketi hiçbir backend
    çağrısı GEREKTİRMEZ (mega-spec "FINAL FEATURE PASS" Part 14)."""
    a = np.clip(arr, 0, 255).astype(np.uint8)
    encode_src = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", encode_src)
    if not ok:
        raise ValueError("Görüntü PNG'ye kodlanamadı.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def before_after_slider_html(
    original: np.ndarray, reconstructed: np.ndarray,
    left_label: str, right_label: str, accent: str = ACCENT_CYAN,
    position: float = 50.0,
) -> str:
    """İnteraktif ÖNCE/SONRA kaydırıcısı (mega-spec "FINAL FEATURE PASS"
    Part 10-15). Saf CSS + tek bir native <input type=range>: sürükleme
    SIRASINDA hiçbir backend çağrısı yapılmaz, yalnız zaten hesaplanmış iki
    PNG arasında CSS clip-path ile geçiş yapılır (Part 14) — bu yüzden
    akıcıdır ve dokunmatik/fare ile native olarak çalışır (Part 15).

    Pixel-aligned (Part 11): rekonstrüksiyon HER ZAMAN orijinalle birebir
    aynı (H,W) şeklindedir (bkz. test_reconstruction_dimensions_match_
    original) — konteynerin aspect-ratio'su GERÇEK (w/h) oranından
    hesaplanır ve iki görüntü de aynı mutlak-konumlu kutunun içine %100
    genişlik/yükseklikle yerleştirilir; iki farklı <img> öğesinin bağımsız
    ölçeklenmesinden kaynaklanan sahte bir kalite/kayma farkı oluşamaz."""
    h, w = original.shape[:2]
    aspect = (w / h) if h else 1.0
    before_uri = _image_to_data_uri(original)
    after_uri = _image_to_data_uri(reconstructed)
    pos = max(0.0, min(100.0, float(position)))
    return f"""
<div class="ba-wrap">
  <div class="ba-frame" style="--ba-ar:{aspect:.6f}">
    <img class="ba-after" src="{after_uri}" alt="{_esc(right_label)}">
    <div class="ba-clip" style="--ba-pos:{pos:.3f}%">
      <img class="ba-before" src="{before_uri}" alt="{_esc(left_label)}">
    </div>
    <div class="ba-divider" style="left:{pos:.3f}%"></div>
    <span class="ba-tag ba-tag-left">{_esc(left_label)}</span>
    <span class="ba-tag ba-tag-right">{_esc(right_label)}</span>
  </div>
  <input class="ba-range" type="range" min="0" max="100" step="0.1" value="{pos:.3f}"
    style="--ba-accent:{accent}"
    aria-label="Önce / sonra karşılaştırma kaydırıcısı — {_esc(left_label)} / {_esc(right_label)}"
    oninput="
      var frame = this.previousElementSibling;
      var clip = frame.querySelector('.ba-clip');
      var div = frame.querySelector('.ba-divider');
      clip.style.setProperty('--ba-pos', this.value + '%');
      div.style.left = this.value + '%';
    ">
</div>"""
