# -*- coding: utf-8 -*-
"""DCT Explorer: tek bir bloğun orijinal -> seviye-kaydırma -> DCT ->
kuantalama -> rekonstrüksiyon adımlarını GERÇEK görüntü verisinden hesaplar.

Sabit/dekoratif matris yok: her çağrı, kullanıcının yüklediği görüntüden
seçilen gerçek bloğu işler (mega-spec: "Do not use hard-coded matrices").
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

from src.engines.dct_engine import get_qtable, quality_to_qtable
from src.engines.entropy import zigzag_indices


def extract_block(gray: np.ndarray, block_size: int, row: int, col: int) -> np.ndarray:
    """gray görüntüden (row,col) indeksli block_size x block_size bloğu keser
    (kenar-tekrarı ile doldurulmuş ızgara üzerinden — motorla aynı yöntem)."""
    h, w = gray.shape
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    padded = np.pad(gray, ((0, ph), (0, pw)), mode="edge")
    nh, nw = padded.shape[0] // block_size, padded.shape[1] // block_size
    row = max(0, min(row, nh - 1))
    col = max(0, min(col, nw - 1))
    r0, c0 = row * block_size, col * block_size
    return padded[r0:r0 + block_size, c0:c0 + block_size], (nh, nw), (row, col)


def block_pixel_region(gray_shape: tuple[int, int], block_size: int,
                       row: int, col: int) -> tuple[int, int, int, int]:
    """(row,col) blok-ızgara indeksinden GERÇEK piksel bölgesini hesaplar
    (x0,y0,x1,y1) — extract_block ile AYNI r0=row*block_size, c0=col*block_size
    eşlemesi (mega-spec Part 11: "must correspond to exact block used by
    backend"). Orijinal (doldurulmamış) görüntü sınırlarına kırpılır."""
    h, w = gray_shape
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    nh, nw = (h + ph) // block_size, (w + pw) // block_size
    row = max(0, min(row, nh - 1))
    col = max(0, min(col, nw - 1))
    x0, y0 = col * block_size, row * block_size
    x1, y1 = min(x0 + block_size, w), min(y0 + block_size, h)
    return x0, y0, x1, y1


def draw_block_overlay(image: np.ndarray, block_size: int, row: int, col: int) -> np.ndarray:
    """Kaynak görüntü üzerine seçili DCT bloğunu gösteren camgöbeği dikdörtgen
    çizer. extract_block ile BİREBİR aynı koordinat hesabını kullanır —
    dekoratif değil, gerçek analiz bölgesidir (mega-spec Part 11)."""
    import cv2

    h, w = image.shape[:2]
    x0, y0, x1, y1 = block_pixel_region((h, w), block_size, row, col)
    out = image.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    thickness = max(2, min(3, block_size // 4))
    cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)),
                 (49, 200, 255), thickness)
    return out


def inspect_block(
    gray: np.ndarray, block_size: int, row: int, col: int, quality: float,
    base_table: np.ndarray,
) -> dict:
    """Gerçek bir bloğu uçtan uca işler; tüm ara adımları sayısal olarak döner."""
    block, grid_shape, (row, col) = extract_block(gray, block_size, row, col)
    shifted = block.astype(np.float64) - 128.0
    coeffs = dctn(shifted, norm="ortho")

    table = quality_to_qtable(quality, get_qtable(base_table, block_size))
    quantized = np.round(coeffs / table).astype(np.int32)
    dequantized = quantized.astype(np.float64) * table
    recon = np.clip(idctn(dequantized, norm="ortho") + 128.0, 0.0, 255.0)

    err = recon - block.astype(np.float64)
    block_mse = float(np.mean(err ** 2))
    block_psnr = float("inf") if block_mse == 0 else 10.0 * np.log10(255.0 ** 2 / block_mse)
    n_nonzero = int(np.count_nonzero(quantized))

    return dict(
        block=block, shifted=shifted, coeffs=coeffs, qtable=table,
        quantized=quantized, dequantized=dequantized, recon=recon,
        grid_shape=grid_shape, position=(row, col),
        dc=float(coeffs[0, 0]), block_mse=block_mse, block_psnr=block_psnr,
        n_total=block_size * block_size, n_nonzero=n_nonzero,
    )


def block_pipeline_figure(result: dict) -> plt.Figure:
    """Orijinal blok / seviye-kaydırma / DCT katsayıları / kuantalanmış
    katsayılar / rekonstrüksiyon — pipeline'ın TÜM aşamaları (mega-spec
    "01 PIXELS → 02 LEVEL SHIFT → 03 DCT → 04 QUANTIZATION → 06 RECONSTRUCTION";
    zigzag için ayrı zigzag_figure() kullanılır, matris değil doğrusal akış)."""
    from src.viz.style import BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY

    # DCT Lab artık her Satır/Sütun/Quality değişiminde CANLI yeniden
    # hesaplanır (bkz. app.py update_dct_analysis) — önceki, yalnız düğmeyle
    # tetiklenen davranışa göre çok daha sık çağrılır. Matplotlib, döndürülen
    # Figure'ları kapatılana kadar bellekte tutar; her çağrı ÖNCE bir önceki
    # çağrılardan kalan (zaten Gradio'ya gönderilmiş, artık gereksiz)
    # figürleri kapatır — aksi halde uzun bir oturumda figür sızıntısı olur.
    plt.close("all")
    block = result["block"]
    n = block.shape[0]
    annotate = n <= 8

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.0))
    fig.patch.set_facecolor(BG_PANEL)
    panels = [
        (block, "01 Orijinal blok (piksel)", "gray", None),
        (result["shifted"], "02 Seviye kaydırma (piksel−128)", "RdBu_r", None),
        (np.sign(result["coeffs"]) * np.log1p(np.abs(result["coeffs"])),
         "03 DCT katsayıları (işaretli log)", "RdBu_r", result["coeffs"]),
        (np.sign(result["quantized"]) * np.log1p(np.abs(result["quantized"])),
         "04 Kuantalanmış katsayılar", "RdBu_r", result["quantized"]),
        (result["recon"], "06 Rekonstrüksiyon", "gray", None),
    ]
    for ax, (data, title, cmap, annot_src) in zip(axes, panels):
        ax.set_facecolor(BG_PANEL)
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=12, color=TEXT_PRIMARY)
        ax.set_xticks([]); ax.set_yticks([])
        if annotate:
            src = annot_src if annot_src is not None else data
            for i in range(n):
                for j in range(n):
                    val = src[i, j]
                    txt = f"{val:.0f}" if abs(val) < 1000 else f"{val:.1e}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=12.5,
                            color="black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY, labelsize=9)
    fig.suptitle(
        f"DC katsayısı (ortalama parlaklık bileşeni) = {result['dc']:.1f}   |   "
        f"sıfır-olmayan katsayı: {result['n_nonzero']}/{result['n_total']}   |   "
        f"blok PSNR = {result['block_psnr']:.2f} dB",
        fontsize=12, color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    return fig


def zigzag_figure(quantized: np.ndarray) -> plt.Figure:
    """JPEG zigzag tarama sırasındaki 64 (n²) katsayıyı DOĞRUSAL bir akış
    olarak gösterir — matris değil (mega-spec: "path overlay + altında
    linear coefficient stream"). Gerçek zigzag_indices (src/engines/entropy)
    kullanılır — src/engines/entropy.py'deki bit-tahmin motoruyla AYNI
    tarama sırası; matris değeri değil, encoder'ın gördüğü gerçek sıra.
    Sıfır katsayılar ikincil/soluk renkte, DC ilk eleman vurgulanır."""
    from src.viz.style import (ACCENT_CYAN, BG_ELEVATED, BG_PANEL, BORDER,
                               TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY)

    n = quantized.shape[0]
    order = zigzag_indices(n)
    stream = quantized.reshape(-1)[order]
    nonzero = stream != 0
    n_nonzero = int(nonzero.sum())

    fig, ax = plt.subplots(figsize=(18, 2.6))
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)
    x = np.arange(len(stream))
    colors = [ACCENT_CYAN if i == 0 else (TEXT_SECONDARY if v != 0 else BG_ELEVATED)
             for i, v in enumerate(stream)]
    ax.bar(x, stream, color=colors, width=0.7, zorder=2)
    ax.axhline(0, color=BORDER, linewidth=0.8, zorder=1)
    ax.set_xlim(-1, len(stream))
    ax.set_xticks(list(range(0, len(stream), max(1, len(stream) // 16))))
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_xlabel("zigzag sırası (0 = DC)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("katsayı", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(
        f"05 Zigzag tarama — sıfır olmayan: {n_nonzero} / {len(stream)}  "
        f"(camgöbeği = DC — kuantalanmış, soluk = sıfırlanmış)",
        fontsize=11, color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    return fig
