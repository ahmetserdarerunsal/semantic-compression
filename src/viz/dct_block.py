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
    """Orijinal blok / DCT katsayıları / kuantalanmış katsayılar / rekonstrüksiyon."""
    block = result["block"]
    n = block.shape[0]
    annotate = n <= 8

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    panels = [
        (block, "Orijinal blok (piksel)", "gray", None),
        (np.sign(result["coeffs"]) * np.log1p(np.abs(result["coeffs"])),
         "DCT katsayıları (işaretli log ölçek)", "RdBu_r", result["coeffs"]),
        (np.sign(result["quantized"]) * np.log1p(np.abs(result["quantized"])),
         "Kuantalanmış katsayılar", "RdBu_r", result["quantized"]),
        (result["recon"], "Rekonstrüksiyon", "gray", None),
    ]
    for ax, (data, title, cmap, annot_src) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if annotate:
            src = annot_src if annot_src is not None else data
            for i in range(n):
                for j in range(n):
                    val = src[i, j]
                    txt = f"{val:.0f}" if abs(val) < 1000 else f"{val:.1e}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                            color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"DC (ortalama parlaklık) = {result['dc']:.1f}   |   "
        f"sıfır-olmayan katsayı: {result['n_nonzero']}/{result['n_total']}   |   "
        f"blok PSNR = {result['block_psnr']:.2f} dB",
        fontsize=9,
    )
    fig.tight_layout()
    return fig
