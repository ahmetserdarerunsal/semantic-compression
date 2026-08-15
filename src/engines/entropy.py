# -*- coding: utf-8 -*-
"""Bit maliyeti tahmini (gerçek Huffman kodlayıcı yerine entropi modeli).

Neden tahmin? Projenin odağı bit YÖNLENDİRME mantığı; kanonik Huffman tablosu
yazmak mühendislik yükü ekler ama bilimsel katkı sağlamaz. Bunun yerine
Shannon alt sınırına dayalı, JPEG/JPEG2000'in sembol yapısına sadık bir
tahminci kullanıyoruz (yöntem README'de anlatılır):

- DCT: DC katsayıları DPCM farkları olarak, AC katsayıları zigzag sırada
  (sıfır-koşu, boyut-kategorisi) sembolleri olarak modellenir (JPEG ile aynı).
  bit = H(semboller) * sembol_sayısı + toplam genlik biti.
- Wavelet: her subband için kuantalanmış katsayıların order-0 entropisi.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import DCT_BLOCK_SIZE


def _shannon_entropy_bits(symbols: np.ndarray) -> float:
    """Sembol dizisinin toplam bit maliyeti: H(p) * N (order-0 entropi)."""
    if symbols.size == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / symbols.size
    return float(-(p * np.log2(p)).sum() * symbols.size)


def _size_category(values: np.ndarray) -> np.ndarray:
    """JPEG 'size category': değeri temsil etmek için gereken bit sayısı."""
    return np.ceil(np.log2(np.abs(values).astype(np.float64) + 1.0)).astype(np.int64)


@lru_cache(maxsize=8)
def zigzag_indices(n: int = DCT_BLOCK_SIZE) -> np.ndarray:
    """n x n blok için zigzag tarama sırasının düz indeksleri."""
    idx = np.arange(n * n).reshape(n, n)
    order: list[int] = []
    for s in range(2 * n - 1):
        diag = [(i, s - i) for i in range(max(0, s - n + 1), min(n, s + 1))]
        if s % 2 == 0:
            diag.reverse()  # çift diyagonaller yukarı doğru taranır
        order.extend(idx[i, j] for i, j in diag)
    return np.array(order)


def estimate_bits_dct(quantized: np.ndarray) -> float:
    """Kuantalanmış blok DCT katsayıları için toplam bit tahmini.

    quantized: (nbh, nbw, block, block) int dizisi. Blok boyutu dizinin
    kendisinden çıkarılır (config'teki varsayılan 8'e bağımlı değildir),
    böylece DCT Explorer'daki 4x4/16x16 seçenekleri de doğru zigzag sırayla
    modellenir.
    """
    block = quantized.shape[-1]
    zigzag = zigzag_indices(block)
    flat = quantized.reshape(-1, block * block)[:, zigzag]  # (N, block^2) zigzag
    n_blocks = flat.shape[0]

    # --- DC: bloklar arası DPCM farkları ---
    dc = flat[:, 0].astype(np.int64)
    dc_diff = np.diff(dc, prepend=0)
    dc_cat = _size_category(dc_diff)
    dc_bits = _shannon_entropy_bits(dc_cat) + float(dc_cat.sum())

    # --- AC: (sıfır-koşu, kategori) sembolleri + EOB ---
    ac = flat[:, 1:]
    rows, cols = np.nonzero(ac)
    if rows.size == 0:
        # Tüm AC'ler sıfır: blok başına yalnız EOB sembolü
        return dc_bits + _shannon_entropy_bits(np.zeros(n_blocks, dtype=np.int64))

    values = ac[rows, cols].astype(np.int64)
    cats = _size_category(values)

    # Aynı bloktaki ardışık sıfır-olmayanlar arası koşu; bloğun ilki için
    # koşu = sütun indeksi (baştan o ana kadarki sıfır sayısı)
    first_of_block = np.empty(rows.size, dtype=bool)
    first_of_block[0] = True
    first_of_block[1:] = rows[1:] != rows[:-1]
    runs = np.empty(rows.size, dtype=np.int64)
    runs[first_of_block] = cols[first_of_block]
    runs[~first_of_block] = (cols[1:] - cols[:-1] - 1)[~first_of_block[1:]]

    # Sembol = koşu ve kategorinin birleşimi (JPEG'in RRRRSSSS baytı gibi)
    ac_symbols = runs * 16 + cats
    # Her blok için bir EOB sembolü (ayırt edici negatif değer)
    eob = np.full(n_blocks, -1, dtype=np.int64)
    stream = np.concatenate([ac_symbols, eob])

    ac_bits = _shannon_entropy_bits(stream) + float(cats.sum())
    return dc_bits + ac_bits


def estimate_bits_subband(quantized: np.ndarray) -> float:
    """Bir wavelet subband'i için order-0 entropi tabanlı bit tahmini."""
    return _shannon_entropy_bits(quantized.astype(np.int64).ravel())
