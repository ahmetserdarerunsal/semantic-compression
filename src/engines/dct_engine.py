# -*- coding: utf-8 -*-
"""8x8 blok DCT tabanlı sıkıştırma motoru (JPEG mantığı).

Kütüphaneden SADECE matematiksel dönüşüm alınır (scipy.fft.dctn/idctn).
Kuantalama, quality ölçekleme, zigzag tarama ve bit maliyeti tahmini
tamamen bu dosyada elle yazılmıştır.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from scipy.fft import dctn, idctn

from config import DCT_BLOCK_SIZE, JPEG_LUMA_QTABLE, JPEG_CHROMA_QTABLE
from src.engines.entropy import estimate_bits_dct


def get_qtable(base_table: np.ndarray, block_size: int) -> np.ndarray:
    """8x8 standart JPEG kuantalama tablosunu block_size x block_size'a ölçekler.

    JPEG standardı yalnız 8x8 tanımlar; 4x4/16x16 gibi boyutlar bu projede
    eğitim amaçlı bir DCT-boyutu deneyidir, standarda uygun değildir. Düşük
    ->yüksek frekans artışını korumak için 8x8 tablo bilinear enterpolasyonla
    yeniden örneklenir (gerçek JPEG tablosu DEĞİLDİR; UI bunu ayrıca belirtir).
    """
    if block_size == 8:
        return base_table
    factor = block_size / 8.0
    resized = zoom(base_table, factor, order=1)
    return np.clip(resized, 1.0, 255.0)


def quality_to_qtable(quality: float, base_table: np.ndarray) -> np.ndarray:
    """JPEG standardındaki (IJG) quality -> kuantalama matrisi ölçekleme formülü.

    quality 1-100 arası; 50 = standart tablo, >50 daha ince, <50 daha kaba.
    """
    quality = float(np.clip(quality, 1.0, 100.0))
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality
    table = np.floor((base_table * scale + 50.0) / 100.0)
    return np.clip(table, 1.0, 255.0)


def _pad_to_block(channel: np.ndarray, block: int) -> np.ndarray:
    """Kanalı blok boyutunun katına kenar-tekrarı ile doldurur."""
    h, w = channel.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    return np.pad(channel, ((0, ph), (0, pw)), mode="edge")


def _blockwise_dct(channel: np.ndarray, block: int) -> np.ndarray:
    """Kanalı (H/8, W/8, 8, 8) blok dizisine çevirip her bloğa 2D DCT uygular."""
    h, w = channel.shape
    blocks = channel.reshape(h // block, block, w // block, block).swapaxes(1, 2)
    return dctn(blocks, axes=(2, 3), norm="ortho")


def _blockwise_idct(coeffs: np.ndarray, block: int) -> np.ndarray:
    """Blok dizisindeki katsayılara ters DCT uygulayıp kanalı geri birleştirir."""
    rec = idctn(coeffs, axes=(2, 3), norm="ortho")
    nbh, nbw = rec.shape[:2]
    return rec.swapaxes(1, 2).reshape(nbh * block, nbw * block)


def compress_channel(
    channel: np.ndarray,
    quality: float,
    base_table: np.ndarray,
    block_importance: np.ndarray | None = None,
    bg_coarseness: float = 1.0,
    block_size: int = DCT_BLOCK_SIZE,
) -> tuple[np.ndarray, float]:
    """Tek kanalı sıkıştırıp geri açar; (rekonstrüksiyon, tahmini bit sayısı) döner.

    block_importance: (H/block, W/block) bool dizisi. True bloklar `quality`
    ile (ince), False bloklar bg_coarseness kat kaba bir tabloyla kuantalanır.
    None ise tüm bloklar uniform `quality` ile kuantalanır (baseline).
    block_size: standart JPEG 8'dir; 4/16 gibi başka değerler DCT Explorer'daki
    eğitim amaçlı deney içindir (bkz. get_qtable).
    """
    block = block_size
    h, w = channel.shape
    padded = _pad_to_block(channel.astype(np.float64) - 128.0, block)
    coeffs = _blockwise_dct(padded, block)
    nbh, nbw = coeffs.shape[:2]

    sized_table = get_qtable(base_table, block)
    fine_table = quality_to_qtable(quality, sized_table)
    if block_importance is None:
        qtables = np.broadcast_to(fine_table, (nbh, nbw, block, block))
    else:
        coarse_table = np.clip(fine_table * bg_coarseness, 1.0, 255.0)
        imp = block_importance[:nbh, :nbw].astype(bool)
        qtables = np.where(imp[:, :, None, None], fine_table, coarse_table)

    # Kuantalama: katsayıyı tabloya böl, en yakın tam sayıya yuvarla
    quantized = np.round(coeffs / qtables).astype(np.int32)

    total_bits = estimate_bits_dct(quantized)

    # Ters yol: dequantize -> IDCT -> kırp
    recon = _blockwise_idct(quantized.astype(np.float64) * qtables, block)
    recon = np.clip(recon[:h, :w] + 128.0, 0.0, 255.0)
    return recon, total_bits


def compress_image(
    image: np.ndarray,
    quality: float,
    block_importance: np.ndarray | None = None,
    bg_coarseness: float = 1.0,
    block_size: int = DCT_BLOCK_SIZE,
) -> tuple[np.ndarray, float]:
    """Gri veya RGB görüntüyü sıkıştırıp geri açar; (rekonstrüksiyon, bpp) döner.

    RGB girdiler YCbCr'ye çevrilir: luma için luminance tablosu, kroma için
    chroma tablosu kullanılır (JPEG ile aynı felsefe; kroma alt örnekleme yok).
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    luma_t = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    chroma_t = np.asarray(JPEG_CHROMA_QTABLE, dtype=np.float64)

    if image.ndim == 2:
        recon, bits = compress_channel(
            image, quality, luma_t, block_importance, bg_coarseness, block_size
        )
        return recon.astype(np.uint8), bits / (h * w)

    ycbcr = _rgb_to_ycbcr(image.astype(np.float64))
    out = np.empty_like(ycbcr)
    total_bits = 0.0
    for c in range(3):
        table = luma_t if c == 0 else chroma_t
        out[:, :, c], bits = compress_channel(
            ycbcr[:, :, c], quality, table, block_importance, bg_coarseness, block_size
        )
        total_bits += bits
    rgb = _ycbcr_to_rgb(out)
    return np.clip(rgb, 0, 255).astype(np.uint8), total_bits / (h * w)


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 tam-aralık RGB -> YCbCr dönüşümü (float, 0-255 ölçeği)."""
    m = np.array(
        [[0.299, 0.587, 0.114],
         [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]]
    )
    ycbcr = rgb @ m.T
    ycbcr[:, :, 1:] += 128.0
    return ycbcr


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """YCbCr -> RGB ters dönüşümü."""
    x = ycbcr.copy()
    x[:, :, 1:] -= 128.0
    m = np.array(
        [[1.0, 0.0, 1.402],
         [1.0, -0.344136, -0.714136],
         [1.0, 1.772, 0.0]]
    )
    return x @ m.T
