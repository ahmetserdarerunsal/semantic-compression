# -*- coding: utf-8 -*-
"""Çok seviyeli DWT tabanlı sıkıştırma motoru (JPEG2000 mantığı).

Kütüphaneden SADECE dönüşüm alınır (pywt.wavedec2 / waverec2).
Subband kuantalama, subband'ler arası bit allocation ve ROI'ye göre
katsayı-seviyesi adım haritası tamamen bu dosyada elle yazılmıştır.

Bit allocation mantığı (README'de de anlatılır): her subband'in birim
kuantalama hatasının piksel alanında yarattığı bozulma, o subband'in sentez
kazancıyla (synthesis gain) orantılıdır. Kazancı yüksek subband'e küçük adım
(çok bit), düşük olana büyük adım (az bit) verilir: step_b = base_step / gain_b.
Bu, JPEG2000 Annex E'deki enerji-ağırlık yaklaşımının kendisidir ve
"distortion'ı en çok düşüren subband'e daha çok bit" kuralına denk gelir.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pywt

from config import WAVELET_LEVELS
from src.engines.dct_engine import _rgb_to_ycbcr, _ycbcr_to_rgb
from src.engines.entropy import estimate_bits_subband


@lru_cache(maxsize=32)
def _synthesis_gains(wavelet: str, levels: int) -> tuple[tuple[float, ...], ...]:
    """Her subband için sentez kazancı: birim katsayının waverec2 sonrası L2 normu.

    Dönüş: (LL_kazancı,), sonra her seviye için (LH, HL, HH) kazançları.
    Bir kez hesaplanıp önbelleklenir (görüntüden bağımsızdır).
    """
    size = 2 ** (levels + 4)
    zero = np.zeros((size, size))
    coeffs = pywt.wavedec2(zero, wavelet, level=levels)
    gains: list[tuple[float, ...]] = []

    # LL bandı
    a = coeffs[0]
    a[a.shape[0] // 2, a.shape[1] // 2] = 1.0
    gains.append((float(np.linalg.norm(pywt.waverec2(coeffs, wavelet))),))
    a[a.shape[0] // 2, a.shape[1] // 2] = 0.0

    # Detay bantları (kaba seviyeden inceye)
    for li in range(1, levels + 1):
        level_gains = []
        for bi in range(3):
            d = coeffs[li][bi]
            d[d.shape[0] // 2, d.shape[1] // 2] = 1.0
            level_gains.append(float(np.linalg.norm(pywt.waverec2(coeffs, wavelet))))
            d[d.shape[0] // 2, d.shape[1] // 2] = 0.0
        gains.append(tuple(level_gains))
    return tuple(gains)


def _resize_mask_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Piksel maskesini subband çözünürlüğüne indirger (herhangi bir örtüşme
    varsa katsayı önemli sayılır — ROI sınırında ringing'i önlemek için)."""
    h, w = mask.shape
    th, tw = shape
    ys = (np.arange(th + 1) * h / th).astype(int)
    xs = (np.arange(tw + 1) * w / tw).astype(int)
    out = np.zeros((th, tw), dtype=bool)
    csum = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for i in range(th):
        y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
        block = (
            csum[y1, xs[1:].clip(min=1)] - csum[y0, xs[1:].clip(min=1)]
            - csum[y1, xs[:-1]] + csum[y0, xs[:-1]]
        )
        out[i] = block > 0
    return out


def _quantize_band(
    band: np.ndarray,
    step: float,
    mask: np.ndarray | None,
    bg_coarseness: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bir subband'i kuantalar; (kuantalanmış, adım haritası, bit) döner.

    Adım int32'ye sığmayacak kadar ince (aşırı küçük step / aşırı büyük
    katsayı) olabileceğinden kuantalanmış değerler int64'te tutulur; aksi
    halde `astype(int32)` sessizce taşar ve rekonstrüksiyonu bozardı.
    """
    if mask is None:
        steps = np.full(band.shape, step)
    else:
        m = _resize_mask_to(mask, band.shape)
        steps = np.where(m, step, step * bg_coarseness)
    steps = np.maximum(steps, 1e-8)  # sıfıra bölünmeyi önle
    q = np.round(band / steps).astype(np.int64)
    return q, steps, estimate_bits_subband(q)


def max_decomposition_level(shape: tuple[int, int], wavelet: str) -> int:
    """Görüntü boyutu ve dalgacık için pywt'nin kabul ettiği azami seviye.

    pywt.dwt_max_level filtre uzunluğuna ve en küçük kenara göre hesaplar;
    bunun üzerinde bir seviye istemek anlamsız (1x1'e yakın, bilgisiz)
    subband'ler üretir. UI bu değeri üst sınır olarak kullanmalı.
    """
    h, w = shape
    return max(1, pywt.dwt_max_level(min(h, w), pywt.Wavelet(wavelet).dec_len))


def decompose_for_viz(
    channel: np.ndarray, wavelet: str, levels: int
) -> list[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sıkıştırma/kuantalama YAPMADAN gerçek DWT katsayılarını döner.

    Dönüş pywt.wavedec2 ile aynı yapıdadır: [cA_n, (cH_n,cV_n,cD_n), ...,
    (cH_1,cV_1,cD_1)] — indeks 0 en kaba yaklaşım (LL), sonraki elemanlar
    kaba seviyeden inceye detay bantları. Sadece görselleştirme/inceleme
    içindir; DWT Explorer bu gerçek katsayılar üzerinden çalışır.
    """
    return pywt.wavedec2(channel.astype(np.float64) - 128.0, wavelet, level=levels)


def compress_channel(
    channel: np.ndarray,
    base_step: float,
    wavelet: str,
    levels: int = WAVELET_LEVELS,
    importance_mask: np.ndarray | None = None,
    bg_coarseness: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Tek kanalı DWT ile sıkıştırıp geri açar; (rekonstrüksiyon, bit) döner.

    `levels`, görüntü boyutu ve dalgacık filtre uzunluğu için geçerli azami
    seviyeyle sessizce sınırlanır (aksi halde pywt sınır-etkisi baskın,
    anlamsız subband'ler üretir — bkz. max_decomposition_level).
    """
    h, w = channel.shape
    levels = min(levels, max_decomposition_level((h, w), wavelet))
    coeffs = pywt.wavedec2(channel.astype(np.float64) - 128.0, wavelet, level=levels)
    gains = _synthesis_gains(wavelet, levels)

    total_bits = 0.0
    # LL bandı: ince adım, ROI uygulanmaz (katsayı sayısı ve bit payı çok
    # küçük; kaba LL tüm görüntüde renk/parlaklık bloklaşması yaratır)
    ll_step = base_step / gains[0][0]
    q, steps, bits = _quantize_band(coeffs[0], ll_step, None, 1.0)
    new_coeffs = [q.astype(np.float64) * steps]
    total_bits += bits

    for li in range(1, levels + 1):
        bands = []
        for bi in range(3):
            step_b = base_step / gains[li][bi]
            q, steps, bits = _quantize_band(
                coeffs[li][bi], step_b, importance_mask, bg_coarseness
            )
            bands.append(q.astype(np.float64) * steps)
            total_bits += bits
        new_coeffs.append(tuple(bands))

    recon = pywt.waverec2(new_coeffs, wavelet)[:h, :w] + 128.0
    return np.clip(recon, 0.0, 255.0), total_bits


def compress_image(
    image: np.ndarray,
    base_step: float,
    wavelet: str,
    levels: int = WAVELET_LEVELS,
    importance_mask: np.ndarray | None = None,
    bg_coarseness: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Gri veya RGB görüntüyü sıkıştırıp geri açar; (rekonstrüksiyon, bpp) döner.

    RGB girdiler YCbCr'ye çevrilir; kroma kanalları luma'dan 2 kat kaba
    kuantalanır (JPEG2000 pratiğindeki görsel ağırlıklandırmaya benzer).
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    if image.ndim == 2:
        recon, bits = compress_channel(
            image, base_step, wavelet, levels, importance_mask, bg_coarseness
        )
        return recon.astype(np.uint8), bits / (h * w)

    ycbcr = _rgb_to_ycbcr(image.astype(np.float64))
    out = np.empty_like(ycbcr)
    total_bits = 0.0
    for c in range(3):
        step_c = base_step if c == 0 else base_step * 2.0
        out[:, :, c], bits = compress_channel(
            ycbcr[:, :, c], step_c, wavelet, levels, importance_mask, bg_coarseness
        )
        total_bits += bits
    rgb = _ycbcr_to_rgb(out)
    return np.clip(rgb, 0, 255).astype(np.uint8), total_bits / (h * w)
