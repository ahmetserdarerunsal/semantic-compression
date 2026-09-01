# -*- coding: utf-8 -*-
"""ROI bit yönlendirme yardımcıları ve bit bütçesi eşleme.

İki mekanizma:
1. mask_to_block_importance: piksel maskesini 8x8 DCT blok ızgarasına indirger
   (wavelet motoru maskeyi subband seviyesinde kendisi ölçekler).
2. match_bpp: verilen hedef bpp'ye ulaşmak için motorun quality/step
   parametresini bisection ile ayarlar. Adil kıyaslamanın temeli budur:
   baseline ve semantic sonuçlar AYNI gerçek bpp'de karşılaştırılır.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from config import (
    BPP_MATCH_MAX_ITER,
    BPP_MATCH_TOLERANCE,
    DCT_BLOCK_SIZE,
    ROI_BLOCK_OVERLAP_THRESHOLD,
)

# Bir encode çağrısı: parametre -> (rekonstrüksiyon, gerçek bpp)
EncodeFn = Callable[[float], tuple[np.ndarray, float]]


def rectangle_mask(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Manuel ROI için dikdörtgen bool piksel maskesi.

    YOLO tabanlı get_importance_mask ile AYNI arayüze (bool piksel maskesi)
    uyar — bu sayede manuel ROI, mask_to_block_importance / wavelet
    importance_mask gibi MEVCUT sıkıştırma yolunu değiştirmeden kullanır
    (mega-spec: "Manuel ROI gerçek compression pipeline'ına bağlanmalı").
    """
    h, w = shape
    x0, x1 = sorted((max(0, min(x0, w)), max(0, min(x1, w))))
    y0, y1 = sorted((max(0, min(y0, h)), max(0, min(y1, h))))
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def mask_to_block_importance(
    mask: np.ndarray,
    block: int = DCT_BLOCK_SIZE,
    threshold: float = ROI_BLOCK_OVERLAP_THRESHOLD,
) -> np.ndarray:
    """Piksel maskesini (H/8, W/8) blok önem ızgarasına çevirir.

    Bir blok, pikselinin en az `threshold` oranı maskeyle örtüşüyorsa
    önemli sayılır.
    """
    h, w = mask.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    padded = np.pad(mask.astype(np.float64), ((0, ph), (0, pw)))
    nh, nw = padded.shape[0] // block, padded.shape[1] // block
    ratios = padded.reshape(nh, block, nw, block).mean(axis=(1, 3))
    return ratios >= threshold


def match_bpp(
    encode: EncodeFn,
    target_bpp: float,
    param_lo: float,
    param_hi: float,
    bpp_increases_with_param: bool,
    tolerance: float = BPP_MATCH_TOLERANCE,
    max_iter: int = BPP_MATCH_MAX_ITER,
) -> tuple[np.ndarray, float, float]:
    """Hedef bpp'yi tutturan parametreyi bisection ile bulur.

    encode: param -> (rekonstrüksiyon, bpp). bpp'nin parametrede monoton
    olduğu varsayılır; yönü bpp_increases_with_param belirtir.
    Dönüş: (rekonstrüksiyon, gerçek bpp, kullanılan parametre) — hedefe en
    yakın bulunan nokta.
    """
    lo, hi = float(param_lo), float(param_hi)
    best: tuple[np.ndarray, float, float] | None = None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        recon, bpp = encode(mid)
        if best is None or abs(bpp - target_bpp) < abs(best[1] - target_bpp):
            best = (recon, bpp, mid)
        if abs(bpp - target_bpp) / target_bpp < tolerance:
            break
        too_big = bpp > target_bpp
        if too_big == bpp_increases_with_param:
            hi = mid
        else:
            lo = mid
    assert best is not None
    return best
