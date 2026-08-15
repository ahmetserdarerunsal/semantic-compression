# -*- coding: utf-8 -*-
"""Ana karşılaştırma ekranının hesaplama mantığı: aynı hedef bpp'de
JPEG/DCT (özel motor) vs Wavelet/DWT (özel motor) [+ isteğe bağlı gerçek
JPEG2000 kodek]. UI (app.py) yalnız bu fonksiyonları çağırır; sıkıştırma
mantığının tamamı burada ve src/engines/* altındadır (Part 20: transform/
sıkıştırma ayrımı, Part 19: tek metrik kaynağı).

Hedef bpp yalnızca DCT ve özel DWT motoru için bisection (match_bpp) ile
YAKLAŞIK tutturulur — bu bir arama/optimizasyon sonucu olduğundan gerçek
bpp hedeften küçük farkla sapabilir (mega-spec: "Target BPP ≠ Actual BPP").
Gerçek JPEG2000 (OpenJPEG) rate-control'ü ise doğrudan sıkıştırma oranı
kabul ettiğinden bisection'a gerek duymaz ve hedefe çok daha yakın çıkar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.engines import dct_engine, jpeg2000_engine, wavelet_engine
from src.engines.wavelet_engine import max_decomposition_level
from src.metrics.quality import calculate_metrics
from src.roi.bit_allocation import match_bpp

# Bisection arama aralıkları (mega-spec'te "binary search" olarak istenen
# rate-control mekanizması; bit_allocation.match_bpp'yi kullanır).
_DCT_QUALITY_RANGE = (1.0, 100.0)
_WAVELET_STEP_RANGE = (0.05, 512.0)


@dataclass
class EngineResult:
    label: str
    recon: np.ndarray
    metrics: dict
    param_label: str  # kullanıcıya gösterilecek "quality=35" / "step=12.4" vb.
    is_real_codec: bool  # True: gerçek bitstream kodek (JPEG2000/Pillow)
    note: str = ""


def run_dct(image: np.ndarray, target_bpp: float, block_size: int) -> EngineResult:
    recon, bpp, quality = match_bpp(
        lambda q: dct_engine.compress_image(image, q, None, 1.0, block_size),
        target_bpp, *_DCT_QUALITY_RANGE, True,
    )
    metrics = calculate_metrics(image, recon, bpp)
    return EngineResult(
        label="JPEG / DCT (özel eğitim motoru)",
        recon=recon, metrics=metrics,
        param_label=f"quality={quality:.0f}, blok={block_size}x{block_size}",
        is_real_codec=False,
        note="bpp, gerçek bir entropi kodlayıcı değil Shannon-alt-sınırı "
             "tahminidir (src/engines/entropy.py).",
    )


def run_wavelet(
    image: np.ndarray, target_bpp: float, wavelet: str, levels: int
) -> EngineResult:
    levels = min(levels, max_decomposition_level(image.shape[:2], wavelet))
    recon, bpp, step = match_bpp(
        lambda s: wavelet_engine.compress_image(image, s, wavelet, levels),
        target_bpp, *_WAVELET_STEP_RANGE, False,
    )
    metrics = calculate_metrics(image, recon, bpp)
    return EngineResult(
        label=f"Wavelet / DWT (özel eğitim motoru, {wavelet})",
        recon=recon, metrics=metrics,
        param_label=f"base_step={step:.2f}, seviye={levels}",
        is_real_codec=False,
        note="bpp, gerçek bir entropi kodlayıcı değil order-0 entropi "
             "tahminidir (src/engines/entropy.py). JPEG2000'in KENDİSİ "
             "değildir; JPEG2000'in ilkesiyle aynı dalgacık dönüşümünü "
             "kullanan özel bir eğitim motorudur.",
    )


def run_jpeg2000(image: np.ndarray, target_bpp: float) -> EngineResult | None:
    if not jpeg2000_engine.JPEG2000_AVAILABLE:
        return None
    recon, bpp, size_bytes = jpeg2000_engine.compress_at_target_bpp(image, target_bpp)
    metrics = calculate_metrics(image, recon, bpp)
    return EngineResult(
        label="JPEG2000 (gerçek kodek — Pillow/OpenJPEG)",
        recon=recon, metrics=metrics,
        param_label=f"gerçek sıkıştırılmış boyut={size_bytes} bayt",
        is_real_codec=True,
        note="Standarda uygun gerçek bir JPEG2000 bitstream'i; boyut "
             "tahmini değil, kodlayıcının ürettiği gerçek bayt sayısıdır.",
    )


def rd_sweep(
    image: np.ndarray, targets: list[float], wavelet: str, levels: int,
    block_size: int,
) -> dict[str, list[tuple[float, float]]]:
    """Verilen hedef bpp noktalarında DCT ve Wavelet için (gerçek_bpp, PSNR)
    listesi üretir — RD grafiği bu GERÇEK ölçümlerden çizilir."""
    dct_pts, wav_pts = [], []
    for t in targets:
        d = run_dct(image, t, block_size)
        w = run_wavelet(image, t, wavelet, levels)
        dct_pts.append((d.metrics["bpp"], d.metrics["psnr"]))
        wav_pts.append((w.metrics["bpp"], w.metrics["psnr"]))
    return {"JPEG / DCT": sorted(dct_pts), f"Wavelet / DWT ({wavelet})": sorted(wav_pts)}
