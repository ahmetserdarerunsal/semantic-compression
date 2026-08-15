# -*- coding: utf-8 -*-
"""Gerçek JPEG2000 kodek sarmalayıcısı (Pillow -> OpenJPEG üzerinden).

ÖNEMLİ AYRIM (mega-spec Part 16): src/engines/wavelet_engine.py bu projenin
KENDİ yazdığı, eğitim amaçlı DWT sıkıştırma motorudur — standarda uygun bir
JPEG2000 kodlayıcı DEĞİLDİR (gerçek bit akışı, EBCOT/bit-plane kodlama,
paketleme yoktur). Bu dosya ise ortamda gerçekten bulunan bir JPEG2000
kodeği (Pillow'un derlendiği OpenJPEG kütüphanesi) üzerinden ÇALIŞAN,
standarda uygun gerçek bir JPEG2000 kodlayıcı/çözücüdür: sıkıştırılmış boyut
gerçek bayt sayısıdır (tahmini değildir), Pillow ile kodlanıp çözülür.

Kullanım: JPEG2000_AVAILABLE False ise UI bu seçeneği gizlemeli/devre dışı
bırakmalı ve yalnız özel DWT motorunu göstermelidir — asla sahte bir
JPEG2000 sonucu üretilmemelidir.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, features

JPEG2000_AVAILABLE: bool = features.check("jpg_2000")


def compress_image(image: np.ndarray, compression_ratio: float) -> tuple[np.ndarray, float, int]:
    """Gerçek JPEG2000 ile sıkıştırıp geri açar.

    compression_ratio: hedef ham/sıkıştırılmış oranı (OpenJPEG'in
    `quality_layers` parametresi — bu değer bir tahmin/bisection hedefi
    DEĞİL, kodekin doğrudan uyguladığı gerçek bir sıkıştırma oranı
    hedefidir; üretilen bayt sayısı bu orana çok yakın çıkar).

    Dönüş: (rekonstrüksiyon uint8, gerçek bpp, gerçek sıkıştırılmış bayt sayısı).
    """
    if not JPEG2000_AVAILABLE:
        raise RuntimeError(
            "Bu ortamda gerçek bir JPEG2000 kodeği (OpenJPEG) bulunamadı; "
            "Pillow JPEG2000 desteğiyle kurulmalı."
        )
    image = np.asarray(image)
    h, w = image.shape[:2]
    mode = "L" if image.ndim == 2 else "RGB"
    im = Image.fromarray(image, mode=mode)

    buf = io.BytesIO()
    im.save(
        buf,
        format="JPEG2000",
        quality_mode="rates",
        quality_layers=[max(compression_ratio, 1.0)],
        irreversible=True,  # gerçek (lossy) DWT yolu; reversible=5/3 tam tersi kayıpsızdır
    )
    size_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert(mode))

    total_bits = size_bytes * 8.0
    bpp = total_bits / (h * w)
    return decoded.astype(np.uint8), bpp, size_bytes


def compress_at_target_bpp(image: np.ndarray, target_bpp: float) -> tuple[np.ndarray, float, int]:
    """target_bpp'ye karşılık gelen sıkıştırma oranını hesaplayıp kodlar.

    OpenJPEG rate-control'ü doğrudan bpp değil compression-ratio kabul
    eder; oran = ham_bpp / hedef_bpp dönüşümü buradadır (ham = kanal*8).
    """
    image = np.asarray(image)
    channels = 1 if image.ndim == 2 else image.shape[2]
    raw_bpp = channels * 8.0
    ratio = max(raw_bpp / max(target_bpp, 1e-6), 1.0)
    return compress_image(image, ratio)
