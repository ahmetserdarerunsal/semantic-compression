# -*- coding: utf-8 -*-
"""Gerçek JPEG kodek sarmalayıcısı (OpenCV -> libjpeg üzerinden).

ÖNEMLİ AYRIM (mega-spec "JPEG/DCT BPP VE RATE-DISTORTION DENETİMİ"):
src/engines/dct_engine.py bu projenin KENDİ yazdığı, eğitim amaçlı DCT
sıkıştırma motorudur — bit maliyeti gerçek bir Huffman kodlayıcı değil
Shannon-alt-sınırı TAHMİNİDİR (src/engines/entropy.py); JFIF/EXIF başlık,
Huffman tablosu aktarımı gibi gerçek bir bitstream'in taşıdığı ek yük
İÇERMEZ — bu yüzden aynı görsel kalitede her zaman gerçek bir JPEG
kodlayıcıdan biraz DAHA AZ bit rapor eder (teorik alt sınır olduğu için
matematiksel olarak beklenen bir durumdur, hata değildir).

Bu dosya ise ortamda gerçekten bulunan bir JPEG kodeği (OpenCV'nin
derlendiği libjpeg) üzerinden ÇALIŞAN, gerçek bir JPEG kodlayıcı/
çözücüdür — sıkıştırılmış boyut GERÇEK bayt sayısıdır (Huffman tabloları +
JFIF başlığı + gerçek entropi-kodlanmış veri dahil), tahmini DEĞİLDİR.
src/engines/jpeg2000_engine.py ile AYNI mimari desen: gerçek kodek, ayrı
dosya, `_AVAILABLE` bayrağı, sahte/simüle sonuç ASLA üretilmez.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.roi.bit_allocation import match_bpp

#: cv2'nin JPEG desteği (libjpeg) OpenCV'nin neredeyse TÜM standart
#: derlemelerinde (pip 'opencv-python' dahil) bulunur; yine de eldeki
#: kurulumda gerçekten çalıştığını (sahte bir "mevcut" bayrağı değil)
#: doğrulamak için gerçek bir kodlama denenir.
try:
    _ok, _buf = cv2.imencode(".jpg", np.zeros((8, 8), dtype=np.uint8),
                             [cv2.IMWRITE_JPEG_QUALITY, 50])
    REAL_JPEG_AVAILABLE: bool = bool(_ok) and _buf.size > 0
except Exception:  # noqa: BLE001 — yalnız ortam-kontrolü, kapsamlı olmalı
    REAL_JPEG_AVAILABLE = False

_QUALITY_RANGE = (1.0, 100.0)


def encode(image: np.ndarray, quality: float) -> tuple[np.ndarray, int]:
    """Gerçek libjpeg ile kodlar VE geri çözer.

    Dönüş: (rekonstrüksiyon uint8, GERÇEK sıkıştırılmış bayt sayısı —
    cv2.imencode'un ürettiği tam JPEG dosyası boyutu, tahmini değil).
    """
    if not REAL_JPEG_AVAILABLE:
        raise RuntimeError("Bu ortamda gerçek bir JPEG kodeği (libjpeg) bulunamadı.")
    q = int(np.clip(round(quality), 1, 100))
    image = np.asarray(image)
    grayscale = image.ndim == 2
    src = image if grayscale else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ok, buf = cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("Gerçek JPEG kodlama başarısız oldu.")
    size_bytes = int(buf.size)  # GERÇEK dosya boyutu: header + Huffman tablosu + entropi-kodlu veri

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(buf, flag)
    if not grayscale:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return decoded.astype(np.uint8), size_bytes


def compress_at_target_bpp(image: np.ndarray, target_bpp: float) -> tuple[np.ndarray, float, int, float]:
    """Hedef bpp'yi tutturan JPEG quality'sini bisection ile bulur ve
    GERÇEK kodlanmış boyutla kodlar (mega-spec Part 3: "target_bpp ≈
    actual_bpp" — actual_bpp burada GERÇEK dosya boyutundan gelir, bir
    entropi tahmininden DEĞİL).

    Dönüş: (rekonstrüksiyon uint8, gerçek bpp, gerçek bayt sayısı, kullanılan quality).
    """
    image = np.asarray(image)
    h, w = image.shape[:2]

    def _encode_for_search(q: float) -> tuple[np.ndarray, float]:
        recon, size_bytes = encode(image, q)
        bpp = size_bytes * 8.0 / (h * w)
        return recon, bpp

    recon, bpp, quality = match_bpp(_encode_for_search, target_bpp, *_QUALITY_RANGE, True)
    _, size_bytes = encode(image, quality)
    return recon, bpp, size_bytes, quality
