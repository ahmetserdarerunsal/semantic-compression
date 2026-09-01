# -*- coding: utf-8 -*-
"""Ana karşılaştırma ekranının hesaplama mantığı: aynı hedef bpp'de
GERÇEK JPEG vs GERÇEK JPEG2000 (mega-spec "JPEG vs JPEG2000 GERÇEK CODEC
KARŞILAŞTIRMASI") — ikisi de gerçek, çalıştırılan bir kodlayıcı/çözücü;
hiçbiri entropi tahmini DEĞİLDİR. UI (app.py) yalnız bu fonksiyonları
çağırır; sıkıştırma mantığının tamamı burada ve src/engines/* altındadır.

'JPEG / DCT' paneli run_jpeg_primary() üzerinden GERÇEK libjpeg kullanır
(block_size==8 ve ortamda kodek varsa) — özel entropi-tahmini motoru
(run_dct) yalnız standart-dışı blok boyutlarında (4/16) veya kodek yoksa
devreye girer, bu durumda panel ENTROPİ TAHMİNİ olarak işaretlenir.
'JPEG2000 / DWT' paneli run_jpeg2000() üzerinden GERÇEK OpenJPEG kullanır
(is_real_codec=True HER ZAMAN) — özel entropi-tahmini DWT motoru
(run_wavelet, Shannon order-0 tahmini) artık ana karşılaştırmada
KULLANILMAZ; yalnız DWT Lab ve Semantik ROI'de kalır (bkz. app.py, bu
dosyadaki run_wavelet fonksiyonunun kendisi SİLİNMEDİ).

Hedef bpp DCT motorunda bisection (match_bpp) ile YAKLAŞIK tutturulur —
bu bir arama/optimizasyon sonucu olduğundan gerçek bpp hedeften küçük
farkla sapabilir (mega-spec: "Target BPP ≠ Actual BPP"). Gerçek JPEG2000
(OpenJPEG) rate-control'ü ise doğrudan sıkıştırma oranı kabul ettiğinden
bisection'a gerek duymaz ve hedefe çok daha yakın çıkar (ölçüldü: %0.0-0.6
sapma). PSNR/SSIM hiçbir parametre aramasında SEÇİM KRİTERİ DEĞİLDİR —
yalnız |actual_bpp - target_bpp| minimize edilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.engines import dct_engine, jpeg2000_engine, real_jpeg_engine, wavelet_engine
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
    # Yapısal parametreler (interaktif RD tooltip'i için) — param_label'ın
    # metin haline getirdiği AYNI değerler, ayrıştırmaya gerek kalmadan
    # programatik erişim için (mega-spec: "Do NOT fabricate... use actually
    # measured" — bu alanlar gerçek bisection sonucundan gelir).
    extra: dict = field(default_factory=dict)


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
        extra={"quality": round(quality, 1), "block_size": block_size},
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
        extra={"wavelet": wavelet, "level": levels, "base_step": round(step, 2)},
    )


def run_jpeg_primary(image: np.ndarray, target_bpp: float, block_size: int) -> EngineResult:
    """Karşılaştırma ekranının ANA 'JPEG / DCT' paneli için tek giriş
    noktası (mega-spec 'BPP/PSNR/MSE/SSIM GÜVENİLİRLİK DENETİMİ' Part 2:
    "Ana JPEG karşılaştırması entropy estimate OLMASIN, gerçek bir JPEG
    encoder kullan"). block_size==8 (standart JPEG) VE bu ortamda gerçek
    bir libjpeg varsa GERÇEK JPEG bitstream'i (run_real_jpeg) kullanılır —
    is_real_codec=True, bpp GERÇEK kodlanmış bayt sayısından gelir.

    block_size 4/16 (config.py'de kasıtlı olarak "eğitim amaçlı deney"
    diye işaretli — standart JPEG'in DIŞINDA, gerçek bir kodekte karşılığı
    YOK) VEYA bu ortamda libjpeg yoksa özel entropi-tahmini motoruna
    (run_dct) düşer — bu durumda panel ENTROPİ TAHMİNİ rozetiyle
    (is_real_codec=False, EngineResult.is_real_codec üzerinden) dürüstçe
    işaretlenir, gerçekmiş gibi GÖSTERİLMEZ."""
    if block_size == 8 and real_jpeg_engine.REAL_JPEG_AVAILABLE:
        res = run_real_jpeg(image, target_bpp)
        if res is not None:
            return res
    return run_dct(image, target_bpp, block_size)


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
        extra={"size_bytes": size_bytes},
    )


def run_real_jpeg(image: np.ndarray, target_bpp: float) -> EngineResult | None:
    """Gerçek libjpeg ile kodlar (mega-spec "JPEG/DCT BPP VE RATE-
    DISTORTION DENETİMİ" Part 2/3) — src/engines/real_jpeg_engine.py,
    dct_engine'in Shannon-entropi TAHMİNİNİN aksine GERÇEK bir JPEG
    bitstream'i üretir (JFIF başlığı + Huffman tabloları + gerçek entropi-
    kodlanmış veri dahil gerçek bayt sayısı). `actual_bpp`, gerçek dosya
    boyutundan hesaplanır — bir entropi tahmininden DEĞİL; bu yüzden
    dct_engine'in özel motoruna göre akademik referans eğrileriyle DAHA
    doğrudan karşılaştırılabilir bir çapraz-doğrulamadır (JPEG2000
    seçeneğiyle AYNI mimari desen)."""
    if not real_jpeg_engine.REAL_JPEG_AVAILABLE:
        return None
    recon, bpp, size_bytes, quality = real_jpeg_engine.compress_at_target_bpp(image, target_bpp)
    metrics = calculate_metrics(image, recon, bpp)
    return EngineResult(
        label="JPEG (gerçek kodek — OpenCV/libjpeg)",
        recon=recon, metrics=metrics,
        param_label=f"quality={quality:.0f}, gerçek sıkıştırılmış boyut={size_bytes} bayt",
        is_real_codec=True,
        note="Standarda uygun gerçek bir JPEG bitstream'i (libjpeg); boyut "
             "tahmini değil, kodlayıcının ürettiği gerçek bayt sayısıdır. "
             "dct_engine'deki özel eğitim motorundan (Shannon-entropi "
             "tahmini) BAĞIMSIZ, ayrı bir ölçümdür.",
        extra={"quality": round(quality, 1), "size_bytes": size_bytes},
    )


def _point_dict(res: EngineResult, target_bpp: float) -> dict:
    """Bir RD noktası için grafikte GERÇEKTEN ölçülmüş değerleri taşır —
    interaktif tooltip'ler bu sözlükten okur, hiçbir alan uydurulmaz."""
    m = res.metrics
    return {
        "bpp": m["bpp"], "psnr": m["psnr"], "ssim": m["ssim"],
        "compression_ratio": m["compression_ratio"],
        "param_label": res.param_label, "target_bpp": target_bpp,
        "extra": res.extra,
    }


def rd_sweep(image: np.ndarray, targets: list[float], block_size: int) -> dict[str, list[dict]]:
    """Verilen hedef bpp noktalarında JPEG ve JPEG2000 için GERÇEK ölçüm
    noktaları üretir — her nokta yalnız (bpp, psnr) değil, interaktif RD
    grafiğinin hover tooltip'i için gereken tüm GERÇEK ölçülmüş metrikleri
    (ssim, sıkıştırma oranı, motora özgü parametre) taşır. İkisi de GERÇEK
    bitstream kodek (mega-spec 'JPEG vs JPEG2000 GERÇEK CODEC KARŞILAŞTIRMASI')
    — özel entropi-tahmini wavelet motoru (run_wavelet) burada KULLANILMAZ."""
    dct_pts, wav_pts = [], []
    for t in targets:
        d = run_jpeg_primary(image, t, block_size)
        w = run_jpeg2000(image, t)
        dct_pts.append(_point_dict(d, t))
        if w is not None:
            wav_pts.append(_point_dict(w, t))
    dct_pts.sort(key=lambda p: p["bpp"])
    wav_pts.sort(key=lambda p: p["bpp"])
    return {"JPEG / DCT": dct_pts, "JPEG2000 / DWT": wav_pts}
