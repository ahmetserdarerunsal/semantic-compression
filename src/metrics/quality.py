# -*- coding: utf-8 -*-
"""Kalite metrikleri: PSNR ve SSIM — global, foreground (maske içi) ve
background (maske dışı) varyantlarıyla.

Maskeli SSIM, tam SSIM haritası çıkarılıp yalnız ilgili bölge üzerinde
ortalanarak hesaplanır (bölgeyi kırpıp ayrı SSIM hesaplamak pencere
sınır etkileri yüzünden yanlı olurdu).

PSNR yorumu (UI'da da gösterilir): PSNR, orijinale sayısal (piksel-bazlı)
yakınlığı ölçer; yüksek PSNR genelde daha az bozulma demektir ama mükemmel
bir algısal kalite metriği DEĞİLDİR — aynı PSNR farklı görüntülerde farklı
algısal etkiye sahip olabilir. "40 dB = mükemmel" gibi mutlak bir eşik yoktur.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(reference: np.ndarray, test: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Ortalama kare hata (MSE). mask verilirse yalnız maskedeki pikseller."""
    ref = reference.astype(np.float64)
    tst = test.astype(np.float64)
    err = (ref - tst) ** 2
    if mask is not None:
        err = err[mask.astype(bool)]
    return float(err.mean()) if err.size else float("nan")


def psnr(reference: np.ndarray, test: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR (dB) = 10*log10(255^2 / MSE). mask verilirse yalnız maskedeki pikseller.

    MSE=0 (birebir aynı görüntü) durumunda tanım gereği +inf döner; bu bir
    hata değildir, bozulmasız/lossless rekonstrüksiyonu ifade eder.
    """
    m = mse(reference, test, mask)
    if m == 0:
        return float("inf")
    if not np.isfinite(m):
        return float("nan")
    return 10.0 * np.log10(255.0**2 / m)


def ssim(reference: np.ndarray, test: np.ndarray, mask: np.ndarray | None = None) -> float:
    """SSIM. mask verilirse SSIM haritasının maske üzerindeki ortalaması.

    GERÇEK BUG DÜZELTMESİ ("FINAL MATHEMATICAL VALIDATION & AUDIT"):
    skimage'ın `structural_similarity(..., full=True)` çağrısı
    `(mssim, S)` döner; `mssim` kütüphanenin KENDİSİNİN "SSIM skoru" olarak
    tanımladığı kanonik değerdir (kayan pencerenin tam oturduğu iç bölge
    üzerinden ortalanır), `S` ise KENAR pikselleri için farklı/daha az
    güvenilir değerler içeren TAM-BOYUTLU haritadır. Önceden burada
    (mask=None durumunda, yani DCT/DWT/Compare ekranlarındaki TÜM global
    SSIM değerleri) `smap.mean()` kullanılıyordu — bu, skimage'ın kendi
    "SSIM" tanımıyla AYNI DEĞİL: gerçek astronaut görüntüsünde ~1e-4
    mertebesinde (gösterilen 4 ondalık hassasiyetin tam sınırında) sistematik
    bir sapma yaratıyordu (bağımsız doğrulama script'iyle ölçüldü). Artık
    mask=None iken skimage'ın döndürdüğü kanonik `mssim` DOĞRUDAN kullanılır;
    yalnız maskeli (foreground/background) ortalama için `full=True`'nun
    uzamsal haritası GEREKLİDİR (bir alt-kümenin ortalaması kütüphanenin
    tek-sayı API'siyle alınamaz), o durumda hâlâ `smap` kullanılır."""
    multichannel = reference.ndim == 3
    mssim, smap = structural_similarity(
        reference,
        test,
        data_range=255,
        channel_axis=2 if multichannel else None,
        full=True,
    )
    if mask is None:
        return float(mssim)
    m = mask.astype(bool)
    selected = smap[m]
    # Boş bölge (maske bu tarafta hiç piksel içermiyor, ör. ROI tüm
    # görüntüyü kaplayınca arka plan boş kalır) -> tanımsız, NaN; numpy'nin
    # "Mean of empty slice" uyarısını burada AÇIKÇA ele alarak önleriz
    # (mse() ile aynı desen — bkz. yukarısı).
    return float(selected.mean()) if selected.size else float("nan")


def evaluate(
    reference: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Tüm metrik setini tek sözlükte döner.

    mask: bool piksel maskesi (True = önemli / foreground). None ise yalnız
    global metrikler hesaplanır.

    ÖNEMLİ (numerical-correctness audit ile bulunan gerçek bug): mask,
    görüntünün TAMAMINI kaplıyorsa (ör. Manuel ROI'de genişlik/yükseklik
    slider'ları maksimuma çekilmişse — 512x512 bir görüntüde erişilebilir,
    ezoterik bir durum değildir) arka plan bölgesi boştur; tersi durumda
    (maske tamamen boşsa) ön plan boştur. Bu durumda fg/bg anahtarları
    ÖNCEDEN sessizce sözlükten ÇIKARILIYORDU — çağıran taraf (app.py
    run_semantic_pipeline) bunları koşulsuz `mb["fg_psnr"]` ile okuduğundan
    bu KeyError ile ÇÖKÜYORDU. Artık fg/bg anahtarları maske verildiğinde
    HER ZAMAN mevcuttur; boş bölge durumunda değer NaN'dır (uydurulmuş bir
    sayı DEĞİL — "bu bölgede ölçülecek piksel yok" anlamına gelir) ve UI
    katmanı (src/viz/cards.fmt_psnr/fmt_ssim) NaN'ı doğru şekilde "N/A"
    olarak gösterir (+Inf/"∞ (kayıpsız)" ile KARIŞTIRILMAZ)."""
    out = {
        "psnr": psnr(reference, test),
        "ssim": ssim(reference, test),
    }
    if mask is not None:
        m = mask.astype(bool)
        out.update(
            fg_psnr=psnr(reference, test, m),
            bg_psnr=psnr(reference, test, ~m),
            fg_ssim=ssim(reference, test, m),
            bg_ssim=ssim(reference, test, ~m),
        )
    return out


# ---------------------------------------------------------------------------
# bpp / sıkıştırma oranı — tek tanım, tüm ekranlarda (karşılaştırma, RD
# grafiği, DCT/DWT explorer) bu fonksiyonlar üzerinden kullanılır.
# ---------------------------------------------------------------------------
def raw_bits(shape: tuple[int, ...], bits_per_channel: int = 8) -> float:
    """Sıkıştırılmamış ham payload: H*W*kanal*bits_per_channel (dosya boyutu
    değil — PNG/JPEG kapsayıcı üst verisi hariç, çıplak piksel verisi)."""
    h, w = shape[:2]
    channels = shape[2] if len(shape) == 3 else 1
    return float(h * w * channels * bits_per_channel)


def bpp_from_bits(total_bits: float, shape: tuple[int, ...]) -> float:
    """bit / piksel = toplam_bit / (H*W). Payda KANAL SAYISINI içermez —
    (bu, projede DCT/Wavelet motorlarının döndürdüğü bpp ile aynı tanımdır:
    motorlar total_bits'i zaten tüm kanalların toplamı olarak hesaplar ve
    yalnız H*W'ye böler; renkli görüntüde luma+chroma bitleri tek bpp'de
    birleşir, JPEG/JPEG2000 literatüründeki yaygın kullanımla tutarlıdır)."""
    h, w = shape[:2]
    return float(total_bits) / (h * w)


def compression_ratio(shape: tuple[int, ...], total_bits: float,
                       bits_per_channel: int = 8) -> float:
    """oran = ham_bit / sıkıştırılmış_bit. Ham = H*W*kanal*8 (bkz. raw_bits)."""
    if total_bits <= 0:
        return float("inf")
    return raw_bits(shape, bits_per_channel) / float(total_bits)


def ratio_to_bpp(ratio: float, channels: int = 3, bits_per_channel: int = 8) -> float:
    """Hedef sıkıştırma oranı -> hedef bpp (H*W'ye bölünmüş toplam bit)."""
    return channels * bits_per_channel / max(ratio, 1e-6)


def bpp_to_ratio(bpp: float, channels: int = 3, bits_per_channel: int = 8) -> float:
    """Hedef bpp -> hedef sıkıştırma oranı."""
    return channels * bits_per_channel / max(bpp, 1e-6)


def calculate_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    bpp: float,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Tek doğruluk kaynağı: MSE/PSNR/SSIM/BPP/sıkıştırma oranı/boyut bir arada.

    bpp: dct_engine/wavelet_engine/jpeg2000_engine'in zaten döndürdüğü
    bit/piksel değeri (motorların ortak sözleşmesi: `compress_image` her
    zaman (rekonstrüksiyon, bpp) döner — bkz. bpp_from_bits ile aynı tanım).
    Bu fonksiyon compressed_size ve compression_ratio'yu bpp'den türetir;
    her ekran (karşılaştırma, RD grafiği, explorer) AYNI tanımı kullanır.
    """
    h, w = original.shape[:2]
    total_bits = bpp * h * w
    out = evaluate(original, reconstructed, mask)
    out["mse"] = mse(original, reconstructed)
    out["bpp"] = float(bpp)
    out["compression_ratio"] = compression_ratio(original.shape, total_bits)
    out["compressed_size_bytes"] = total_bits / 8.0
    # "Orijinal boyut" burada HAM GÖRÜNTÜ boyutudur (H*W*kanal*8 bit,
    # bkz. raw_bits) — YÜKLENEN DOSYANIN disk boyutu DEĞİLDİR (PNG/JPEG
    # kapsayıcı sıkıştırması projenin transform/kuantalama pipeline'ıyla
    # ilgisizdir). compression_ratio ZATEN aynı tanımı (raw_bits) paydası
    # olarak kullanıyor — bu yüzden size_reduction_pct, compression_ratio
    # ile MATEMATİKSEL OLARAK TUTARLIDIR (elma-elma karşılaştırma, mega-spec
    # "FINAL FEATURE PASS" Part 6/7/8): reduction = 100*(1 - 1/ratio).
    out["original_size_bytes"] = raw_bits(original.shape) / 8.0
    out["size_reduction_pct"] = (
        100.0 * (1.0 - out["compressed_size_bytes"] / out["original_size_bytes"])
        if out["original_size_bytes"] > 0 else float("nan")
    )
    return out
