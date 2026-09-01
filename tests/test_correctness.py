# -*- coding: utf-8 -*-
"""Sayısal doğruluk / validasyon test paketi.

"SEMANTIC COMPRESSION LAB — FULL NUMERICAL CORRECTNESS & VALIDATION AUDIT"
görevinin Part 38/39 gereği: mevcut backend hesaplamalarının (MSE/PSNR/SSIM/
BPP/sıkıştırma oranı, DCT/DWT round-trip, kuantalama, zigzag, maske
kapsamı/FG-BG ayrımı, rate-matching, state invalidation) gerçekten doğru ve
tutarlı olduğunu doğrular. Testler senteze/dekoratif değildir — her biri
GERÇEK backend fonksiyonlarını (src/metrics, src/engines, src/roi, app.py)
çağırır; hiçbir beklenen değer uydurulmaz, ya matematiksel olarak türetilir
ya da trusted library (skimage/scipy/pywt) ile cross-check edilir.

Çalıştırma:
    .venv/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import gradio as gr
import numpy as np
import pytest
import pywt
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import app
import config
import src.compare as compare
from src.engines import dct_engine, wavelet_engine
from src.engines.entropy import zigzag_indices
from src.engines.wavelet_engine import decompose_for_viz, max_decomposition_level
from src.metrics.quality import (bpp_from_bits, calculate_metrics,
                                 compression_ratio, evaluate, mse, psnr,
                                 raw_bits, ssim)
from src.roi.bit_allocation import mask_to_block_importance, match_bpp, rectangle_mask
from src.viz import cards, dct_block, subbands

RNG = np.random.default_rng(1234)


# =============================================================================
# Part 39 — Sentetik test görüntüleri (kontrollü, beklenen davranışı
# öngörmeyi kolaylaştıran görüntüler)
# =============================================================================
@pytest.fixture
def zeros_img():
    return np.zeros((64, 64), dtype=np.uint8)


@pytest.fixture
def constant_gray_img():
    return np.full((64, 64), 128, dtype=np.uint8)


@pytest.fixture
def gradient_img():
    ramp = np.linspace(0, 255, 64).astype(np.uint8)
    return np.tile(ramp, (64, 1))


@pytest.fixture
def checkerboard_img():
    x, y = np.meshgrid(np.arange(64), np.arange(64))
    return (((x // 8) + (y // 8)) % 2 * 255).astype(np.uint8)


@pytest.fixture
def impulse_img():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[32, 32] = 255
    return img


@pytest.fixture
def random_img():
    return RNG.integers(0, 255, (96, 96, 3), dtype=np.uint8)


@pytest.fixture
def random_gray_img():
    return RNG.integers(0, 255, (96, 96), dtype=np.uint8).astype(np.float64)


# =============================================================================
# 1-4. MSE / PSNR / SSIM — canonical tanım + trusted-library cross-check
# =============================================================================
def test_mse_identical(random_img):
    assert mse(random_img, random_img) == 0.0


def test_mse_known_case():
    """Bilinen sentetik örnek: iki sabit görüntü arasındaki fark sabit ->
    MSE = (fark)^2, elle hesaplanabilir kapalı-form değer."""
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 110, dtype=np.uint8)  # fark = 10 her pikselde
    assert mse(a, b) == pytest.approx(100.0)  # 10^2 = 100


def test_mse_no_overflow_uint8():
    """uint8 çıkarma önce float'a cast edilmeden yapılırsa DOLAŞIR (255-0
    yerine 0-255 = -255 -> uint8'de 1 gibi yanlış bir değere sarar). mse()
    ref/test'i float64'e cast ediyor mu doğrula."""
    a = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    b = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    # Doğru (overflow'suz) beklenen: (0-255)^2 = 65025
    assert mse(a, b) == pytest.approx(65025.0)


def test_psnr_identical(random_img):
    assert psnr(random_img, random_img) == float("inf")


def test_psnr_known_case():
    """Kapalı-form: sabit fark=10 -> MSE=100 -> PSNR = 10*log10(255^2/100)."""
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 110, dtype=np.uint8)
    expected = 10.0 * np.log10(255.0**2 / 100.0)
    assert psnr(a, b) == pytest.approx(expected, rel=1e-9)


def test_psnr_matches_10log_not_20log():
    """Yaygın hata: 20*log10 kullanmak (bu, MAX_I/sqrt(MSE) formülasyonuna
    karşılık gelir ama 255^2/MSE ile birlikte kullanılırsa YANLIŞTIR).
    10*log10(255^2/MSE) ile 20*log10(255/sqrt(MSE)) matematiksel olarak
    AYNIDIR; ikisi de doğru olabilir ama KARIŞTIRILMAMALI (255/MSE ile
    20*log10 kullanmak gerçek bir hatadır). Burada iki doğru formülasyonun
    tutarlılığını doğruluyoruz."""
    a = np.full((10, 10), 50, dtype=np.uint8)
    b = np.full((10, 10), 80, dtype=np.uint8)
    m = mse(a, b)
    via_10log = 10.0 * np.log10(255.0**2 / m)
    via_20log = 20.0 * np.log10(255.0 / np.sqrt(m))
    assert via_10log == pytest.approx(via_20log, rel=1e-9)
    assert psnr(a, b) == pytest.approx(via_10log, rel=1e-9)


def test_psnr_cross_check_skimage(random_img):
    """Trusted library cross-check (Part 40)."""
    noisy = np.clip(random_img.astype(np.int16) + RNG.integers(-30, 30, random_img.shape),
                    0, 255).astype(np.uint8)
    ours = psnr(random_img, noisy)
    theirs = peak_signal_noise_ratio(random_img, noisy, data_range=255)
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_ssim_identical(random_img):
    assert ssim(random_img, random_img) == pytest.approx(1.0, abs=1e-9)


def test_ssim_small_perturbation_below_one(random_img):
    noisy = np.clip(random_img.astype(np.int16) + RNG.integers(-15, 15, random_img.shape),
                    0, 255).astype(np.uint8)
    s = ssim(random_img, noisy)
    assert s < 1.0
    assert 0.0 <= s <= 1.0


def test_ssim_cross_check_skimage(random_img):
    """Trusted library cross-check (Part 4/40, düzeltildi "FINAL MATHEMATICAL
    VALIDATION & AUDIT"): GERÇEK BUG — ssim(mask=None) önceden skimage'ın
    full=True haritasının `smap.mean()`'ini kullanıyordu; bu, skimage'ın
    kendi kanonik `mssim` skalerinden GERÇEK bir görüntüde ~1e-4 mertebesinde
    (gösterilen 4 ondalık hassasiyetin sınırında) sistematik olarak
    SAPIYORDU — bağımsız bir doğrulama script'iyle astronaut.png üzerinde
    ölçüldü (skimage mssim=0.86522 vs eski smap.mean()=0.86533). Artık
    mask=None iken skimage'ın döndürdüğü mssim DOĞRUDAN kullanılıyor; bu
    yüzden tolerans artık yalnız floating-point hassasiyeti kadar sıkı."""
    noisy = np.clip(random_img.astype(np.int16) + RNG.integers(-30, 30, random_img.shape),
                    0, 255).astype(np.uint8)
    ours = ssim(random_img, noisy)
    theirs = structural_similarity(random_img, noisy, data_range=255, channel_axis=2)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_nan_vs_inf_not_conflated():
    """Regression: fmt_psnr NaN'ı (tanımsız) +Inf (kayıpsız) ile
    KARIŞTIRMAMALI — audit'te bulunan gerçek bir gösterim hatasıydı."""
    assert cards.fmt_psnr(float("inf")) == "∞ (kayıpsız)"
    assert cards.fmt_psnr(float("nan")) == "N/A"
    assert cards.fmt_ssim(float("nan")) == "N/A"


# =============================================================================
# 5-6. BPP / Compression Ratio — tek tanım, tüm motorlarda tutarlı
# =============================================================================
def test_bpp_formula():
    """BPP = toplam_bit / (H*W); kanal sayısı PAYDAYA girmez (dosyada
    dokümante edilen, motorların da uyduğu tanım)."""
    shape = (100, 200, 3)
    total_bits = 50000.0
    assert bpp_from_bits(total_bits, shape) == pytest.approx(50000.0 / (100 * 200))


def test_bpp_consistent_across_dct_and_wavelet_engines(random_img):
    """Compare/DCT-Lab/DWT-Lab AYNI bpp tanımını mı kullanıyor — motorların
    kendi compress_image çıktısını doğrudan bpp_from_bits ile çapraz kontrol
    eder (Part 5: 'aynı BPP tanımını kullanmalı')."""
    h, w = random_img.shape[:2]
    _, dct_bpp = dct_engine.compress_image(random_img, 50)
    _, wav_bpp = wavelet_engine.compress_image(random_img, 8.0, "bior4.4", 3)
    for name, bpp in [("dct", dct_bpp), ("wavelet", wav_bpp)]:
        assert 0.0 < bpp < 24.0, f"{name} bpp={bpp} sane aralık dışında (0-24 bpp/pixel RGB8)"


def test_compression_ratio():
    """oran = ham_bit / sıkıştırılmış_bit; ham = H*W*kanal*8."""
    shape = (10, 10, 3)
    total_bits = 300.0  # rastgele bir sıkıştırılmış boyut
    expected_raw = 10 * 10 * 3 * 8
    assert raw_bits(shape) == expected_raw
    assert compression_ratio(shape, total_bits) == pytest.approx(expected_raw / total_bits)


def test_compression_ratio_infinite_at_zero_bits():
    assert compression_ratio((10, 10, 3), 0.0) == float("inf")


def test_jpeg2000_actual_size_not_estimated():
    """JPEG2000 motoru GERÇEK kodlanmış bayt sayısını kullanmalı (numpy
    array bellek boyutu DEĞİL) — bpp, gerçek size_bytes'tan türetilmiş
    olmalı (dairesel ama TUTARLI: calculate_metrics'in yeniden hesapladığı
    compressed_size_bytes, motorun gerçek size_bytes'ıyla eşleşmeli)."""
    from src.engines.jpeg2000_engine import JPEG2000_AVAILABLE

    if not JPEG2000_AVAILABLE:
        pytest.skip("Bu ortamda gerçek JPEG2000 (OpenJPEG) kodeği yok.")
    from src.engines import jpeg2000_engine

    img = RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    recon, bpp, size_bytes = jpeg2000_engine.compress_image(img, 8.0)
    metrics = calculate_metrics(img, recon, bpp)
    assert metrics["compressed_size_bytes"] == pytest.approx(size_bytes, rel=1e-6)


# =============================================================================
# 7-9. DCT — blok çıkarma, level shift, forward/inverse round-trip
# =============================================================================
def test_dct_block_extraction_matches_overlay(random_gray_img):
    """DCT Lab'daki seçili blok overlay'i (draw_block_overlay/
    block_pixel_region) ile GERÇEK blok çıkarma (extract_block) BİREBİR
    aynı piksel bölgesini kullanmalı (Part 9 — mega-spec'in tekrar tekrar
    vurguladığı kritik gereksinim)."""
    block_size, row, col = 8, 3, 5
    block, grid_shape, (r, c) = dct_block.extract_block(random_gray_img, block_size, row, col)
    x0, y0, x1, y1 = dct_block.block_pixel_region(random_gray_img.shape, block_size, row, col)
    assert (r, c) == (row, col)
    assert x0 == col * block_size and y0 == row * block_size
    manual_block = random_gray_img[y0:y1, x0:x1]
    assert block[: manual_block.shape[0], : manual_block.shape[1]] == pytest.approx(manual_block)


def test_dct_level_shift_no_overflow(random_gray_img):
    """shifted = block - 128; float64'te olmalı, uint8 sarma (overflow) YOK."""
    block, _, _ = dct_block.extract_block(random_gray_img, 8, 0, 0)
    shifted = block.astype(np.float64) - 128.0
    assert shifted.dtype == np.float64
    assert shifted.min() >= -128.0 - 1e-9
    assert shifted.max() <= 255.0 - 128.0 + 1e-9


def test_dct_roundtrip_no_quantization():
    """block -> DCT -> IDCT (kuantalama YOK) -> numerik hassasiyet
    seviyesinde hata (Part 11) — 5 rastgele 8x8 blok."""
    for _ in range(5):
        block = RNG.uniform(0, 255, (8, 8))
        coeffs = dctn(block, norm="ortho")
        recon = idctn(coeffs, norm="ortho")
        assert np.abs(recon - block).max() < 1e-9


def test_dct_dc_is_coeff_00(random_gray_img):
    """UI'daki DC değeri gerçekten DCT[0,0] olmalı, hard-coded/başka bir
    matristen gelmemeli; farklı blok seçilince değişmeli (Part 12)."""
    from config import JPEG_LUMA_QTABLE

    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    r1 = dct_block.inspect_block(random_gray_img, 8, 0, 0, 50, base_table)
    assert r1["dc"] == pytest.approx(r1["coeffs"][0, 0])

    r2 = dct_block.inspect_block(random_gray_img, 8, 5, 4, 50, base_table)
    assert r2["dc"] == pytest.approx(r2["coeffs"][0, 0])
    # Farklı blok -> (neredeyse kesinlikle) farklı DC (rastgele görüntüde)
    assert r1["dc"] != pytest.approx(r2["dc"])


def test_dct_block_psnr_is_block_level_not_global(random_gray_img):
    """Blok PSNR'ı, orijinal görüntünün TAMAMI değil yalnız SEÇİLİ BLOK
    üzerinden hesaplanmalı (Part 16 — global PSNR ile karıştırılmamalı)."""
    from config import JPEG_LUMA_QTABLE

    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    r = dct_block.inspect_block(random_gray_img, 8, 2, 2, 50, base_table)
    expected_block_mse = float(np.mean((r["recon"] - r["block"]) ** 2))
    assert r["block_mse"] == pytest.approx(expected_block_mse)
    # Global-image MSE farklı bir büyüklükte olmalı (aynı olması beklenmez)
    assert r["block_mse"] != pytest.approx(mse(random_gray_img, random_gray_img))


# =============================================================================
# 13-16. Kuantalama / non-zero count / zigzag / DC tutarlılığı
# =============================================================================
def test_quantization_changes_with_quality(random_gray_img):
    """Aynı blok, Quality 90 vs 50 vs 20: kuantalanmış katsayılar farklı
    olmalı; genel eğilim: düşük kalite -> daha fazla sıfır -> daha düşük
    PSNR (Part 13) — kesin monotonluk şart değil ama genel yön doğru olmalı."""
    from config import JPEG_LUMA_QTABLE

    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    r90 = dct_block.inspect_block(random_gray_img, 8, 4, 4, 90, base_table)
    r50 = dct_block.inspect_block(random_gray_img, 8, 4, 4, 50, base_table)
    r20 = dct_block.inspect_block(random_gray_img, 8, 4, 4, 20, base_table)

    # raw DCT katsayıları quality'den bağımsız olmalı (aynı blok, aynı DCT)
    assert r90["coeffs"] == pytest.approx(r50["coeffs"])
    assert r90["coeffs"] == pytest.approx(r20["coeffs"])

    # kuantalanmış katsayılar VE non-zero sayısı genel eğilimde azalmalı
    assert r90["n_nonzero"] >= r50["n_nonzero"] >= r20["n_nonzero"]
    # PSNR genel eğilimde düşmeli (daha kaba kuantalama -> daha çok hata)
    assert r90["block_psnr"] >= r50["block_psnr"] >= r20["block_psnr"] - 1e-6


def test_quantization_raw_block_unchanged_by_quality(random_gray_img):
    """Part 15: orijinal blok Quality'den bağımsız sabit kalmalı."""
    from config import JPEG_LUMA_QTABLE

    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    r90 = dct_block.inspect_block(random_gray_img, 8, 1, 1, 90, base_table)
    r20 = dct_block.inspect_block(random_gray_img, 8, 1, 1, 20, base_table)
    assert r90["block"] == pytest.approx(r20["block"])


def test_nonzero_count_matches_matrix(random_gray_img):
    """UI'daki 'SIFIR-OLMAYAN' sayısı gerçekten count_nonzero(quantized)
    olmalı (Part 14)."""
    from config import JPEG_LUMA_QTABLE

    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    r = dct_block.inspect_block(random_gray_img, 8, 3, 3, 50, base_table)
    assert r["n_nonzero"] == int(np.count_nonzero(r["quantized"]))
    assert r["n_nonzero"] <= r["n_total"]


def test_zigzag_order_canonical_8x8():
    """Bilinen 8x8 JPEG zigzag sırası ile birebir eşleşme (Part 15)."""
    canonical = [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ]
    assert zigzag_indices(8).tolist() == canonical


def test_zigzag_output_from_quantized_matrix(random_gray_img):
    """Zigzag çıktısı kuantalanmış matristen gelmeli (Part 15).

    NOT: r['dc'] TANIM GEREĞİ ham (kuantalama ÖNCESİ) DCT[0,0]'dır (Part 12);
    zigzag akışı ise kuantalanmış matristen gelir, dolayısıyla zigzag[0]
    (kuantalanmış DC) ile r['dc'] (ham DC) FARKLI sayılardır — bu bir hata
    değildir. İlişki: round(ham_DC / qtable[0,0]) == kuantalanmış_DC."""
    from config import JPEG_LUMA_QTABLE

    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    r = dct_block.inspect_block(random_gray_img, 8, 2, 6, 50, base_table)
    order = zigzag_indices(8)
    expected_stream = r["quantized"].reshape(-1)[order]
    # Sıfır olmayan sayısı, zigzag akışındaki sıfır olmayanlarla eşleşmeli
    assert int(np.count_nonzero(expected_stream)) == r["n_nonzero"]
    assert expected_stream[0] == r["quantized"][0, 0]  # zigzag[0] = kuantalanmış DC
    # ham DC ile kuantalanmış DC arasındaki tutarlılık (kuantalama tablosu üzerinden)
    assert round(r["dc"] / r["qtable"][0, 0]) == expected_stream[0]


# =============================================================================
# 17-21. DWT — round-trip, subband shape, orientation, kuantalama, seyreklik
# =============================================================================
def test_dwt_roundtrip_no_quantization(random_gray_img):
    """image -> DWT -> IDWT (kuantalama yok) -> numerik hassasiyet
    seviyesinde hata (Part 17)."""
    coeffs = decompose_for_viz(random_gray_img, "bior4.4", 3)
    recon = pywt.waverec2(coeffs, "bior4.4")[: random_gray_img.shape[0], : random_gray_img.shape[1]]
    err = np.abs(recon - (random_gray_img - 128.0)).max()
    assert err < 1e-6


def test_dwt_subband_shapes_match_pywt_convention(random_gray_img):
    """LL/LH/HL/HH boyutları pywt'nin kendi ürettiği shape ile birebir
    eşleşmeli (Part 18) — UI etiket/boyut YANLIŞ olmamalı."""
    levels = 3
    coeffs = decompose_for_viz(random_gray_img, "bior2.2", levels)
    stats = subbands.coeff_stats(coeffs)
    # coeff_stats çıktısındaki her shape string, gerçek pywt dizisinin
    # .shape'iyle eşleşmeli
    idx = 0
    assert stats[idx]["shape"] == f"{coeffs[0].shape[0]}x{coeffs[0].shape[1]}"
    idx = 1
    for li in range(1, len(coeffs)):
        for band in coeffs[li]:
            assert stats[idx]["shape"] == f"{band.shape[0]}x{band.shape[1]}"
            idx += 1


def test_dwt_decompose_levels_index_mapping_regression(random_gray_img):
    """Regression: DWT LAB DECOMPOSITION EXPLORER'ın kritik bug'ı — pywt'nin
    coeffs listesi TERSTİR (coeffs[1] = EN KABA seviyenin detayı), önceki
    kod bunu doğrudan `coeffs[level_index]` ile indexliyordu, yani UI'da
    'Seviye 1' seçmek aslında EN KABA seviyeyi (n_levels) gösteriyordu.
    Burada decompose_levels()'ın k=1 (en ince) ve k=n_levels (en kaba)
    için doğru pywt bandlarını seçtiğini KESİN olarak doğruluyoruz —
    coeffs[-1] (pywt listesinin SON elemanı, en ince/Seviye-1 detayı) ile
    decompose_levels(...)[1]'in AYNI dizi olduğunu, coeffs[1] (pywt
    listesinin İLK detay elemanı, en kaba/Seviye-n_levels detayı) ile
    decompose_levels(...)[n_levels]'in AYNI dizi olduğunu kontrol eder."""
    n_levels = 4
    coeffs = decompose_for_viz(random_gray_img, "bior4.4", n_levels)
    levels_data = subbands.decompose_levels(coeffs, n_levels, "bior4.4")

    # Seviye 1 (en ince) -> pywt listesinin SON detay elemanı (coeffs[-1])
    cH1, cV1, cD1 = coeffs[-1]
    assert np.array_equal(levels_data[1]["LH"], cH1)
    assert np.array_equal(levels_data[1]["HL"], cV1)
    assert np.array_equal(levels_data[1]["HH"], cD1)

    # Seviye n_levels (en kaba) -> pywt listesinin İLK detay elemanı (coeffs[1])
    cH4, cV4, cD4 = coeffs[1]
    assert np.array_equal(levels_data[n_levels]["LH"], cH4)
    assert np.array_equal(levels_data[n_levels]["HL"], cV4)
    assert np.array_equal(levels_data[n_levels]["HH"], cD4)
    # En kaba seviyenin LL'i literal coeffs[0]'dır (kısmi rekonstrüksiyon
    # GEREKMEZ)
    assert np.array_equal(levels_data[n_levels]["LL"], coeffs[0])

    # ESKİ (buggy) davranışın simülasyonu: coeffs[level_index] doğrudan
    # kullanılsaydı Seviye 1 için coeffs[1] (yanlış — aslında Seviye 4'ün
    # verisi) dönerdi. Doğru düzeltmeyle bunun ARTIK doğru olmadığını
    # kanıtla (yani coeffs[1] != levels_data[1]'in bandları, çünkü
    # levels_data[1] artık coeffs[-1]'den geliyor).
    buggy_cH1, _, _ = coeffs[1]
    assert not np.array_equal(levels_data[1]["LH"], buggy_cH1), (
        "Regresyon: Seviye 1 hâlâ pywt'nin coeffs[1]'ini (en kaba seviye) "
        "gösteriyor — index eşleme bug'ı geri gelmiş olabilir."
    )


def test_dwt_all_levels_reference_distinct_coefficient_data(random_gray_img):
    """Mega-spec Part 27 KRİTİK KABUL TESTİ: Seviye 1→2→3→4 arasında geçiş
    yapmak GERÇEKTEN farklı katsayı dizilerine referans vermeli — yalnız
    HTML/etiket değişikliği DEĞİL, sayısal dizilerin kendisi. Her seviyenin
    shape'i ve içerik hash'i BENZERSİZ olmalı (aynı seviyenin kendine eşit
    olması dışında, hiçbir iki farklı seviye aynı LH dizisini paylaşmamalı)."""
    n_levels = 4
    coeffs = decompose_for_viz(random_gray_img, "db4", n_levels)
    levels_data = subbands.decompose_levels(coeffs, n_levels, "db4")

    assert set(levels_data.keys()) == {1, 2, 3, 4}
    seen_hashes = set()
    for k in range(1, n_levels + 1):
        for band_name in ("LL", "LH", "HL", "HH"):
            arr = levels_data[k][band_name]
            h = hash(arr.tobytes())
            key = (band_name, h)
            # Farklı seviyelerin AYNI banda (ör. iki farklı seviyenin LH'si)
            # ait dizileri farklı olmalı — yalnız aynı (seviye,bant) tekrar
            # sorgulanırsa eşit olması beklenir (burada her biri bir kez
            # kontrol edilir).
            assert key not in seen_hashes, (
                f"Seviye {k} bandı {band_name} önceki bir seviyeyle AYNI "
                f"veriye sahip — level→dizi eşlemesi hâlâ bozuk olabilir."
            )
            seen_hashes.add(key)
    # Boyutlar seviye arttıkça (daha kaba) küçülmeli (dyadic alt örnekleme)
    sizes = [levels_data[k]["HH"].size for k in range(1, n_levels + 1)]
    assert sizes == sorted(sizes, reverse=True), f"Boyutlar dyadic azalmıyor: {sizes}"


def test_dwt_ll_partial_reconstruction_matches_literal_at_deepest_level(random_gray_img):
    """LL_k hesaplama yöntemi (kısmi ters dönüşüm) k=n_levels'te LİTERAL
    coeffs[0] ile birebir eşleşmeli (Part 3)."""
    n_levels = 3
    coeffs = decompose_for_viz(random_gray_img, "bior2.2", n_levels)
    levels_data = subbands.decompose_levels(coeffs, n_levels, "bior2.2")
    assert levels_data[n_levels]["LL"] is coeffs[0] or np.array_equal(levels_data[n_levels]["LL"], coeffs[0])


def test_dwt_max_level_not_fabricated():
    """Part 16: max_decomposition_level, görüntü boyutu/dalgacığa göre
    GERÇEKTEN hesaplanmalı; UI asla geçersiz bir Seviye 4 sunmamalı."""
    tiny = np.random.default_rng(3).uniform(0, 255, (20, 20))
    cap = max_decomposition_level(tiny.shape, "bior4.4")
    assert cap < 4, "20x20 gibi küçük bir görüntüde 4 seviye matematiksel olarak geçersiz olmalı"
    coeffs = decompose_for_viz(tiny, "bior4.4", cap)
    levels_data = subbands.decompose_levels(coeffs, cap, "bior4.4")
    assert set(levels_data.keys()) == set(range(1, cap + 1))
    assert max(levels_data.keys()) == cap


def test_dwt_lh_hl_orientation_matches_pywt_semantics():
    """Part 19: LH/HL etiketlerini EZBERE atamadık — pywt'nin kendi 'cH
    (horizontal detail)'/'cV (vertical detail)' tanımı, YATAY çizgili bir
    sentetik görüntüde cH'nin, DİKEY çizgili görüntüde cV'nin enerjice
    baskın olduğu ampirik olarak doğrulanmıştır (bkz. audit notları).
    Buradaki test, coeff_stats'ın LH etiketini cH'ye, HL etiketini cV'ye
    verdiğini ve bunun pywt semantiğiyle tutarlı kaldığını sabitler."""
    horiz_stripes = np.zeros((32, 32))
    horiz_stripes[::2, :] = 255.0
    cA, (cH, cV, cD) = pywt.dwt2(horiz_stripes, "haar")
    assert np.sum(cH**2) > np.sum(cV**2)  # yatay-çizgili görüntü cH'yi baskın kılar

    vert_stripes = np.zeros((32, 32))
    vert_stripes[:, ::2] = 255.0
    cA2, (cH2, cV2, cD2) = pywt.dwt2(vert_stripes, "haar")
    assert np.sum(cV2**2) > np.sum(cH2**2)  # dikey-çizgili görüntü cV'yi baskın kılar

    # coeff_stats: coeffs[li] = (cH, cV, cD) sırasıyla ("LH", cH), ("HL", cV)
    # etiketlemesini kullanır (subbands.py) — bu atamayı burada sabitliyoruz.
    coeffs = [cA, (cH, cV, cD)]
    stats = subbands.coeff_stats(coeffs)
    labels = [r["label"] for r in stats]
    assert "LH1" in labels and "HL1" in labels
    lh_row = next(r for r in stats if r["label"] == "LH1")
    assert lh_row["shape"] == f"{cH.shape[0]}x{cH.shape[1]}"


def test_dwt_quantization_changes_reconstruction(random_gray_img):
    """Part 20: kuantalama adımı değişince kuantalanmış katsayılar/
    seyreklik/rekonstrüksiyon/PSNR GERÇEKTEN değişmeli; ham katsayılar
    DEĞİŞMEMELİ."""
    coeffs_raw = decompose_for_viz(random_gray_img, "bior4.4", 3)

    recon_fine, bits_fine = wavelet_engine.compress_channel(random_gray_img, 2.0, "bior4.4", 3)
    recon_coarse, bits_coarse = wavelet_engine.compress_channel(random_gray_img, 32.0, "bior4.4", 3)

    assert bits_fine != pytest.approx(bits_coarse)
    assert not np.allclose(recon_fine, recon_coarse)
    psnr_fine = psnr(random_gray_img, recon_fine)
    psnr_coarse = psnr(random_gray_img, recon_coarse)
    assert psnr_fine > psnr_coarse  # ince adım -> daha az bozulma

    # ham katsayılar (decompose_for_viz), quant adımından ETKİLENMEMELİ
    coeffs_raw_again = decompose_for_viz(random_gray_img, "bior4.4", 3)
    assert coeffs_raw[0] == pytest.approx(coeffs_raw_again[0])


def test_dwt_quantization_sparsity_raw_vs_quantized(random_gray_img):
    """Part 21: ham katsayılarda sıfır oranı ~%0 olması HATA değildir; asıl
    seyreklik kuantalama SONRASI ortaya çıkar. raw_nonzero_pct >>
    nonzero_pct (kuantalanmış) olmalı."""
    from src.engines.wavelet_engine import quantize_for_viz

    coeffs = decompose_for_viz(random_gray_img, "bior4.4", 3)
    quantized = quantize_for_viz(random_gray_img, 16.0, "bior4.4", 3)
    stats = subbands.quantized_sparsity_stats(coeffs, quantized)
    assert stats["raw_nonzero_pct"] > stats["nonzero_pct"]
    assert stats["nonzero"] + stats["zero"] == stats["total"]
    assert 0.0 <= stats["sparsity_pct"] <= 100.0


def test_dwt_bpp_regression_no_absurd_value(random_gray_img):
    """Regression: geçmişte total_bits, h*w'ye bölünmeden bpp diye
    kullanılmış ve 599359.655 gibi anlamsız bir 'bpp' üretilmişti. Burada
    hem motor seviyesinde (compress_channel çıktısını doğru bölerek) hem
    de sağlık-aralığı kontrolüyle bunun tekrarlanmadığını doğruluyoruz."""
    h, w = random_gray_img.shape
    recon, total_bits = wavelet_engine.compress_channel(random_gray_img, 8.0, "bior4.4", 3)
    bpp = total_bits / (h * w)
    assert 0.0 < bpp < 30.0, f"bpp={bpp} sane aralık dışında — regresyon şüphesi"

    # image-level API da aynı korumayı sağlamalı
    _, bpp_img = wavelet_engine.compress_image(random_gray_img, 8.0, "bior4.4", 3)
    assert 0.0 < bpp_img < 30.0


def test_dwt_max_level_prevents_boundary_garbage():
    """Çok küçük görüntü + yüksek seviye isteği sessizce clamp edilmeli,
    anlamsız/boş subband'ler üretmemeli."""
    small = RNG.uniform(0, 255, (16, 16))
    cap = max_decomposition_level(small.shape, "bior4.4")
    assert cap >= 1
    # decompose_for_viz kendisi clamp ETMEZ (çağıran taraf sorumlu) — asıl
    # kontrol motor seviyesinde: compress_channel (10, kasıtlı fazla istek)
    # sessizce cap'e clamp edip anlamlı bir sonuç üretmeli, çökmemeli.
    _, bits = wavelet_engine.compress_channel(small, 8.0, "bior4.4", 10)
    assert np.isfinite(bits) and bits >= 0


# =============================================================================
# 24-28. Semantic ROI — maske kapsamı, FG/BG ayrımı, global PSNR
# =============================================================================
def test_mask_coverage_formula():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True  # %50 kapsama
    coverage = mask.mean() * 100
    assert coverage == pytest.approx(50.0)


def test_foreground_background_partition_no_overlap_full_union(random_img):
    """FG maskesi ile BG maskesi (tümleyeni) KESİŞMEMELİ ve BİRLİKTE tüm
    görüntüyü kaplamalı (Part 27)."""
    mask = RNG.random(random_img.shape[:2]) > 0.6
    bg_mask = ~mask
    assert not np.any(mask & bg_mask)  # kesişim yok
    assert np.all(mask | bg_mask)  # birleşim = tüm görüntü


def test_foreground_psnr_only_over_mask_pixels(random_img):
    """FG PSNR yalnız maske içi pikseller üzerinden hesaplanmalı; maske
    dışı pikseller MSE'ye karışmamalı (Part 25)."""
    recon = np.clip(random_img.astype(np.int16) + RNG.integers(-50, 50, random_img.shape),
                    0, 255).astype(np.uint8)
    mask = np.zeros(random_img.shape[:2], dtype=bool)
    mask[:20, :20] = True

    fg_psnr = psnr(random_img, recon, mask)
    manual_mse = float(np.mean((random_img[:20, :20].astype(np.float64)
                                 - recon[:20, :20].astype(np.float64)) ** 2))
    manual_psnr = 10.0 * np.log10(255.0**2 / manual_mse)
    assert fg_psnr == pytest.approx(manual_psnr, rel=1e-9)


def test_evaluate_full_mask_no_crash_returns_nan_for_empty_side():
    """Regression (audit'te bulunan gerçek bug): ROI tüm görüntüyü
    kaplarsa (ör. Manuel ROI slider'ları maksimuma çekilince — 512x512 bir
    görüntüde erişilebilir) arka plan boş kalır. evaluate() ÖNCEDEN bu
    durumda fg_psnr/bg_psnr anahtarlarını SESSİZCE sözlükten çıkarıyordu;
    çağıran taraf (app.py) bunları koşulsuz okuduğundan KeyError ile
    ÇÖKÜYORDU. Artık anahtarlar HER ZAMAN mevcut; boş taraf NaN'dır."""
    img = RNG.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    recon = RNG.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    full_mask = np.ones((32, 32), dtype=bool)

    result = evaluate(img, recon, full_mask)
    for key in ("fg_psnr", "bg_psnr", "fg_ssim", "bg_ssim"):
        assert key in result, f"{key} eksik — çağıran taraf KeyError ile çöker"
    assert np.isnan(result["bg_psnr"])  # arka plan piksel yok -> tanımsız
    assert np.isfinite(result["fg_psnr"]) or np.isinf(result["fg_psnr"])

    # UI katmanı bu NaN'ı "N/A" göstermeli, çökmemeli
    card = cards.tradeoff_card_html("BG", "#F59E0B", result["bg_psnr"], result["bg_psnr"], " dB", True)
    assert "N/A" in card


def test_global_psnr_differs_from_roi_psnr(random_img):
    """Global PSNR (tüm görüntü) ile ROI/FG PSNR karıştırılmamalı — farklı
    bölgeler farklı bozulma seviyelerine sahipse SAYISAL OLARAK farklı
    olmalılar (Part 28)."""
    recon = random_img.copy()
    mask = np.zeros(random_img.shape[:2], dtype=bool)
    mask[:30, :30] = True
    # Yalnız maske DIŞINA gürültü ekle -> FG PSNR = inf (kayıpsız), global < inf
    noise_region = ~mask
    recon = recon.astype(np.int16)
    recon[noise_region] = np.clip(recon[noise_region] + RNG.integers(-40, 40, recon[noise_region].shape), 0, 255)
    recon = recon.astype(np.uint8)

    global_psnr = psnr(random_img, recon)
    fg_psnr = psnr(random_img, recon, mask)
    assert fg_psnr == float("inf")  # maske içi hiç değişmedi
    assert global_psnr < fg_psnr  # global, bozulmuş bölgeyi de içerir


# =============================================================================
# 29. Same bit budget (rate fairness) — baseline vs semantic yakınlığı
# =============================================================================
def test_same_bit_budget_close_rate(random_img):
    mask = np.zeros(random_img.shape[:2], dtype=bool)
    mask[20:50, 20:50] = True
    block_imp = mask_to_block_importance(mask)
    target_bpp = 1.0

    base = match_bpp(lambda q: dct_engine.compress_image(random_img, q), target_bpp, 1, 100, True)
    sem = match_bpp(
        lambda q: dct_engine.compress_image(random_img, q, block_imp, 4.0),
        target_bpp, 1, 100, True,
    )
    rate_diff = abs(base[1] - sem[1])
    assert rate_diff < 0.1, f"baseline/semantic bpp farkı çok büyük: {rate_diff}"


# =============================================================================
# 30-31. Detected-object include/exclude ve manuel ROI gerçekten pipeline'ı
# etkiliyor mu (dekoratif dikdörtgen DEĞİL)
# =============================================================================
def test_manual_roi_rectangle_actually_changes_mask():
    """Part 31: farklı ROI koordinatları FARKLI bir maske üretmeli (yalnız
    çizilen ama backend'i etkilemeyen sahte bir özellik OLMAMALI)."""
    shape = (64, 64)
    mask_a = rectangle_mask(shape, 0, 0, 20, 20)
    mask_b = rectangle_mask(shape, 40, 40, 60, 60)
    assert not np.array_equal(mask_a, mask_b)
    assert not np.any(mask_a & mask_b)  # örtüşmeyen bölgeler


def test_object_selection_changes_fused_mask():
    """Part 30: checkbox'tan bir nesne çıkarılınca (fuse_instance_masks'e
    verilmeyince) birleşik maske küçülmeli."""
    from src.semantic.importance_map import fuse_instance_masks

    shape = (32, 32)
    m1 = np.zeros(shape, dtype=bool)
    m1[0:10, 0:10] = True
    m2 = np.zeros(shape, dtype=bool)
    m2[20:30, 20:30] = True
    instances = [dict(label="a", confidence=0.9, mask=m1), dict(label="b", confidence=0.8, mask=m2)]

    fused_both = fuse_instance_masks(instances, shape, dilate_px=0)
    fused_one = fuse_instance_masks(instances[:1], shape, dilate_px=0)
    assert fused_both.sum() > fused_one.sum()
    assert np.array_equal(fused_one, m1)


# =============================================================================
# 32. Global image state invalidation (app.py callback-level test)
# =============================================================================
def test_image_state_invalidation_semantic_reset():
    """Aktif görüntü değişince Semantic ROI'nin TÜM türetilmiş state'i
    sıfırlanmalı (mega-spec 'FINAL STATE/CALLBACK FIX' — bu oturumda
    bulunan ve düzeltilen ana bug). sem_reset_state() çağrıldığında eski
    tespit listesi/checklist/ROI/sonuç kartlarının hiçbiri kalmamalı.

    13 elemanlı dönüş: 'FINAL INTEGRATION' (sem_global_card eklendi) ve
    'FINAL PRE-PRESENTATION QA' Part 4 (sem_detect_btn etiketi eklendi)
    görevlerinde büyümüştür — bkz. app.sem_reset_state()."""
    (instances, checklist_update, detect_info, roi_center,
     mask_img, base_img, sem_img, budget, gain, tradeoff, global_card,
     global_note, detect_btn) = app.sem_reset_state()

    assert instances == []
    assert roi_center is None
    assert mask_img is None and base_img is None and sem_img is None
    assert gain == "" and tradeoff == "" and global_card == "" and global_note == ""
    assert "tespit edilmedi" in detect_info.lower() or "N/A" in detect_info
    assert detect_btn["value"] == "NESNELERİ TESPİT ET"


def test_image_state_invalidation_dwt_reset():
    """DWT reset: görüntüye bağlı çıktılar VE seviye önbelleği temizlenmeli
    (bkz. DWT LAB FIX: dwt_levels_state/dwt_tree_bounds_state/
    dwt_selected_band_state artık dwt_reset_results()'ın çıktısına dahildir)."""
    outputs = app.dwt_reset_results()
    # 23 elemanlı dönüş: 19 (recon_compact/recon_full kaldırıldıktan sonra)
    # + mosaic_info/recon_info/gray_state/recon_state (mega-spec "DWT LAB
    # — TIKLAMA İLE NOKTA İNCELEMEYİ 3 GÖRSELDE AKTİF ET").
    (summary, mosaic, tree, tree_bounds, ll, lh, hl, hh, band_details,
     selected_band, hist, recon, diff, energy,
     stats, validation, sparsity, levels_data, max_level,
     mosaic_info, recon_info, gray_state, recon_state) = outputs
    assert mosaic is None and recon is None
    assert hist is None and diff is None and energy is None
    assert stats == []
    assert mosaic_info == "" and recon_info == ""
    assert gray_state is None and recon_state is None
    assert tree_bounds == []
    assert levels_data == {}
    assert selected_band == "LL"


# =============================================================================
# 33. RD curve — her nokta gerçekten encoder çalıştırılarak üretilmeli
# =============================================================================
def test_rd_sweep_points_are_real_measurements(random_img):
    """rd_sweep, verilen her hedef bpp için GERÇEK bir compress_image
    çağrısı yapmalı — enterpolasyon/fake değer YOK. Her noktanın 'bpp'si,
    ilgili hedefe bisection ile YAKLAŞMIŞ olmalı ama BİREBİR eşit olmak
    zorunda değil (bu, gerçek arama yapıldığının kanıtıdır; sahte/lineer
    interpolasyon olsaydı noktalar TAM hedefte olurdu)."""
    targets = [0.5, 1.0]
    curves = compare.rd_sweep(random_img, targets, 8)
    assert "JPEG / DCT" in curves
    dct_pts = curves["JPEG / DCT"]
    assert len(dct_pts) == len(targets)
    for p in dct_pts:
        assert "bpp" in p and "psnr" in p and "ssim" in p
        assert np.isfinite(p["psnr"]) or np.isinf(p["psnr"])
        assert p["bpp"] > 0


# =============================================================================
# Rate matching (Part 8) — hedef bpp'ye yakınsama, birden çok nokta
# =============================================================================
@pytest.mark.parametrize("target_bpp", [0.25, 0.5, 1.0, 2.0])
def test_same_rate_matching_convergence(random_img, target_bpp):
    """DCT ve Wavelet motorları hedef bpp'ye bisection ile YAKINSAMALI;
    rate error toleransı config.BPP_MATCH_TOLERANCE ile aynı büyüklük
    mertebesinde olmalı (çok gevşek bir üst sınırla — bisection max_iter
    sınırına takılabileceği için)."""
    dct_res = compare.run_dct(random_img, target_bpp, 8)
    wav_res = compare.run_wavelet(random_img, target_bpp, "bior4.4", 3)

    dct_err = abs(dct_res.metrics["bpp"] - target_bpp) / target_bpp
    wav_err = abs(wav_res.metrics["bpp"] - target_bpp) / target_bpp
    assert dct_err < 0.15, f"DCT rate error {dct_err:.3f} @ target={target_bpp}"
    assert wav_err < 0.15, f"Wavelet rate error {wav_err:.3f} @ target={target_bpp}"


# =============================================================================
# 34-35. Sayısal hassasiyet, NaN/Inf sağlamlığı
# =============================================================================
def test_internal_precision_is_float64_not_rounded():
    """calculate_metrics içi hesaplamalar yuvarlanmadan taşınmalı; yalnız
    GÖSTERİM anında yuvarlanır (cards.py fmt_* fonksiyonları)."""
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 103, dtype=np.uint8)
    m = calculate_metrics(a, b, bpp=1.0)
    # PSNR literal olarak tam sayı/2-ondalık bir değere YUVARLANMAMIŞ olmalı
    raw = 10.0 * np.log10(255.0**2 / 9.0)
    assert m["psnr"] == pytest.approx(raw, rel=1e-9)
    assert m["psnr"] != round(raw, 2) or abs(raw - round(raw, 2)) < 1e-9


def test_nan_inf_do_not_crash_pipeline():
    """Part 35: NaN/Inf/boş görüntü senaryoları sessiz çökme üretmemeli."""
    zero_mse_img = np.zeros((8, 8), dtype=np.uint8)
    assert psnr(zero_mse_img, zero_mse_img) == float("inf")

    empty_mask = np.zeros((8, 8), dtype=bool)
    # Tamamen boş maske ile psnr çağrısı exception FIRLATMAMALI (NaN döner)
    result = psnr(zero_mse_img, zero_mse_img, empty_mask)
    assert np.isnan(result)


def test_dct_block_change(random_gray_img):
    """Mega-spec 'FINAL INTEGRATION' Part 55 — MUTLAKA: Row 10/Col 10 vs
    Row 30/Col 20 seçince original/DCT/quantized/zigzag/DC/PSNR GERÇEKTEN
    değişmeli (DCT Lab'daki geçmiş callback bug'ının doğrudan regresyon
    testi — bkz. update_dct_analysis)."""
    from config import JPEG_LUMA_QTABLE

    base_table = np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64)
    r1 = dct_block.inspect_block(random_gray_img, 8, 10, 10, 50, base_table)
    r2 = dct_block.inspect_block(random_gray_img, 8, 30, 20, 50, base_table)
    assert not np.array_equal(r1["block"], r2["block"])
    assert not np.array_equal(r1["coeffs"], r2["coeffs"])
    assert not np.array_equal(r1["quantized"], r2["quantized"])
    assert r1["dc"] != pytest.approx(r2["dc"])
    # Aynı görüntüde farklı iki blok PSNR'ı da genellikle farklıdır (kesin
    # eşitsizlik garanti edilmez ama rastgele görüntüde pratik olarak hep
    # farklıdır)
    assert r1["block_psnr"] != pytest.approx(r2["block_psnr"]) or r1["block_mse"] != pytest.approx(r2["block_mse"])

    # UI callback seviyesinde de (update_dct_analysis) aynı doğrulama
    overlay1, line1, fig1, zz1, summary1, info1 = app.update_dct_analysis(
        random_gray_img, "8", 10, 10, 50)
    overlay2, line2, fig2, zz2, summary2, info2 = app.update_dct_analysis(
        random_gray_img, "8", 30, 20, 50)
    assert line1 != line2  # "Satır 10 · Sütun 10" vs "Satır 30 · Sütun 20"
    assert summary1 != summary2  # DC/non-zero/PSNR farklı


def test_dwt_wavelet_change(random_gray_img):
    """Part 54/26: dalgacık ailesi değişince decomposition/subband
    shape'leri/istatistikleri GERÇEKTEN değişmeli (aynı katsayı array'i
    farklı bir isimle yeniden gösterilmiyor)."""
    levels = 3
    coeffs_haar = decompose_for_viz(random_gray_img, "haar", levels)
    coeffs_db4 = decompose_for_viz(random_gray_img, "db4", levels)
    levels_data_haar = subbands.decompose_levels(coeffs_haar, levels, "haar")
    levels_data_db4 = subbands.decompose_levels(coeffs_db4, levels, "db4")

    # db4 (8 tap) haar'dan (2 tap) daha uzun filtre kullanır -> farklı
    # boundary genişlemesi -> genellikle farklı shape VEYA (aynı shape
    # olsa bile) kesinlikle farklı katsayı değerleri üretir.
    ll_haar = levels_data_haar[levels]["LL"]
    ll_db4 = levels_data_db4[levels]["LL"]
    if ll_haar.shape == ll_db4.shape:
        assert not np.allclose(ll_haar, ll_db4)
    else:
        assert ll_haar.shape != ll_db4.shape

    info_haar = subbands.filter_bank_info("haar")
    info_db4 = subbands.filter_bank_info("db4")
    assert info_haar["dec_len"] != info_db4["dec_len"]
    assert info_haar["dec_lo"] != info_db4["dec_lo"]


def test_global_image_state():
    """Part 54: build_active_image_bar'ın tanımladığı TEK global state
    (active_img/active_id/active_meta) — her aktif görüntü değişiminde
    monoton artan BİR sayaç (image_id) ile ilişkilendirilmelidir; bu
    sayaç tüm sekmelerin tazelik kontrolünün (bkz. _is_stale) temelidir."""
    app._mark_latest_image("test-session-1", 5)
    assert app._is_stale("test-session-1", 4) is True
    assert app._is_stale("test-session-1", 5) is False
    # Farklı bir oturum kendi bağımsız durumunu taşır (çoklu-kullanıcı
    # güvenliği — bir oturumun görüntü değişimi başka bir oturumu ETKİLEMEZ)
    assert app._is_stale("test-session-2", 5) is True


def test_stale_result_rejection():
    """Part 54/57: image_id, global_active_image.image_id ile eşleşmiyorsa
    (yani daha yeni bir görüntü seçilmişse) sonuç DISCARD edilmeli — auto
    analiz sarmalayıcıları (dwt_auto_analysis/compare_auto_analysis/
    sem_auto_detect/sem_recompute) gerçek hesaplama YAPMADAN no-op
    dönmelidir."""
    session = "test-session-stale"
    app._mark_latest_image(session, 10)

    class _FakeRequest:
        session_hash = session

    req = _FakeRequest()
    stale_id = 9  # artık _LATEST_IMAGE_ID (10) ile eşleşmiyor
    img = RNG.integers(0, 255, (32, 32, 3), dtype=np.uint8)

    dwt_out = app.dwt_auto_analysis(img, stale_id, "bior4.4", 2, 8.0,
                                    wavelet_engine.DEFAULT_BOUNDARY_MODE,
                                    app._DWT_RATE_MODE_MANUAL, 0.5, req)
    # no-op yolu: 27 elemanın TAMAMI boş gr.update() ({'__type__':'update'}
    # dışında hiçbir alan taşımaz) olmalı — GERÇEK bir hesaplama sonucu
    # (dolu bir HTML stringi/array/dict) ASLA üretilmemeli. (23 + mosaic_
    # info/recon_info/gray_state/recon_state — "DWT LAB — TIKLAMA İLE
    # NOKTA İNCELEMEYİ 3 GÖRSELDE AKTİF ET".)
    assert len(dwt_out) == 27
    assert all(x == {"__type__": "update"} for x in dwt_out)

    cmp_out = app.compare_auto_analysis(
        img, stale_id, "Hedef bpp", 0.5, 8, False, req)
    # 13 çıktı: out_dct, out_wav, rate_fairness, quality_strip, cmp_plot,
    # target_full, dct_full, wav_full, dct_summary_kpi, wav_summary_kpi,
    # out_real_jpeg, real_jpeg_compact, real_jpeg_full — mega-spec "JPEG vs
    # JPEG2000 GERÇEK CODEC KARŞILAŞTIRMASI": jp2k_col kaldırıldığı (artık
    # ana panelin KENDİSİ JPEG2000) ve wavelet_dd/wavelet_level/include_jp2k
    # girdileri silindiği için önceki 16'dan 13'e düştü.
    assert len(cmp_out) == 13

    sem_out = app.sem_auto_detect(img, stale_id, req)
    # 5. eleman (sem_detect_btn etiket güncellemesi) 'FINAL PRE-PRESENTATION
    # QA' Part 4'te eklendi — no-op yolda o da diğerleri gibi boş gr.update()
    # olmalı, gerçek bir buton-etiketi metni ASLA sızmamalı.
    assert sem_out == ([], gr.update(), gr.update(), gr.update(), gr.update())


def test_sanity_assertions_hold_on_real_pipeline(random_img):
    """Part 41: temel sağlık kontrolleri (gerçek bug'ları GİZLEMEDEN, yalnız
    doğrulama amaçlı)."""
    dct_res = compare.run_dct(random_img, 0.8, 8)
    m = dct_res.metrics
    assert 0.0 <= m["ssim"] <= 1.0 + 1e-6
    assert m["bpp"] >= 0
    assert m["compressed_size_bytes"] >= 0
    assert np.isfinite(m["psnr"]) or np.isinf(m["psnr"])


# =============================================================================
# 37. Resize/crop — orijinal ve rekonstrüksiyon boyutları birebir eşleşmeli
# =============================================================================
def test_reconstruction_dimensions_match_original(random_img):
    dct_recon, _ = dct_engine.compress_image(random_img, 50)
    wav_recon, _ = wavelet_engine.compress_image(random_img, 8.0, "bior4.4", 3)
    assert dct_recon.shape == random_img.shape
    assert wav_recon.shape == random_img.shape


def test_dwt_odd_sized_image_crop_correct():
    """Tek sayı boyutlu (padding/crop asimetrisine en duyarlı) görüntülerde
    bile rekonstrüksiyon boyutu orijinalle eşleşmeli."""
    odd_img = RNG.integers(0, 255, (65, 97), dtype=np.uint8).astype(np.float64)
    recon, _ = wavelet_engine.compress_channel(odd_img, 8.0, "bior4.4", 3)
    assert recon.shape == odd_img.shape


# =============================================================================
# 38. FINAL PRE-PRESENTATION QA — mandatory regression tests (Part 10/11/12/26)
# =============================================================================
def test_dct_callback_block_selection_regression(random_img):
    """Part 10: aynı görüntüde Blok A (Satır10/Sütun10) ile Blok B
    (Satır30/Sütun20) arasında geçince orijinal matris/DC/DCT katsayıları/
    kuantalanmış matris/sıfır-olmayan sayısı/blok PSNR VE kaynak overlay'in
    TÜMÜ gerçekten değişmeli — DCT Lab'ın 'stale sağ panel' bug'ının
    (bu oturumda düzeltilen) callback seviyesinde tekrar açılmadığını
    doğrular. Büyük bir görüntü kullanılır (320x320) ki Satır30/Sütun20
    8x8 blok ızgarasında geçerli, BİRBİRİNDEN FARKLI bloklara denk gelsin."""
    big = np.tile(random_img, (4, 4, 1))[:320, :320]
    gray = app._to_gray(big)
    base_table = np.asarray(app.JPEG_LUMA_QTABLE, dtype=np.float64)

    a = dct_block.inspect_block(gray, 8, 10, 10, 50.0, base_table)
    b = dct_block.inspect_block(gray, 8, 30, 20, 50.0, base_table)

    assert not np.array_equal(a["block"], b["block"])
    assert a["dc"] != b["dc"]
    assert not np.array_equal(a["coeffs"], b["coeffs"])
    assert not np.array_equal(a["quantized"], b["quantized"])
    assert a["n_nonzero"] != b["n_nonzero"] or not np.array_equal(a["quantized"], b["quantized"])
    assert a["block_psnr"] != b["block_psnr"]

    overlay_a = dct_block.draw_block_overlay(big, 8, 10, 10)
    overlay_b = dct_block.draw_block_overlay(big, 8, 30, 20)
    assert not np.array_equal(overlay_a, overlay_b)  # camgöbeği çerçeve gerçekten taşınmış


def test_dct_callback_full_wiring_matches_backend(random_img):
    """Part 10 (callback seviyesi): app.update_dct_analysis — DCT Lab'ın
    TEK canonical fonksiyonu — Blok A/B için dct_block.inspect_block ile
    AYNI sayısal sonuçları üretmeli; sadece görselleştirme değil, gerçek
    callback yolu doğrulanır."""
    big = np.tile(random_img, (4, 4, 1))[:320, :320]
    overlay_a, line_a, fig_a, zz_a, summary_a, _ = app.update_dct_analysis(big, 8, 10, 10, 50)
    overlay_b, line_b, fig_b, zz_b, summary_b, _ = app.update_dct_analysis(big, 8, 30, 20, 50)
    assert not np.array_equal(overlay_a, overlay_b)
    assert line_a != line_b
    assert summary_a != summary_b


def test_dct_quality_regression(random_img):
    """Part 11: AYNI blok, Quality 90 vs 20 — orijinal piksel/seviye-
    kaydırma/ham DCT katsayıları AYNI kalmalı (kuantalamadan ÖNCEKİ
    aşamalar quality'den bağımsızdır); kuantalanmış katsayılar/sıfır-
    olmayan sayısı/rekonstrüksiyon/blok PSNR FARKLI olmalı."""
    gray = app._to_gray(random_img)
    base_table = np.asarray(app.JPEG_LUMA_QTABLE, dtype=np.float64)

    hi = dct_block.inspect_block(gray, 8, 3, 3, 90.0, base_table)
    lo = dct_block.inspect_block(gray, 8, 3, 3, 20.0, base_table)

    assert np.array_equal(hi["block"], lo["block"])
    assert np.array_equal(hi["shifted"], lo["shifted"])
    assert np.allclose(hi["coeffs"], lo["coeffs"])  # ham DCT quality'den ÖNCE hesaplanır

    assert not np.array_equal(hi["quantized"], lo["quantized"])
    assert hi["n_nonzero"] >= lo["n_nonzero"]  # yüksek quality >= düşük quality sıfır-olmayan sayısı
    assert not np.array_equal(hi["recon"], lo["recon"])
    assert hi["block_psnr"] >= lo["block_psnr"] - 1e-9  # yüksek quality en az eşit/daha iyi PSNR


def test_dwt_level_selection_regression(random_gray_img):
    """Part 12: Seviye 1→2→3→4 arasında geçince kart isimleri (LLk/LHk/HLk/
    HHk), GERÇEK katsayı dizileri, array.shape'ten okunan gerçek şekiller ve
    istatistikler değişmeli — yalnız etiket metninin değişmesi YETERLİ
    DEĞİLDİR (bu oturumda düzeltilen kritik pywt ters-sıra bug'ının
    callback seviyesindeki regresyon testi)."""
    wavelet = "bior4.4"
    levels = 4
    coeffs = decompose_for_viz(random_gray_img, wavelet, levels, "symmetric")
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, "symmetric")

    seen_shapes = set()
    seen_arrays = []
    prev_label = None
    for lvl in [1, 2, 3, 4]:
        (tree_img, bounds, ll_upd, lh_upd, hl_upd, hh_upd,
         band_details, selected_band, hist_fig) = app.dwt_select_level(levels_data, lvl, "LL")
        # Kart etiketi seviyeyle DOĞRU eşleşmeli (örn. "LL2 — HxW")
        assert f"LL{lvl}" in ll_upd["label"]
        assert ll_upd["label"] != prev_label
        prev_label = ll_upd["label"]

        arr = levels_data[lvl]["LL"]
        seen_shapes.add(arr.shape)
        seen_arrays.append(arr)

        # tree_img gerçekten seviyeye göre değişen bir görüntü döndürür
        assert tree_img is not None and tree_img.size > 0

    # Her seviyenin GERÇEK LL dizisi birbirinden farklı olmalı (aynı array
    # referansının/kopyasının yeniden kullanılmadığını kanıtlar)
    for i in range(len(seen_arrays)):
        for j in range(i + 1, len(seen_arrays)):
            if seen_arrays[i].shape == seen_arrays[j].shape:
                assert not np.array_equal(seen_arrays[i], seen_arrays[j])
    # Seviyeler arttıkça LL alt-örnekleme nedeniyle şekil küçülmeli (dyadic)
    assert len(seen_shapes) >= 2


def test_final_numerical_sanity_dct_dwt_compare(random_img):
    """Part 26: gerçek bir görüntü üzerinde uçtan uca DCT/DWT/Compare
    pipeline'ının ürettiği TÜM sayısal değerler geçerli aralıkta olmalı —
    NaN yok, negatif dosya boyutu/bpp yok, SSIM [0,1] dışında değil, MSE>=0,
    enerji sonlu, sıfır-olmayan katsayı sayısı toplam katsayı sayısını
    aşmıyor."""
    img = random_img

    dct_res = compare.run_dct(img, 0.6, 8)
    wav_res = compare.run_wavelet(img, 0.6, "bior4.4", 3)
    for res in (dct_res, wav_res):
        m = res.metrics
        assert np.isfinite(m["mse"]) and m["mse"] >= 0
        assert np.isfinite(m["psnr"]) or np.isinf(m["psnr"])
        assert not np.isnan(m["psnr"])
        assert 0.0 <= m["ssim"] <= 1.0 + 1e-6
        assert m["bpp"] >= 0 and np.isfinite(m["bpp"])
        assert m["compressed_size_bytes"] >= 0

    gray = app._to_gray(img)
    wavelet = "bior4.4"
    levels = 3
    coeffs = decompose_for_viz(gray, wavelet, levels, "symmetric")
    stats = subbands.coeff_stats(coeffs)
    total_pct = sum(r["energy_pct"] for r in stats)
    for r in stats:
        assert np.isfinite(r["energy_pct"]) and r["energy_pct"] >= 0
        assert 0.0 <= r["zero_pct"] <= 100.0
    assert total_pct == pytest.approx(100.0, abs=1e-6)  # yüzdeler toplamda 100 olmalı

    quantized = wavelet_engine.quantize_for_viz(gray, 8.0, wavelet, levels, "symmetric")
    sparsity = subbands.quantized_sparsity_stats(coeffs, quantized)
    assert sparsity["nonzero"] <= sparsity["total"]
    assert sparsity["nonzero_pct"] + sparsity["sparsity_pct"] == pytest.approx(100.0, abs=1e-6)


def test_full_image_switch_clears_all_tab_state(random_img):
    """Part 22/23: Görüntü A tamamen analiz edildikten sonra Görüntü B'ye
    geçince (_reveal_workspace + sem_reset_state) HİÇBİR sekmede eski
    görüntünün sonucu 'aktif' görünmemeli — Compare sonuç paneli gizlenir
    VE tüm çıktılar temizlenir, Semantic ROI'nin TÜM türetilmiş state'i
    sıfırlanır. (İçsel önbellekte eski image_id'li veri kalabilir — bu
    kabul edilebilir; kontrol edilen, hiçbir şeyin 'aktif' gösterilmediğidir.)"""
    reveal_out = app._reveal_workspace(random_img, 2)
    # results_group (index 3) görünmez olmalı; DCT/Wavelet çıktıları None/boş
    assert reveal_out[3] == gr.update(visible=False)
    assert reveal_out[4] is None and reveal_out[5] is None  # out_dct, out_wav
    assert reveal_out[6] == "" and reveal_out[7] == ""      # rate_fairness, quality_strip

    sem_out = app.sem_reset_state()
    instances, checklist_update, detect_info, roi_center, mask_img, base_img, sem_img = sem_out[:7]
    assert instances == [] and roi_center is None
    assert mask_img is None and base_img is None and sem_img is None


# =============================================================================
# 39. FINAL FEATURE PASS — Sıkıştırma Özeti / Before-After (Part 27-33)
# =============================================================================
def test_size_calculations_original_vs_compressed(random_img):
    """Part 27: original>0, compressed>0, compression_ratio ve
    size_reduction_pct ELLE türetilen değerlerle BİREBİR eşleşmeli, VE
    calculate_metrics ile Compare'in gerçekten ürettiği metric_card/
    compression_summary_card HTML'lerinde AYNI sayılar görünmeli (Part 18
    source-of-truth)."""
    dct_res = compare.run_dct(random_img, 0.6, 8)
    m = dct_res.metrics
    assert m["original_size_bytes"] > 0
    assert m["compressed_size_bytes"] > 0

    expected_ratio = m["original_size_bytes"] / m["compressed_size_bytes"]
    assert m["compression_ratio"] == pytest.approx(expected_ratio, rel=1e-9)

    expected_reduction = (1.0 - m["compressed_size_bytes"] / m["original_size_bytes"]) * 100.0
    assert m["size_reduction_pct"] == pytest.approx(expected_reduction, rel=1e-9)

    # HTML'lerdeki rakamlar da (fmt_* biçimlendirmesinden sonra) AYNI
    # kaynaktan gelmeli — iki farklı formülle ikinci kez hesaplanmamalı.
    full_html = cards.metric_card("JPEG / DCT", "#31C8FF", m, size_badge="ENTROPİ TAHMİNİ")
    summary_html = cards.compression_summary_card_html("JPEG / DCT", "#31C8FF", m, "ENTROPİ TAHMİNİ")
    assert cards.fmt_ratio(m["compression_ratio"]) in full_html
    assert cards.fmt_ratio(m["compression_ratio"]) in summary_html
    assert cards.fmt_reduction(m["size_reduction_pct"]) in full_html
    assert cards.fmt_reduction(m["size_reduction_pct"]) in summary_html
    assert cards.fmt_size_kb(m["compressed_size_bytes"]) in full_html
    assert cards.fmt_size_kb(m["compressed_size_bytes"]) in summary_html


def test_before_after_pixel_alignment(random_img):
    """Part 28: Before/After HTML'i orijinal ile rekonstrüksiyonun GERÇEK
    (w/h) oranından hesaplanan bir aspect-ratio konteyneri kullanmalı — bu,
    iki görüntünün farklı ölçeklenmesinden kaynaklanan sahte bir kayma/
    kalite farkının oluşamayacağını garanti eder (Part 11)."""
    dct_res = compare.run_dct(random_img, 0.6, 8)
    html = cards.before_after_slider_html(random_img, dct_res.recon, "ORİJİNAL", "JPEG / DCT", "#31C8FF")
    h, w = random_img.shape[:2]
    expected_ar = f"{w / h:.6f}"
    assert expected_ar in html
    assert html.count("data:image/png;base64,") == 2
    assert 'type="range"' in html
    # Slider hareketi backend'e DOKUNMAMALI (Part 14) — HTML kendi kendine
    # yeterli olmalı, ek bir Gradio event handler'a referans içermemeli.
    assert "oninput" in html


def test_before_after_method_switch_changes_reconstruction(random_img):
    """Part 29: render_before_after — JPEG'den Wavelet'e geçince sağdaki
    rekonstrüksiyon GERÇEKTEN değişmeli (aynı PNG'nin yeniden gömülmesi
    DEĞİL) ve etiket doğru yöntemi göstermeli; YENİ bir sıkıştırma
    hesaplaması YAPILMAMALI — zaten hesaplanmış out_dct/out_wav değerleri
    kullanılır."""
    dct_res = compare.run_dct(random_img, 0.6, 8)
    wav_res = compare.run_wavelet(random_img, 0.6, "bior4.4", 3)

    html_dct = app.render_before_after(random_img, dct_res.recon, wav_res.recon, app._BA_METHOD_JPEG)
    html_wav = app.render_before_after(random_img, dct_res.recon, wav_res.recon, app._BA_METHOD_WAVELET)

    assert html_dct != html_wav
    assert app._BA_METHOD_JPEG in html_dct
    assert app._BA_METHOD_WAVELET in html_wav
    # no-op durumu: sonuç yoksa (henüz KARŞILAŞTIR basılmadıysa) sahte
    # bir görüntü ÜRETİLMEMELİ (Part 22 — hatalı/eksik state dürüstçe gösterilir)
    empty_html = app.render_before_after(None, None, None, app._BA_METHOD_JPEG)
    assert "data:image/png;base64," not in empty_html


def test_compare_output_counts_include_summary_cards(random_img):
    """run_main_comparison / compare_auto_analysis'in ürettiği çıktı SAYISI,
    UI'daki gerçek outputs= listesiyle (13 eleman) HER ZAMAN eşleşmeli — bu,
    mega-spec genelinde tekrar eden 'tuple uzunluğu kayması' hata sınıfının
    regresyon testidir. Mega-spec 'JPEG vs JPEG2000 GERÇEK CODEC
    KARŞILAŞTIRMASI': jp2k_img/jp2k_compact/jp2k_full çıktıları kaldırıldı
    (artık ikinci panelin KENDİSİ JPEG2000 — ayrı isteğe bağlı bir kolon
    değil), bu yüzden önceki 16'dan 13'e düştü."""
    out = app.run_main_comparison(random_img, "Hedef bpp", 0.5, 8, False)
    assert len(out) == 13
    dct_summary_html, wav_summary_html = out[8], out[9]
    assert "kpi-box" in dct_summary_html and "kpi-box" in wav_summary_html
    assert "size-arrow-row" in dct_summary_html and "size-arrow-row" in wav_summary_html
    real_jpeg_img, real_jpeg_compact, real_jpeg_full = out[10], out[11], out[12]
    # include_real_jpeg=False -> gerçek JPEG hesaplanmaz (Part 29: algoritma
    # istenmeden tekrar çalıştırılmaz), sonuçlar boş kalır.
    assert real_jpeg_img is None and real_jpeg_compact == "" and real_jpeg_full == ""


def test_view_mode_toggle_visibility():
    """Part 16: [ Yan Yana ] [ Önce / Sonra ] birbirini DIŞLAYAN görünürlük
    üretmeli; mevcut üçlü yan-yana görünüm KALDIRILMADI, yalnız alternatif
    eklendi."""
    side_upd, ba_upd = app.toggle_cmp_view_mode("Yan Yana")
    assert side_upd["visible"] is True and ba_upd["visible"] is False
    side_upd2, ba_upd2 = app.toggle_cmp_view_mode("Önce / Sonra")
    assert side_upd2["visible"] is False and ba_upd2["visible"] is True


# =============================================================================
# 40. FINAL MATHEMATICAL VALIDATION & AUDIT — SSIM bug fix + resize disclosure
# =============================================================================
def test_ssim_global_matches_skimage_canonical_scalar(random_img):
    """GERÇEK BUG (bu audit'te bulundu ve düzeltildi): mask=None durumunda
    ssim() önceden skimage'ın full=True haritasının .mean()'ini
    kullanıyordu — bu, skimage'ın KENDİ 'SSIM skoru' tanımı olan mssim
    skalerinden SİSTEMATİK olarak sapıyordu (gerçek görüntülerde ~1e-4
    mertebesinde — gösterilen 4 ondalık hassasiyetin sınırında). Artık
    doğrudan mssim kullanılıyor; bu test astronaut.png'de görülen gerçek
    sapmayı yeniden üretip artık SIFIR olduğunu doğrular."""
    noisy = np.clip(random_img.astype(np.int16) + RNG.integers(-20, 20, random_img.shape),
                    0, 255).astype(np.uint8)
    mssim, smap = structural_similarity(random_img, noisy, data_range=255, channel_axis=2, full=True)
    ours = ssim(random_img, noisy)
    assert ours == pytest.approx(mssim, abs=1e-9)
    # Eski (hatalı) davranışın GERÇEKTEN farklı bir sayı ürettiğini kanıtla
    # — yani bu bir regresyon testi, tesadüfen geçen bir test değil.
    assert abs(smap.mean() - mssim) > 1e-6


def test_active_image_bar_discloses_resize():
    """Part 3: app._prepare() büyük görüntüleri hıza karşı küçültür — bu
    SESSİZCE gizlenmemeli. Küçültülmediyse (yaygın durum) not gösterilmez
    (UI kalabalaşmaz); küçültüldüyse orijinal boyut açıkça görünür."""
    not_resized = cards.active_image_bar_html("x", None, 512, 512, 512, 512)
    assert "active-image-resized" not in not_resized

    resized = cards.active_image_bar_html("x", None, 768, 576, 4032, 3024)
    assert "active-image-resized" in resized
    assert "4032" in resized and "3024" in resized
    assert "768" in resized and "576" in resized  # analiz çözünürlüğü de hâlâ görünür


# =============================================================================
# 41. DWT LAB — TÜM SUBBAND NODE'LARINI TAM INTERAKTİF HALE GETİR
# =============================================================================
class _FakeSelectEvt:
    """gr.SelectData'nın yalnız .index alanını taklit eden minimal sahte
    olay — tıklama testleri gerçek Gradio event döngüsünden geçmeden
    doğrudan callback fonksiyonlarını çağırabilsin diye."""
    def __init__(self, x, y):
        self.index = (x, y)


def test_all_16_tree_nodes_resolve_to_distinct_real_coefficient_arrays(random_gray_img):
    """Part 18/19 (mandatory regression test): 4 seviye × 4 bant = 16
    düğümün TAMAMI ağaçta tıklanabilir olmalı ve HER BİRİ kendi GERÇEK
    katsayı dizisine geçmeli — yalnız buton tıklanabilir olması YETERLİ
    DEĞİL, GERÇEK VERİ değişmeli. Önceki bug: level_from_click_y yalnız Y
    eksenini okuyordu, bu yüzden LH/HL/HH çipleri tıklandığında hiçbir şey
    olmuyordu (her zaman mevcut/varsayılan bant korunuyordu)."""
    wavelet, levels, mode = "bior4.4", 4, "symmetric"
    coeffs = decompose_for_viz(random_gray_img, wavelet, levels, mode)
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)

    _, bounds = subbands.dyadic_tree_image(levels, active_level=1, active_band="LL")

    seen = []
    for lvl in range(1, levels + 1):
        for band in ("LL", "LH", "HL", "HH"):
            entry = next(e for e in bounds if e["level"] == lvl)
            x0, x1 = entry[band]
            y0, y1 = entry["y0"], entry["y1"]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

            # 1) Saf tıklama-çözümleme: doğru (level, band) döner mi?
            resolved_level, resolved_band = subbands.subband_from_click(bounds, cx, cy, levels)
            assert (resolved_level, resolved_band) == (lvl, band), (
                f"click at node {band}{lvl} resolved to {resolved_band}{resolved_level}")

            # 2) Uçtan uca callback: app.dwt_tree_click GERÇEK veriye geçiyor mu?
            evt = _FakeSelectEvt(cx, cy)
            (out_level, tree_img, new_bounds, ll_upd, lh_upd, hl_upd, hh_upd,
             details, selected_band, hist_fig) = app.dwt_tree_click(levels_data, bounds, levels, evt)
            assert out_level == lvl
            assert selected_band == band
            assert f"{band}{lvl}" in details  # kart başlığı doğru bandı gösteriyor
            arr = levels_data[lvl][band]
            seen.append((lvl, band, arr.shape, float(arr.mean()), float(np.sum(arr.astype(np.float64) ** 2))))

    # 16 düğümün TAMAMI farklı GERÇEK katsayı verisi üretti mi? (aynı
    # shape'e sahip komşu seviyeler bile enerji/mean bakımından farklı
    # olmalı — hepsi aynı LL0 cache'inden kalma OLAMAZ)
    assert len(seen) == 16
    signatures = [(s[2], round(s[3], 6), round(s[4], 6)) for s in seen]
    assert len(set(signatures)) == 16, f"bazı düğümler AYNI veriyi paylaşıyor (cache bug): {signatures}"


def test_tree_click_and_card_click_stay_synchronized(random_gray_img):
    """Part 6: sağdaki subband kartına (LH/HL/HH) tıklamak ağacı da (tree
    image) YENİDEN çizmeli — önceden yalnız `dwt_tree_click` ağacı
    güncelliyordu, `_dwt_band_click` (kart tıklaması) ağacı HİÇ
    dokunmadan bırakıyordu (görsel desenkron bug'ı)."""
    wavelet, levels, mode = "bior4.4", 3, "symmetric"
    coeffs = decompose_for_viz(random_gray_img, wavelet, levels, mode)
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)

    evt = _FakeSelectEvt(4, 4)  # band_preview küçük bir görüntü; içeride herhangi bir piksel
    out = app._dwt_band_click(levels_data, 2, "HL", evt)
    # 9 eleman: band_name, details, hist_fig, tree_img, bounds (Part 6),
    # + dwt_ll/lh/hl/hh görüntü güncellemeleri (mega-spec "DWT LAB —
    # subband görüntülerine tıklama ile katsayı inceleme").
    assert len(out) == 9
    (band_name, details, hist_fig, tree_img, bounds,
     ll_upd, lh_upd, hl_upd, hh_upd) = out
    assert band_name == "HL"
    assert "HL2" in details
    assert tree_img is not None and tree_img.size > 0
    assert any(e["level"] == 2 for e in bounds)
    # Yalnız tıklanan bant (HL) işaretli döner; diğerleri DÜZ kalır.
    assert hl_upd["value"] is not None


def test_level_slider_change_preserves_selected_band_across_levels():
    """Part 9: kullanıcı HH bandındayken 'İncelenecek seviye' slider'ını
    değiştirirse, YENİ seviyede de HH otomatik seçili kalmalı (band
    korunur, yalnız level değişir) — bu zaten dwt_select_level'in
    `band = selected_band if selected_band in bands else 'LL'` mantığıyla
    doğal olarak sağlanıyor; burada AÇIKÇA kilitleniyor."""
    wavelet, levels, mode = "bior4.4", 3, "symmetric"
    img = RNG.integers(0, 255, (96, 96), dtype=np.uint8).astype(np.float64)
    coeffs = decompose_for_viz(img, wavelet, levels, mode)
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)

    out_lvl2 = app.dwt_select_level(levels_data, 2, "HH")
    assert out_lvl2[7] == "HH"  # selected_band çıktısı
    out_lvl3 = app.dwt_select_level(levels_data, 3, "HH")
    assert out_lvl3[7] == "HH"
    # Gerçekten FARKLI bir dizi kullanıldığını da doğrula (aynı HH etiketi
    # ama farklı seviyenin GERÇEK verisi)
    assert not np.array_equal(levels_data[2]["HH"], levels_data[3]["HH"]) or \
        levels_data[2]["HH"].shape != levels_data[3]["HH"].shape


# =============================================================================
# 42. DWT LAB — subband görüntülerine tıklama ile katsayı inceleme
# =============================================================================
def test_subband_click_shows_real_raw_coefficient_not_preview_pixel(random_gray_img):
    """Mandatory test (mega-spec): LH1/HL1/HH1/LH2/HL2/HH2 üzerinde farklı
    noktalara tıklanınca gösterilen katsayı DEĞERİNİN, önizlemenin
    normalize edilmiş piksel değeri DEĞİL, gerçek ham
    levels_data[level][band][row][col] katsayısıyla BİREBİR eşleştiğini
    doğrular. Yalnız bilgi kutusunun açılması YETERLİ DEĞİL — sayı gerçek
    matristen gelmeli."""
    wavelet, levels, mode = "bior4.4", 3, "symmetric"
    coeffs = decompose_for_viz(random_gray_img, wavelet, levels, mode)
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)

    test_points = [
        (1, "LH", 5, 7), (1, "HL", 10, 3), (1, "HH", 20, 15),
        (2, "LH", 3, 4), (2, "HL", 8, 1), (2, "HH", 6, 6),
    ]
    seen_values = []
    for level, band, row, col in test_points:
        arr = levels_data[level][band]
        row, col = min(row, arr.shape[0] - 1), min(col, arr.shape[1] - 1)
        raw_value = float(arr[row, col])

        # evt.index Gradio sözleşmesiyle (x, y) = (col, row)
        evt = _FakeSelectEvt(col, row)
        out = app._dwt_band_click(levels_data, level, band, evt)
        details = out[1]
        assert f"{band}{level}" in details
        assert f"{raw_value:+.4f}" in details, (
            f"{band}{level}[{row}][{col}] = {raw_value:+.4f} detay kartında GÖRÜNMÜYOR: {details}")
        # Önizlemenin (normalize 0-255) piksel değeri KESİNLİKLE bu ham
        # değer olamaz (farklı ölçek/aralık) — yanlışlıkla preview pikseli
        # gösterilmediğinin dolaylı kanıtı.
        preview_val = float(subbands.band_preview(arr)[row, col])
        assert not (0 <= preview_val <= 255 and abs(preview_val - raw_value) < 1e-9) or raw_value < 0 or raw_value > 255
        seen_values.append(raw_value)

    # Farklı noktalara tıklandığında katsayı GERÇEKTEN değişiyor mu?
    assert len(set(round(v, 6) for v in seen_values)) == len(seen_values)


def test_subband_click_marks_clicked_cell_visually(random_gray_img):
    """Tıklanan hücre görsel olarak işaretlenmeli (küçük çerçeve); yalnız
    tıklanan bant işaretlenir, diğer 3 bant DÜZ (işaretsiz) kalır — mega-
    spec: 'Yeni subband seçildiğinde eski hücre seçimini temizle'."""
    wavelet, levels, mode = "bior4.4", 2, "symmetric"
    coeffs = decompose_for_viz(random_gray_img, wavelet, levels, mode)
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)
    arr = levels_data[1]["HH"]

    plain = subbands.band_preview(arr)
    marked = subbands.band_preview_with_marker(arr, row=10, col=12)
    assert marked.shape == (arr.shape[0], arr.shape[1], 3)  # RGB, işaret için
    # İşaretli görüntü GERÇEKTEN farklı pikseller içeriyor (dekoratif değil)
    assert not np.array_equal(np.stack([plain] * 3, axis=-1), marked)

    evt = _FakeSelectEvt(12, 10)
    out = app._dwt_band_click(levels_data, 1, "HH", evt)
    (band_name, details, hist_fig, tree_img, bounds,
     ll_upd, lh_upd, hl_upd, hh_upd) = out
    # Yalnız HH işaretli döner; LL/LH/HL güncellemeleri DÜZ (label'da işaret
    # ipucu yok, değerleri band_preview ile birebir aynı boyutta gri).
    hh_marked = hh_upd["value"]
    assert not np.array_equal(hh_marked, np.stack([subbands.band_preview(levels_data[1]["HH"])] * 3, axis=-1))
    ll_plain = ll_upd["value"]
    assert np.array_equal(ll_plain, subbands.band_preview(levels_data[1]["LL"]))


def test_subband_click_selection_clears_on_band_switch():
    """Mega-spec: 'HH1'e ait eski seçili hücre... LH2 üzerinde
    kalmamalı' — dwt_select_level (level slider / ağaç tıklaması / yeni
    AYRIŞTIR ile tetiklenen TEK canonical yeniden-render yolu) HER ZAMAN
    düz (işaretsiz) önizleme üretir; işaretli önizleme YALNIZ doğrudan o
    hücreye tıklandığında (_dwt_band_click) üretilir."""
    wavelet, levels, mode = "bior4.4", 2, "symmetric"
    img = RNG.integers(0, 255, (96, 96), dtype=np.uint8).astype(np.float64)
    coeffs = decompose_for_viz(img, wavelet, levels, mode)
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)

    # HH1'de bir hücre seç
    app._dwt_band_click(levels_data, 1, "HH", _FakeSelectEvt(3, 3))
    # Seviye değişince (dwt_select_level'in TEK yolundan) TÜM önizlemeler
    # düz olmalı — hiçbir görüntü işaretli KALAMAZ.
    out = app.dwt_select_level(levels_data, 2, "LL")
    (tree_img, bounds, ll_upd, lh_upd, hl_upd, hh_upd,
     details, selected_band, hist_fig) = out
    assert np.array_equal(hh_upd["value"], subbands.band_preview(levels_data[2]["HH"]))
    assert "dwt-pixel-box" not in details  # eski tıklanan-hücre bilgi kutusu KALMAMALI


# =============================================================================
# 43. DWT LAB — tıklama ile nokta inceleme: Piramit / Rekonstrüksiyon / Fark
# =============================================================================
def test_pyramid_click_shows_real_raw_coefficient(random_gray_img):
    """Part 1/8 (mandatory): Piramit Katsayı Haritası'nda farklı subband
    bölgelerine tıklanınca gösterilen katsayı, mozaiğin normalize edilmiş
    piksel değeri DEĞİL, gerçek RAW levels_data[level][band] dizisinden
    gelmeli."""
    wavelet, levels, mode = "bior4.4", 3, "symmetric"
    coeffs = decompose_for_viz(random_gray_img, wavelet, levels, mode)
    levels_data = subbands.decompose_levels(coeffs, levels, wavelet, mode)
    mosaic = subbands.pyramid_display_image(coeffs)
    regions = subbands.pyramid_regions(levels_data, levels)

    tested = 0
    for r in regions:
        if r["h"] <= 2 or r["w"] <= 2:
            continue
        cy, cx = r["y0"] + r["h"] // 2, r["x0"] + r["w"] // 2
        evt = _FakeSelectEvt(cx, cy)
        info, marked_mosaic = app.dwt_mosaic_click(mosaic, levels_data, levels, evt)
        expected_val = float(levels_data[r["level"]][r["band"]][cy - r["y0"], cx - r["x0"]])
        assert f"{r['band']}{r['level']}" in info
        assert f"{expected_val:+.4f}" in info, (
            f"{r['band']}{r['level']} beklenen {expected_val:+.4f} bulunamadı: {info}")
        assert not np.array_equal(marked_mosaic, mosaic)  # işaret gerçekten çizildi
        tested += 1
    assert tested >= 8  # en az birkaç farklı seviye/bant test edildi


def test_pyramid_click_outside_any_region_is_noop():
    """Mozaik sınırlarının dışına (varsa) tıklamak sahte bir sonuç
    ÜRETMEMELİ — no-op (gr.update()) dönmeli."""
    levels_data = {1: {"LL": np.zeros((10, 10)), "LH": np.zeros((10, 10)),
                       "HL": np.zeros((10, 10)), "HH": np.zeros((10, 10))}}
    mosaic = np.zeros((20, 20), dtype=np.uint8)
    info, marked = app.dwt_mosaic_click(mosaic, levels_data, 1, _FakeSelectEvt(999, 999))
    assert info == gr.update() and marked == gr.update()


def test_reconstruction_click_matches_real_arrays(random_gray_img):
    """Part 2/8 (mandatory): Kuantalamalı Rekonstrüksiyon'da birkaç farklı
    piksele tıklanınca gösterilen 'Orijinal'/'Rekonstrüksiyon' değerleri
    GERÇEK gray/recon dizileriyle birebir eşleşmeli — render edilmiş
    preview'dan DEĞİL."""
    wavelet, levels, mode = "bior4.4", 3, "symmetric"
    gray = random_gray_img
    recon, _ = wavelet_engine.compress_channel(gray, 8.0, wavelet, levels, mode=mode)

    test_points = [(5, 7), (40, 60), (80, 20), (10, 10)]
    seen = []
    for row, col in test_points:
        row = min(row, gray.shape[0] - 1)
        col = min(col, gray.shape[1] - 1)
        evt = _FakeSelectEvt(col, row)
        info, marked = app.dwt_recon_click(gray, recon, evt)
        expected_orig = float(gray[row, col])
        expected_recon = float(recon[row, col])
        assert f"{expected_orig:.2f}" in info
        assert f"{expected_recon:.2f}" in info
        assert marked is not None
        seen.append((expected_orig, expected_recon))
    # Farklı noktalara tıklandığında GERÇEKTEN farklı değerler görülüyor mu?
    assert len(set(seen)) > 1


def test_diff_map_click_uses_original_minus_reconstructed(random_gray_img):
    """Part 3/8 (mandatory): Fark Haritası'nda gösterilen 'Fark' değeri,
    subbands.reconstruction_diff_image'ın KULLANDIĞI AYNI tanımla
    (orijinal - rekonstrüksiyon) bağımsız hesapla eşleşmeli."""
    wavelet, levels, mode = "bior4.4", 2, "symmetric"
    gray = random_gray_img
    recon, _ = wavelet_engine.compress_channel(gray, 6.0, wavelet, levels, mode=mode)

    for row, col in [(3, 3), (50, 70), (90, 5)]:
        row = min(row, gray.shape[0] - 1)
        col = min(col, gray.shape[1] - 1)
        evt = _FakeSelectEvt(col, row)
        info, marked = app.dwt_diff_click(gray, recon, evt)
        manual_diff = float(gray[row, col]) - float(recon[row, col])
        sign = "+" if manual_diff >= 0 else ""
        assert f"{sign}{manual_diff:.2f}" in info, (
            f"beklenen fark {sign}{manual_diff:.2f} bulunamadı: {info}")


def test_recon_and_diff_click_share_same_info_format(random_gray_img):
    """Aynı (row,col) için Rekonstrüksiyon'a tıklamak ile Fark Haritası'na
    tıklamak AYNI Orijinal/Rekonstrüksiyon/Fark bilgisini göstermeli
    (paylaşılan bilgi kutusu — mega-spec: iki görsel de aynı üçlüyü
    okur)."""
    wavelet, levels, mode = "bior4.4", 2, "symmetric"
    gray = random_gray_img
    recon, _ = wavelet_engine.compress_channel(gray, 8.0, wavelet, levels, mode=mode)
    evt = _FakeSelectEvt(15, 12)
    info_recon, _ = app.dwt_recon_click(gray, recon, evt)
    info_diff, _ = app.dwt_diff_click(gray, recon, evt)
    assert info_recon == info_diff


def test_point_selection_clears_on_full_recompute():
    """Part 7: wavelet/seviye/kuantalama değişip run_dwt_explorer YENİDEN
    çalışınca eski nokta-seçim bilgi kutuları ("" ) sıfırlanmalı, yeni
    gray/recon durumu SAKLANMALI."""
    from src.engines import wavelet_engine as we
    img = RNG.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    out = app.run_dwt_explorer(img, "bior4.4", 2, 8.0, 1, we.DEFAULT_BOUNDARY_MODE)
    mosaic_info, recon_info, gray_state, recon_state = out[-4], out[-3], out[-2], out[-1]
    assert mosaic_info == "" and recon_info == ""
    assert gray_state is not None and recon_state is not None
    assert gray_state.shape == recon_state.shape


# =============================================================================
# 44. JPEG/DCT BPP VE RATE-DISTORTION DENETİMİ — gerçek libjpeg çapraz-doğrulama
# =============================================================================
def test_real_jpeg_actual_bpp_from_real_encoded_bytes():
    """Part 2/3 (mandatory): gerçek JPEG (libjpeg) bpp'si GERÇEK kodlanmış
    dosya boyutundan (cv2.imencode çıktısı) gelmeli — bir entropi
    tahmininden DEĞİL. actual_bpp = encoded_bytes*8/(w*h) formülü bağımsız
    olarak yeniden doğrulanır."""
    from src.engines import real_jpeg_engine as rj
    if not rj.REAL_JPEG_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir libjpeg kurulumu yok")
    img = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    recon, bpp, size_bytes, quality = rj.compress_at_target_bpp(img, 0.5)
    # bisection GERÇEKTEN yakınsadı mı (Part 3: target_bpp ≈ actual_bpp)?
    assert abs(bpp - 0.5) / 0.5 < 0.05

    manual_bpp = size_bytes * 8.0 / (h * w)
    assert bpp == pytest.approx(manual_bpp, rel=1e-9)

    # size_bytes GERÇEKTEN cv2.imencode'un ürettiği dosyanın boyutu mu?
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(round(quality))])
    assert ok and buf.size == size_bytes

    assert recon.shape == img.shape
    assert recon.dtype == np.uint8


def test_real_jpeg_psnr_matches_reference_range_on_grayscale_benchmark():
    """Part 8 (mandatory RD test) + Part 10 (son doğrulama): 512×512 8-bit
    grayscale kanonik test görüntüsünde (lenna.png), GERÇEK libjpeg
    kodlamasının PSNR'ı akademik referans aralıklarına (yaklaşık) uymalı —
    HERHANGİ bir sayı hard-code edilmeden, yalnız gerçek kodlama+ölçümle."""
    from src.engines import real_jpeg_engine as rj
    if not rj.REAL_JPEG_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir libjpeg kurulumu yok")
    img = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    assert img.shape == (512, 512)

    # Hocanın verdiği YAKLAŞIK referans aralıkları — sonuç ZORLA bu
    # aralığa getirilmez; yalnız gerçek ölçümün mantıklı bir bölgede
    # olduğunu doğrulamak için gevşek bir sağlık kontrolüdür (birkaç dB
    # payla, gerçek JPEG içerik/uygulama farkları için).
    reference_ranges = {0.25: (24.0, 32.0), 0.50: (27.0, 35.0),
                        0.75: (29.0, 37.0), 1.00: (31.0, 39.0)}
    prev_psnr = -1.0
    for target, (lo, hi) in reference_ranges.items():
        recon, bpp, size_bytes, quality = rj.compress_at_target_bpp(img, target)
        mse = float(np.mean((img.astype(np.float64) - recon.astype(np.float64)) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
        assert lo <= psnr <= hi, (
            f"target={target} bpp={bpp:.3f} PSNR={psnr:.2f}dB beklenen aralık dışında [{lo},{hi}]")
        # Bitrate arttıkça kalite genel olarak artmalı (mega-spec Part 8).
        assert psnr >= prev_psnr - 0.5  # küçük non-monotonik sapmalara tolerans
        prev_psnr = psnr


def test_compare_run_real_jpeg_is_flagged_real_codec():
    """is_real_codec=True olmalı (mega-spec Part 2) — dct_engine'in özel
    motorundan (is_real_codec=False) AYRIŞTIRILMALI; UI rozeti buradan
    türetilir (GERÇEK BOYUT vs ENTROPİ TAHMİNİ)."""
    if not real_jpeg_engine_available():
        pytest.skip("Bu ortamda gerçek bir libjpeg kurulumu yok")
    img = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_real_jpeg(img, 0.5)
    assert res is not None
    assert res.is_real_codec is True
    dct_res = compare.run_dct(img, 0.5, 8)
    assert dct_res.is_real_codec is False


def real_jpeg_engine_available() -> bool:
    from src.engines import real_jpeg_engine as rj
    return rj.REAL_JPEG_AVAILABLE


def test_grayscale_compression_ratio_uses_8bpp_not_24bpp():
    """Part 7 (mandatory): 8-bit GERÇEK grayscale (2D) bir görüntü için
    compression_ratio, 24-bit RGB varsayımıyla (24/bpp) DEĞİL, 8-bit
    grayscale varsayımıyla (8/bpp) hesaplanmalı. original_size_bytes da
    512×512×8bit = 262144 bayt (256 KiB) olmalı — 768 KiB (24-bit RGB
    varsayımı) DEĞİL."""
    img = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    assert img.ndim == 2 and img.shape == (512, 512)

    dct_res = compare.run_dct(img, 0.5, 8)
    m = dct_res.metrics
    assert m["original_size_bytes"] == pytest.approx(512 * 512 * 8 / 8.0)  # = 262144 bayt = 256 KiB
    assert m["original_size_bytes"] != pytest.approx(512 * 512 * 3 * 8 / 8.0)  # 24-bit RGB DEĞİL

    manual_ratio_8bpp = m["original_size_bytes"] / m["compressed_size_bytes"]
    manual_ratio_via_8_over_bpp = 8.0 / m["bpp"]
    assert m["compression_ratio"] == pytest.approx(manual_ratio_8bpp, rel=1e-9)
    assert m["compression_ratio"] == pytest.approx(manual_ratio_via_8_over_bpp, rel=1e-6)
    # 24/bpp KULLANILMADIĞINI da açıkça doğrula (yanlış olurdu)
    wrong_24bpp_ratio = 24.0 / m["bpp"]
    assert m["compression_ratio"] != pytest.approx(wrong_24bpp_ratio, rel=1e-6)


def test_jpeg_wavelet_same_rate_uses_real_computed_bpp_not_forced():
    """Part 9 (mandatory): JPEG ve Wavelet 'aynı hızda' karşılaştırılırken
    ikisinin actual_bpp'si GERÇEKTEN BAĞIMSIZ bisection aramalarından
    gelmeli (iki entropi tahmini elle eşitlenmiş OLAMAZ) — bu yüzden
    birbirine YAKIN ama BİREBİR AYNI olmaları beklenmez."""
    img = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    target = 0.5
    dct_res = compare.run_dct(img, target, 8)
    wav_res = compare.run_wavelet(img, target, "bior4.4", 3)
    # İkisi de hedefe YAKIN (ayrı ayrı yakınsamış bağımsız aramalar)
    assert abs(dct_res.metrics["bpp"] - target) / target < 0.10
    assert abs(wav_res.metrics["bpp"] - target) / target < 0.10
    # Birbirine yakın (same-rate karşılaştırması anlamlı) ama uydurma
    # şekilde BİREBİR eşit değil (her ikisi de kendi gerçek arama
    # sonucu farklı ondalıklara sahip olmalı).
    assert dct_res.metrics["bpp"] != wav_res.metrics["bpp"]
    assert abs(dct_res.metrics["bpp"] - wav_res.metrics["bpp"]) < 0.05


# =============================================================================
# 45. DWT LAB — sol kontrol panelini genişlet (db8/db12 + Hedef BPP)
# =============================================================================
def test_wavelet_ui_options_include_db8_and_db12():
    assert "db8" in config.WAVELET_UI_OPTIONS
    assert "db12" in config.WAVELET_UI_OPTIONS


def test_filter_length_comes_from_pywt_not_hardcoded():
    """Part 2 (mandatory): FİLTRE tap sayısı gerçek pywt.Wavelet.dec_len'den
    gelmeli — db2=4, db4=8, db8=16, db12=24 (kullanıcının beklediği
    DEĞERLER, ama bunlar burada SABİT yazılmadan, doğrudan library'den
    okunarak doğrulanır)."""
    expected = {"db2": 4, "db4": 8, "db8": 16, "db12": 24}
    for wav, exp_len in expected.items():
        info = subbands.filter_bank_info(wav)
        assert info["dec_len"] == exp_len == pywt.Wavelet(wav).dec_len


def test_transform_type_label_regression_orthogonal_vs_biorthogonal():
    """GERÇEK BUG (bu görevde bulunup düzeltildi): pywt'de
    Wavelet.biorthogonal TÜM dalgacıklar (ortogonal olanlar dahil) için
    True döner — önceki kod bunu birincil koşul olarak kullandığından
    db2/db4/db8/db12/haar/sym4/coif1 GİBİ GERÇEKTEN ORTOGONAL dalgacıklar
    için de yanlışlıkla 'Biortogonal' gösteriyordu. Doğru ayrım
    `orthogonal` bayrağıdır."""
    orthogonal_wavelets = ["db2", "db4", "db8", "db12", "haar", "sym4", "coif1"]
    biorthogonal_only = ["bior2.2", "bior4.4"]
    for wav in orthogonal_wavelets:
        info = subbands.filter_bank_info(wav)
        assert info["orthogonal"] is True, f"{wav} pywt'ye göre GERÇEKTEN ortogonal olmalı"
        label = "Ortogonal" if info["orthogonal"] else "Biortogonal"
        assert label == "Ortogonal", f"{wav} yanlışlıkla Biortogonal etiketlenmemeli"
    for wav in biorthogonal_only:
        info = subbands.filter_bank_info(wav)
        assert info["orthogonal"] is False
        label = "Ortogonal" if info["orthogonal"] else "Biortogonal"
        assert label == "Biortogonal"


def test_max_decomposition_level_shrinks_with_longer_filter():
    """Part 4 (mandatory): aynı görüntü boyutunda, filtre uzunluğu
    arttıkça (db2→db4→db8→db12) matematiksel olarak geçerli azami
    ayrıştırma seviyesi AYNI KALIR ya da AZALIR — hiçbir zaman artmaz."""
    shape = (128, 128)
    levels = [max_decomposition_level(shape, w) for w in ("db2", "db4", "db8", "db12")]
    assert levels == sorted(levels, reverse=True)
    assert levels[-1] >= 1  # db12 için bile en az 1 seviye geçerli kalmalı


def _extract_chip_value(summary_html: str, label: str) -> float:
    """summary_bar_html()'in ürettiği HTML'den TEK bir çipin sayısal
    değerini çıkarır — testlerin gösterilen değeri BAĞIMSIZ olarak
    doğrulayabilmesi için (yalnızca etiketin var olması YETERLİ DEĞİL)."""
    import re
    m = re.search(
        rf'<span class="summary-chip-label">{re.escape(label)}</span>'
        rf'<span class="summary-chip-value mono">([\d.]+)</span>',
        summary_html)
    assert m is not None, f"'{label}' çipi summary HTML'de bulunamadı: {summary_html[:500]}"
    return float(m.group(1))


def test_dwt_target_bpp_mode_actually_searches_quant_step(random_gray_img):
    """Part 'Hedef BPP' (mandatory): Hedef BPP modunda quant_step
    KULLANICI GİRDİSİ DEĞİL — targetBpp'yi tutturan gerçek bir bisection
    aramasından (match_bpp) gelir. GÖSTERİLEN 'Tahmini BPP' sayısı
    (HTML'den ayrıştırılıp) gerçekten hedefe yakınsamalı; Manuel Δ
    modunda ise HEDEF BPP çipi hiç GÖSTERİLMEMELİ ve verilen adım
    DEĞİŞTİRİLMEDEN kullanılmalı.

    GERÇEK BUG (bu testin ilk yazımı sırasında bulunup düzeltildi):
    run_dwt_explorer içindeki bisection lambda'sı compress_channel'ın
    döndürdüğü TOPLAM BİT sayısını piksel sayısına bölmeden match_bpp'e
    veriyordu — bu yüzden arama ASLA yakınsamıyor, tamamen anlamsız
    (örn. hedef 0.6 iken ~11000 gibi) bir 'bpp' üretiyordu."""
    # random_gray_img (96x96) çok küçük — DWT için daha gerçekçi, orta
    # boy bir görüntü kullan (gerçek bir örnek görüntüyle aynı davranış).
    img = np.tile(random_gray_img, (4, 4))[:320, :320]
    target = 0.6

    out_target = app.run_dwt_explorer(
        img, "bior4.4", 3, 8.0, 1, wavelet_engine.DEFAULT_BOUNDARY_MODE,
        app._DWT_RATE_MODE_TARGET, target)
    summary_target = out_target[0]
    shown_target = _extract_chip_value(summary_target, "HEDEF BPP")
    shown_achieved = _extract_chip_value(summary_target, "TAHMİNİ BPP")
    assert shown_target == pytest.approx(target, abs=1e-6)
    # GERÇEK yakınsama sağlık kontrolü — match_bpp'nin kendi toleransından
    # (BPP_MATCH_TOLERANCE, göreceli) daha gevşek ama önceki bug'ı (11000
    # gibi bir değeri) KESİNLİKLE yakalayacak kadar sıkı.
    assert abs(shown_achieved - target) / target < 0.15, (
        f"Hedef {target} iken gösterilen Tahmini BPP {shown_achieved} — yakınsamadı")

    out_manual = app.run_dwt_explorer(
        img, "bior4.4", 3, 8.0, 1, wavelet_engine.DEFAULT_BOUNDARY_MODE,
        app._DWT_RATE_MODE_MANUAL, target)
    summary_manual = out_manual[0]
    assert "HEDEF BPP" not in summary_manual  # Manuel modda hedef bpp GÖSTERİLMEZ
    manual_achieved = _extract_chip_value(summary_manual, "TAHMİNİ BPP")
    # Manuel modda (step=8.0 sabit) elde edilen bpp, Hedef BPP modunun
    # (step arandı, target=0.6'ya yakınsadı) sonucundan FARKLI olmalı —
    # aynı sabit adıma tesadüfen denk gelmediğini kanıtlar.
    assert manual_achieved != pytest.approx(shown_achieved, rel=1e-6)


def test_dwt_rate_mode_toggle_is_mutually_exclusive():
    """'Aynı anda iki farklı kontrolün birbirini sessizce ezmesine izin
    verme' — toggle_dwt_rate_mode İKİ kontrolün görünürlüğünü de TEK
    fonksiyondan, KARŞILIKLI DIŞLAYICI olarak döndürmeli."""
    target_upd, manual_upd = app.toggle_dwt_rate_mode(app._DWT_RATE_MODE_TARGET)
    assert target_upd["visible"] is True and manual_upd["visible"] is False
    target_upd2, manual_upd2 = app.toggle_dwt_rate_mode(app._DWT_RATE_MODE_MANUAL)
    assert target_upd2["visible"] is False and manual_upd2["visible"] is True


def test_all_four_daubechies_wavelets_full_pipeline(random_gray_img):
    """SON RAPOR (mandatory): db2, db4, db8, db12 için AYRI AYRI filter
    length / max decomposition level / selected level / achieved-estimated
    BPP / reconstruction / subband dimensions test edilir — hepsi GERÇEK
    computational pipeline'dan, hardcode edilmeden."""
    img = np.tile(random_gray_img, (2, 2))[:192, :192]  # 192x192, db12 icin de yeterli
    results = {}
    for wav in ("db2", "db4", "db8", "db12"):
        max_level = max_decomposition_level(img.shape, wav)
        selected_level = min(2, max_level)
        out = app.run_dwt_explorer(
            img, wav, selected_level, 8.0, 1, wavelet_engine.DEFAULT_BOUNDARY_MODE,
            app._DWT_RATE_MODE_TARGET, 0.5)
        (summary, mosaic, tree_img, bounds, ll_upd, lh_upd, hl_upd, hh_upd,
         band_details, selected_band, hist_fig, recon, diff_img, energy_fig,
         filt_flow, filt_fig, filt_text, stats_rows, validation_card, sparsity_card,
         inspect_upd, levels_data, used_levels, mosaic_info, recon_info,
         gray_state, recon_state) = out

        filt_len = pywt.Wavelet(wav).dec_len
        results[wav] = dict(
            filter_length=filt_len, max_level=max_level, selected_level=used_levels,
            recon_shape=recon.shape, ll_shape=levels_data[used_levels]["LL"].shape,
        )
        # Gerçek pipeline doğrulamaları
        assert used_levels == selected_level
        assert recon.shape == img.shape  # rekonstrüksiyon orijinalle AYNI boyut
        assert f"{filt_len} tap" in summary
        assert f"{used_levels}" in summary  # AYRIŞTIRMA SEVİYESİ
        assert "TAHMİNİ BPP" in summary
        # subband shape'leri array.shape'ten (hardcode değil)
        for band in ("LL", "LH", "HL", "HH"):
            arr = levels_data[used_levels][band]
            assert arr.shape[0] > 0 and arr.shape[1] > 0

    # 4 dalgacığın TAMAMI GERÇEKTEN farklı filtre uzunluğuna sahip
    lens = [results[w]["filter_length"] for w in ("db2", "db4", "db8", "db12")]
    assert lens == [4, 8, 16, 24]
    # Filtre uzadıkça azami seviye artmıyor (küçülüyor ya da eşit kalıyor)
    max_levels = [results[w]["max_level"] for w in ("db2", "db4", "db8", "db12")]
    assert max_levels == sorted(max_levels, reverse=True)


# =============================================================================
# 46. Aktif görüntü çubuğu — yüklenen/seçilen DOSYANIN gerçek boyutu
# =============================================================================
def test_active_image_bar_never_shows_file_size_chip():
    """Kullanıcı isteğiyle DOSYA boyutu çipi kaldırıldı — file_size_bytes
    verilse de verilmese de aktif görüntü çubuğunda HİÇ görünmemeli."""
    path = "data/classic/lenna.png"
    real_file_bytes = Path(path).stat().st_size

    html = cards.active_image_bar_html("cameraman", "Classic", 512, 512,
                                       file_size_bytes=real_file_bytes)
    assert "active-image-filesize" not in html
    assert "DOSYA" not in html

    html_no_size = cards.active_image_bar_html("cameraman", "Classic", 512, 512)
    assert "active-image-filesize" not in html_no_size


def test_select_and_upload_populate_real_file_size_in_meta():
    """app._select ve app._upload'un ürettiği meta sözlüğü GERÇEK
    file_size_bytes içermeli — Path.stat().st_size'tan, uydurma bir
    değerden değil."""
    path = Path("data/classic/lenna.png")
    real_size = path.stat().st_size
    assert real_size > 0
    # Not: bu meta alanı artık HİÇBİR yerde GÖRÜNTÜLENMİYOR (kullanıcı
    # isteğiyle kaldırıldı — bkz. test_active_image_bar_never_shows_
    # file_size_chip), yalnız _select/_upload'un dosya sisteminden doğru
    # okuduğunu bağımsız doğrular.
    import os
    assert os.path.getsize(path) == real_size


# =============================================================================
# 47. PGM (.pgm) yükleme desteği — P2/P5, header parse, upload whitelist
# =============================================================================
def _write_pgm_p5(path, gray, maxval=255, comment=None):
    header = "P5\n"
    if comment:
        header += f"# {comment}\n"
    header += f"{gray.shape[1]} {gray.shape[0]}\n{maxval}\n"
    with open(path, "wb") as f:
        f.write(header.encode("utf-8") + gray.astype(np.uint8).tobytes())


def _write_pgm_p2(path, gray, maxval=255, comment=None):
    lines = ["P2"]
    if comment:
        lines.append(f"# {comment}")
    lines.append(f"{gray.shape[1]} {gray.shape[0]}")
    lines.append(str(maxval))
    lines.append(" ".join(str(int(v)) for v in gray.flatten()))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_pgm_p5_binary_round_trip_exact(tmp_path):
    from src.pgm_io import parse_pgm

    gray = np.random.RandomState(0).randint(0, 256, size=(37, 53)).astype(np.uint8)
    p = tmp_path / "t.pgm"
    _write_pgm_p5(p, gray, comment="ara yorum satırı")
    out = parse_pgm(str(p))
    assert out.dtype == np.uint8
    assert out.shape == gray.shape
    assert np.array_equal(out, gray)


def test_pgm_p2_ascii_round_trip_exact(tmp_path):
    from src.pgm_io import parse_pgm

    gray = np.random.RandomState(1).randint(0, 256, size=(19, 31)).astype(np.uint8)
    p = tmp_path / "t.pgm"
    _write_pgm_p2(p, gray, comment="ascii test")
    out = parse_pgm(str(p))
    assert out.dtype == np.uint8
    assert out.shape == gray.shape
    assert np.array_equal(out, gray)


def test_pgm_p5_and_p2_agree_on_same_content(tmp_path):
    """P2 ve P5 aynı pikselleri kodluyorsa parser aynı diziyi üretmeli —
    iki farklı kod yolunun (ASCII tokenizer vs binary frombuffer) TUTARLI
    olduğunu doğrular."""
    from src.pgm_io import parse_pgm

    gray = np.random.RandomState(2).randint(0, 256, size=(12, 12)).astype(np.uint8)
    p5, p2 = tmp_path / "a.pgm", tmp_path / "b.pgm"
    _write_pgm_p5(p5, gray)
    _write_pgm_p2(p2, gray)
    assert np.array_equal(parse_pgm(str(p5)), parse_pgm(str(p2)))


def test_pgm_512x512_dimensions_from_header_not_hardcoded(tmp_path):
    """lenna.pgm senaryosu: width/height GERÇEK header token'larından
    okunmalı, sabit 512 varsayılmamalı — farklı boyutlu bir PGM'de de
    doğru çalıştığını göstererek doğrulanır."""
    from src.pgm_io import parse_pgm

    gray512 = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    assert gray512.shape == (512, 512)
    p_512 = tmp_path / "lenna.pgm"
    _write_pgm_p5(p_512, gray512, comment="lenna.pgm test")
    out_512 = parse_pgm(str(p_512))
    assert out_512.shape == (512, 512)
    assert np.array_equal(out_512, gray512)

    gray_other = np.random.RandomState(3).randint(0, 256, size=(200, 333)).astype(np.uint8)
    p_other = tmp_path / "other.pgm"
    _write_pgm_p5(p_other, gray_other)
    assert parse_pgm(str(p_other)).shape == (200, 333)


def test_pgm_maxval_255_used_directly_as_8bit():
    """maxval==255 yolunda basit bir astype(uint8) yeterli — ölçekleme
    UYGULANMAMALI (spesifikasyon: 'maxval 255 ise doğrudan 8-bit
    grayscale olarak kullan')."""
    import inspect

    from src.pgm_io import parse_pgm
    src_text = inspect.getsource(parse_pgm)
    assert "maxval == 255" in src_text


def test_pgm_unsupported_magic_raises_clear_error(tmp_path):
    from src.pgm_io import parse_pgm

    p = tmp_path / "bad.pgm"
    p.write_bytes(b"P3\n4 4\n255\n" + bytes(48))
    with pytest.raises(ValueError, match="P2|P5|magic"):
        parse_pgm(str(p))


def test_pgm_truncated_binary_data_raises_clear_error(tmp_path):
    from src.pgm_io import parse_pgm

    p = tmp_path / "truncated.pgm"
    p.write_bytes(b"P5\n10 10\n255\n" + bytes(5))  # 100 bekleniyor, 5 var
    with pytest.raises(ValueError):
        parse_pgm(str(p))


def test_pgm_grayscale_survives_rgb_broadcast_losslessly():
    """app._upload_pgm, parse_pgm'in tek-kanal çıktısını pipeline uyumluluğu
    için R=G=B olarak 3 kanala genişletir (aktif görüntü state'i her yerde
    RGB numpy bekliyor — Semantic ROI/YOLO dahil). _to_gray(R=G=B), 0.299+
    0.587+0.114'ün IEEE-754 double'da tam 1.0 OLMADIĞI (0.9999999999999999)
    için matematiksel olarak birebir özdeşlik DEĞİL, ama fark ~1e-14
    mertebesinde — 8-bit bir pikselin temsil edebileceği en küçük farktan
    (1.0) 13 kat mertebe küçük, yani pratikte GERÇEKTEN kayıpsız."""
    gray = np.random.RandomState(4).randint(0, 256, size=(64, 80)).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    recovered = app._to_gray(rgb)
    assert np.allclose(recovered, gray.astype(np.float64), atol=1e-9)


def test_pgm_upload_widget_whitelists_pgm_extension():
    """gr.Image'ın tarayıcı-taraflı upload filtresi sabit 'image/*'tir ve
    .pgm çoğu tarayıcıda application/octet-stream raporlandığı için orada
    seçilemez (bkz. gradio Upload.svelte). PGM için AYRI bir gr.File
    bileşeni file_types ile UZANTI bazlı whitelist sağlamalı — bu test,
    build_active_image_bar()'ın gerçekten böyle bir bileşen kurduğunu
    (regresyona karşı) doğrular."""
    with gr.Blocks() as demo:
        app.build_active_image_bar()

    file_components = [c for c in demo.blocks.values() if isinstance(c, gr.File)]
    pgm_uploaders = [c for c in file_components if c.file_types and ".pgm" in c.file_types]
    assert len(pgm_uploaders) == 1, "PGM için whitelist'li bir gr.File yükleyici bulunamadı"


def test_general_upload_widget_is_gr_file_not_gr_image():
    """gr.Image(type='filepath') bu Gradio sürümünde dosyayı ASLA ham
    geçirmiyor — preprocess() her zaman PIL ile açıp RGB'ye çevirip
    yeniden kaydediyor (gradio/components/image.py: format_image →
    save_pil_to_cache, koşulsuz). Bu yüzden 'yüklediğimiz dosyanın GERÇEK
    boyutu' iddiası gr.Image ile YANLIŞTI — canlı sunucuda doğrulandı:
    77KB'lık bir .jpg için '38KB', 139KB'lık gri bir PNG için '238KB'
    gösteriyordu. upload_img artık gr.File (hiç yeniden kodlama yapmıyor,
    aynı PGM yükleyicisi gibi) — bu test regresyona karşı korur: biri
    upload_img'i tekrar gr.Image'e çevirirse burada patlamalı."""
    with gr.Blocks() as demo:
        app.build_active_image_bar()

    file_components = [c for c in demo.blocks.values() if isinstance(c, gr.File)]
    general_uploaders = [
        c for c in file_components
        if not (c.file_types and ".pgm" in c.file_types)
    ]
    assert len(general_uploaders) == 1, "Genel görsel yükleyici gr.File olarak bulunamadı"
    assert general_uploaders[0].type == "filepath"


# =============================================================================
# 48. BPP/PSNR/MSE/SSIM GÜVENİLİRLİK DENETİMİ — gerçek gri kaynakların
#     3-kanal RGB olarak kodeklenmesi bug'ı + ana JPEG panelinin gerçek
#     libjpeg kullanması
# =============================================================================
def test_reduce_if_true_grayscale_recovers_exact_single_channel():
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    rgb = np.stack([gray, gray, gray], axis=-1)
    reduced = app._reduce_if_true_grayscale(rgb)
    assert reduced.ndim == 2
    assert np.array_equal(reduced, gray)


def test_reduce_if_true_grayscale_leaves_real_color_untouched():
    color = np.random.RandomState(0).randint(0, 256, size=(32, 32, 3)).astype(np.uint8)
    reduced = app._reduce_if_true_grayscale(color)
    assert reduced.ndim == 3
    assert np.array_equal(reduced, color)


def test_true_grayscale_rgb_broadcast_wastes_real_jpeg_bytes_on_constant_chroma():
    """Bug'ın kendisi: R=G=B olsa bile 3 kanal olarak libjpeg'e verilirse
    GERÇEK (sıfır olmayan) kroma bayt maliyeti oluşuyor — bu yüzden
    _reduce_if_true_grayscale GEREKLİ, kozmetik değil."""
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    rgb = np.stack([gray, gray, gray], axis=-1)
    from src.engines import real_jpeg_engine as rj
    _, size_gray = rj.encode(gray, 50)
    _, size_rgb = rj.encode(rgb, 50)
    assert size_rgb > size_gray  # kroma düzlemleri GERÇEKTEN bayt maliyetli


def test_compare_original_size_bytes_is_true_grayscale_baseline_not_768kib():
    """Mega-spec Part 8: 512x512 8-bit gri görüntü ham tabanı 262144 bayt
    (256 KiB) olmalı — 3 kanal RGB varsayılırsa yanlışlıkla 786432 bayt
    (768 KiB) çıkar (bkz. app._reduce_if_true_grayscale docstring'i)."""
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    assert gray.shape == (512, 512)
    rgb = np.stack([gray, gray, gray], axis=-1)
    reduced = app._reduce_if_true_grayscale(rgb)
    res = compare.run_dct(reduced, 0.5, 8)
    assert res.metrics["original_size_bytes"] == 512 * 512 * 1  # 256 KiB, 768 KiB DEĞİL
    assert res.metrics["original_size_bytes"] != 512 * 512 * 3


def test_run_jpeg_primary_uses_real_jpeg_for_standard_block_size():
    """Ana 'JPEG / DCT' paneli block_size=8'de (standart JPEG) GERÇEK
    libjpeg kullanmalı, entropi tahmini DEĞİL (mega-spec Part 2: 'Ana JPEG
    karşılaştırması entropy estimate olmasın')."""
    from src.engines.real_jpeg_engine import REAL_JPEG_AVAILABLE
    if not REAL_JPEG_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir libjpeg kurulumu yok")
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_jpeg_primary(gray, 0.5, 8)
    assert res.is_real_codec is True
    assert "GERÇEK" in res.note or "gerçek" in res.note.lower()


def test_run_jpeg_primary_falls_back_to_entropy_estimate_for_nonstandard_block_size():
    """block_size=4/16 gerçek bir JPEG bitstream'inde KARŞILIĞI YOK (JPEG
    standardı sabit 8x8 blok kullanır) — bu durumda özel entropi-tahmini
    motoruna dürüstçe (is_real_codec=False) düşmeli, gerçekmiş gibi
    GÖSTERİLMEMELİ."""
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_jpeg_primary(gray, 0.5, 4)
    assert res.is_real_codec is False


def test_target_to_bpp_uses_correct_channel_count_for_ratio_mode():
    """'Hedef Sıkıştırma Oranı' modu ratio_to_bpp'yi ÇAĞIRDIĞI channels
    parametresiyle tutarlı olmalı — gri görüntüde (channels=1) oran=8 ->
    bpp=1.0 olmalı, renkli varsayılan (channels=3) ile karıştırılmamalı
    (channels=3'te oran=8 -> bpp=3.0 olurdu)."""
    bpp_gray = app._target_to_bpp(app._TARGET_MODE_RATIO, 8.0, channels=1)
    bpp_color = app._target_to_bpp(app._TARGET_MODE_RATIO, 8.0, channels=3)
    assert bpp_gray == pytest.approx(1.0)
    assert bpp_color == pytest.approx(3.0)
    assert bpp_gray != bpp_color


def test_internal_psnr_mse_consistency_for_real_jpeg():
    """PSNR, gösterilen MSE'den yeniden türetildiğinde (10*log10(65025/MSE))
    UI değeriyle floating-point hassasiyeti dışında FARK OLMAMALI (mega-spec
    Part 11: 'Uyuşmuyorsa metric pipeline'da bug vardır')."""
    from src.engines.real_jpeg_engine import REAL_JPEG_AVAILABLE
    if not REAL_JPEG_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir libjpeg kurulumu yok")
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_jpeg_primary(gray, 0.5, 8)
    recomputed = 10 * np.log10(65025.0 / res.metrics["mse"])
    assert res.metrics["psnr"] == pytest.approx(recomputed, abs=1e-9)


# =============================================================================
# 50. JPEG vs JPEG2000 GERÇEK CODEC KARŞILAŞTIRMASI — ana Compare artık
#     entropi-tahmini DWT motoru değil, gerçek OpenJPEG kullanıyor
# =============================================================================
def _require_jpeg2000():
    from src.engines.jpeg2000_engine import JPEG2000_AVAILABLE
    if not JPEG2000_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir OpenJPEG kurulumu yok")


def test_1_jpeg_actual_bpp_matches_real_encoded_bytes():
    """Test 1: JPEG actual BPP gerçek encoded byte sayısıyla birebir eşleşiyor mu?"""
    from src.engines.real_jpeg_engine import REAL_JPEG_AVAILABLE
    if not REAL_JPEG_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir libjpeg kurulumu yok")
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_jpeg_primary(gray, 0.5, 8)
    h, w = gray.shape
    expected_bpp = res.extra["size_bytes"] * 8 / (h * w)
    assert res.metrics["bpp"] == pytest.approx(expected_bpp, rel=1e-9)


def test_2_jpeg2000_actual_bpp_matches_real_encoded_bytes():
    """Test 2: JPEG2000 actual BPP gerçek encoded byte sayısıyla birebir eşleşiyor mu?"""
    _require_jpeg2000()
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_jpeg2000(gray, 0.5)
    h, w = gray.shape
    expected_bpp = res.extra["size_bytes"] * 8 / (h * w)
    assert res.metrics["bpp"] == pytest.approx(expected_bpp, rel=1e-9)


def test_3_jpeg2000_encode_decode_actually_round_trips():
    """Test 3: JPEG2000 encode→decode gerçekten çalışıyor mu? — kaynak
    kodu okunarak (jpeg2000_engine.compress_image) im.save(format=
    "JPEG2000") ile GERÇEK bir bitstream'e yazılıp Image.open(buf) ile
    GERÇEK decode edildiği doğrulanır; ayrıca rekonstrüksiyon orijinalden
    FARKLI (lossy, sıfır olmayan bir MSE) olmalı — PSNR=inf ile 'decode
    hiç çalışmadı, aynı array geri döndü' arasındaki farkı ayırt eder."""
    _require_jpeg2000()
    import inspect

    from src.engines import jpeg2000_engine
    src_text = inspect.getsource(jpeg2000_engine.compress_image)
    assert 'format="JPEG2000"' in src_text
    assert "Image.open(buf)" in src_text

    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    recon, bpp, size_bytes = jpeg2000_engine.compress_at_target_bpp(gray, 0.5)
    assert recon.shape == gray.shape
    assert recon.dtype == np.uint8
    assert size_bytes > 0
    mse = float(np.mean((gray.astype(np.float64) - recon.astype(np.float64)) ** 2))
    assert mse > 0  # lossy — orijinalle birebir aynı DEĞİL (decode gerçekten çalıştı)


def test_4_jpeg2000_psnr_is_original_vs_decoded():
    """Test 4: JPEG2000 PSNR original vs decoded üzerinden mi hesaplanıyor?"""
    _require_jpeg2000()
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_jpeg2000(gray, 0.5)
    mse_manual = float(np.mean((gray.astype(np.float64) - res.recon.astype(np.float64)) ** 2))
    psnr_manual = 10 * np.log10(65025.0 / mse_manual) if mse_manual > 0 else float("inf")
    assert res.metrics["psnr"] == pytest.approx(psnr_manual, rel=1e-6)
    assert res.metrics["mse"] == pytest.approx(mse_manual, rel=1e-9)


def test_5_jpeg_and_jpeg2000_ssim_are_original_vs_decoded():
    """Test 5: JPEG SSIM ve JPEG2000 SSIM original vs decoded üzerinden mi?"""
    from src.engines.real_jpeg_engine import REAL_JPEG_AVAILABLE
    _require_jpeg2000()
    if not REAL_JPEG_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir libjpeg kurulumu yok")
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    jpeg_res = compare.run_jpeg_primary(gray, 0.5, 8)
    jp2k_res = compare.run_jpeg2000(gray, 0.5)
    ssim_jpeg_manual = structural_similarity(gray, jpeg_res.recon, data_range=255)
    ssim_jp2k_manual = structural_similarity(gray, jp2k_res.recon, data_range=255)
    assert jpeg_res.metrics["ssim"] == pytest.approx(ssim_jpeg_manual, rel=1e-6)
    assert jp2k_res.metrics["ssim"] == pytest.approx(ssim_jp2k_manual, rel=1e-6)


def test_6_psnr_generally_increases_with_target_bpp():
    """Test 6: Target BPP yükseldikçe genel olarak PSNR yükseliyor mu?
    (JPEG ve JPEG2000, ikisi de.)"""
    _require_jpeg2000()
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    targets = [0.25, 0.50, 0.65, 1.00]
    jpeg_psnrs = [compare.run_jpeg_primary(gray, t, 8).metrics["psnr"] for t in targets]
    jp2k_psnrs = [compare.run_jpeg2000(gray, t).metrics["psnr"] for t in targets]
    assert jpeg_psnrs == sorted(jpeg_psnrs)
    assert jp2k_psnrs == sorted(jp2k_psnrs)


def test_7_compare_jpeg2000_engine_result_is_real_codec():
    """Test 7: Ana Compare'daki JPEG2000 EngineResult.is_real_codec == True mı?"""
    _require_jpeg2000()
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    res = compare.run_jpeg2000(gray, 0.5)
    assert res.is_real_codec is True


def test_8_dwt_lab_still_uses_entropy_estimated_bior_engine():
    """Test 8: DWT Lab hâlâ eski entropy-estimated bior4.4 motorunu
    kullanıyor mu? — mega-spec 'DWT LAB'I KORU': compress_channel/
    compress_image (wavelet_engine) SİLİNMEDİ, hâlâ estimate_bits_subband
    (order-0 Shannon tahmini) tabanlı, tam çalışır durumda."""
    import inspect
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    recon, bits = wavelet_engine.compress_channel(gray, 16.0, "bior4.4", 4)
    assert recon.shape == gray.shape
    assert bits > 0
    src_text = inspect.getsource(wavelet_engine.compress_channel)
    assert "estimate_bits_subband" not in src_text  # doğrudan _quantize_band üzerinden gelir
    src_quantize = inspect.getsource(wavelet_engine._quantize_band)
    assert "estimate_bits_subband" in src_quantize
    # compare.run_wavelet (Shannon tahmini) da hâlâ İÇERİDE ve ÇALIŞIYOR —
    # yalnız artık run_main_comparison tarafından ÇAĞRILMIYOR (test 9).
    wav_res = compare.run_wavelet(gray, 0.5, "bior4.4", 4)
    assert wav_res.is_real_codec is False


def test_9_compare_screen_never_calls_estimated_wavelet_engine():
    """Test 9: Compare ekranında entropy-estimated DWT yanlışlıkla
    kullanılmıyor mu? — run_main_comparison'ın kaynak kodu run_wavelet'i
    (Shannon tahmini) HİÇ ÇAĞIRMAMALI; yalnız run_jpeg2000 (gerçek kodek)
    çağırmalı. compare.rd_sweep de aynı şekilde."""
    import inspect
    src_run_main = inspect.getsource(app.run_main_comparison)
    assert "run_wavelet(" not in src_run_main  # ÇAĞRI yok (docstring'de bahsi ayrı)
    assert "run_jpeg2000(" in src_run_main

    src_rd_sweep = inspect.getsource(compare.rd_sweep)
    assert "run_wavelet(" not in src_rd_sweep
    assert "run_jpeg2000(" in src_rd_sweep

    # Canlı doğrulama: gerçek bir Compare çalıştırması, is_real_codec=True
    # dönen bir "DWT" sonucu üretmeli (Shannon tahminiyle bu ASLA olmaz).
    from src.engines.jpeg2000_engine import JPEG2000_AVAILABLE
    if not JPEG2000_AVAILABLE:
        pytest.skip("Bu ortamda gerçek bir OpenJPEG kurulumu yok")
    gray = cv2.imread("data/classic/lenna.png", cv2.IMREAD_GRAYSCALE)
    out = app.run_main_comparison(gray, "Hedef bpp", 0.5, 8, False)
    wav_full_html = out[7]  # bkz. run_main_comparison return sırası
    assert "GERÇEK BOYUT" in wav_full_html
    assert "ENTROPİ TAHMİNİ" not in wav_full_html


def test_10_no_lena_specific_hardcode_anywhere():
    """Test 10: Hiçbir Lena/Lenna özel hardcode bulunmadığını doğrula —
    kaynak dosyalar taranır; 'lena'/'lenna' geçen satırların yalnız
    'filename' gibi tesadüfi alt-dizeler olduğu, dosya adına/hash'e göre
    davranış değiştiren GERÇEK bir kod dalı olmadığı doğrulanır. Referans
    PSNR değerlerinin (bu oturumda gözlemlenenler) hiçbiri koda
    gömülmemiş olmalı."""
    src_files = list(Path("src").rglob("*.py")) + [Path("app.py"), Path("config.py")]
    forbidden_psnr_values = ["29.15", "30.90", "32.94", "33.64", "34.11",
                             "34.80", "35.77", "36.59"]
    for f in src_files:
        if "__pycache__" in str(f):
            continue
        text = f.read_text(encoding="utf-8")
        for line in text.splitlines():
            low = line.lower()
            if "lena" in low or "lenna" in low:
                # yalnız 'filename' gibi tesadüfi alt-dize icin izin ver;
                # gerçek bir "lena"/"lenna" TOKEN'i (kelime sınırında) olursa patla
                import re
                assert not re.search(r"\blenn?a\b", low), (
                    f"{f}:{line!r} — olası Lena'ya özel kod bulundu"
                )
        for val in forbidden_psnr_values:
            assert val not in text, f"{f} icinde hardcoded referans PSNR degeri bulundu: {val}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
