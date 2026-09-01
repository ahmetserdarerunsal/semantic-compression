# -*- coding: utf-8 -*-
"""Merkezi konfigürasyon: tüm ayarlanabilir parametreler burada.

Deneyleri değiştirmek için sadece bu dosyayı düzenlemek yeterli olmalı.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Görüntü tipleri -> data/ altındaki alt klasörler (deney betikleri
# — experiments/*.py — bu listeyi kullanır; her biri data/ altında dolu bir
# klasör bekler, değiştirilmemeli).
IMAGE_CATEGORIES = ["natural", "cgi", "mixed"]

# ---------------------------------------------------------------------------
# UI'daki test görüntü kategorileri: (klasör_adı, görünen_etiket).
# Yalnız arayüzün örnek galerisi içindir; deney betiklerini ETKİLEMEZ
# (IMAGE_CATEGORIES ile kasıtlı olarak ayrı tutulur). "classic" klasörü
# experiments/prepare_classic.py ile üretilir (skimage.data — CC0/BSD).
#
# Natural/Hybrid/Synthetic (renkli) kategorileri kaldırıldı — kullanıcı
# yalnız hocanın referans BPP-PSNR aralığına (grayscale JPEG deneyi)
# uyan görüntülerin kalmasını istedi. Geriye kalan 9 görüntü ("classic"
# altında) gerçek libjpeg ile 0.25/0.50/0.75/1.00 bpp'de test edilip
# referans aralığına (±1.5dB tolerans) uyduğu doğrulanmış, gri tonlamaya
# çevrilmiş görüntülerdir.
# ---------------------------------------------------------------------------
UI_IMAGE_CATEGORIES = [
    ("classic", "Classic Test Images (Grayscale)"),
]

# ---------------------------------------------------------------------------
# DCT motoru (JPEG mantığı)
# ---------------------------------------------------------------------------
DCT_BLOCK_SIZE = 8

# Standart JPEG luminance kuantalama matrisi (ITU-T T.81, Annex K)
JPEG_LUMA_QTABLE = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]

# Standart JPEG chroma kuantalama matrisi (renkli görüntüler için)
JPEG_CHROMA_QTABLE = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
]

# RD eğrileri için taranacak quality değerleri (1-100, JPEG anlamında)
DCT_QUALITY_SWEEP = [5, 10, 20, 35, 50, 70, 85, 95]

# UI'da seçilebilir DCT blok boyutları. Yalnız 8 standart JPEG'tir; 4/16
# eğitim amaçlı deneydir (bkz. src/engines/dct_engine.get_qtable).
DCT_BLOCK_SIZE_OPTIONS = [4, 8, 16]

# ---------------------------------------------------------------------------
# Wavelet motoru (JPEG2000 mantığı)
# ---------------------------------------------------------------------------
# Kıyaslanacak filter bank'ler (en az 4; bior4.4 = CDF 9/7, bior2.2 = 5/3)
WAVELET_FILTER_BANKS = ["haar", "db4", "bior4.4", "bior2.2"]
WAVELET_DEFAULT_FILTER = "bior4.4"
WAVELET_LEVELS = 4

# UI'da (DWT Explorer + karşılaştırma ekranı) seçilebilir dalgacık ailesi
# listesi. Her biri pywt'de gerçekten mevcut ve wavedec2/waverec2 ile
# doğrudan çalışır (bkz. src/viz/subbands.filter_bank_info doğrulaması).
WAVELET_UI_OPTIONS = ["haar", "db2", "db4", "db8", "db12", "sym4", "coif1", "bior2.2", "bior4.4"]
# UI'da izin verilen azami seviye üst sınırı; gerçek üst sınır görüntü
# boyutu ve seçili dalgacığa göre wavelet_engine.max_decomposition_level
# ile HER ZAMAN dinamik olarak yeniden hesaplanır (bu yalnız slider tavanıdır).
WAVELET_LEVEL_UI_MAX = 6

# RD eğrileri için taranacak temel kuantalama adımları (büyük adım = düşük bpp)
WAVELET_STEP_SWEEP = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]

# ---------------------------------------------------------------------------
# Semantik önem haritası (YOLO instance segmentation)
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolo11n-seg.pt"        # pretrained; ultralytics otomatik indirir
YOLO_CONF_THRESHOLD = 0.30
# Önemli sayılacak COCO sınıfları (None = tüm tespit edilen nesneler önemli)
IMPORTANT_CLASSES = None
# Maskeyi kaç piksel genişletelim (nesne sınırlarındaki halo artefaktını önler)
MASK_DILATE_PX = 8

# ---------------------------------------------------------------------------
# ROI bit yönlendirme
# ---------------------------------------------------------------------------
# Bir 8x8 DCT bloğunun "önemli" sayılması için maskeyle asgari örtüşme oranı
ROI_BLOCK_OVERLAP_THRESHOLD = 0.10
# Foreground / background kuantalama sertliği oranı:
# background, foreground'dan ROI_BG_COARSENESS kat daha kaba kuantalanır.
ROI_BG_COARSENESS = 6.0

# ---------------------------------------------------------------------------
# Adil kıyaslama: bit bütçesi eşleme
# ---------------------------------------------------------------------------
# Deneylerde hedeflenen bpp noktaları
TARGET_BPP_POINTS = [0.25, 0.50, 0.80, 1.20]
# Hedef bpp'ye yakınsama toleransı (göreli): |gerçek-hedef|/hedef < tolerans
BPP_MATCH_TOLERANCE = 0.02
# Quality parametresi üzerinde bisection için azami iterasyon
BPP_MATCH_MAX_ITER = 30

# ---------------------------------------------------------------------------
# Faz 3 (opsiyonel): algısal metrik
# ---------------------------------------------------------------------------
LPIPS_NET = "alex"
