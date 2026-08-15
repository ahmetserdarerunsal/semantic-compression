# Semantik-Farkında Görüntü Sıkıştırma

Klasik dönüşüm tabanlı sıkıştırma (DCT = JPEG mantığı, DWT = JPEG2000 mantığı)
bit bütçesini görüntünün her yerine eşit dağıtır; neyin önemli olduğunu bilmez.
Bu proje, pretrained bir instance segmentation modeliyle (YOLO-seg) önemli
bölgeleri piksel maskesi olarak tespit eder ve **aynı toplam bit bütçesinde**
o bölgelere daha çok bit yönlendirir: önemli nesneler (insan, araç, tabela…)
net kalır, önemsiz arka plan (bina, yol, gökyüzü) agresif sıkıştırılır.

Ana sonuç biçimi: *"aynı bpp'de semantik yaklaşım foreground PSNR'ını
baseline'a göre X dB artırıyor"* — iki motor ve üç görüntü tipi için sayısal
tablo + yan yana görsellerle.

## Kurulum

```bash
pip install -r requirements.txt
```

İlk koşuda `ultralytics`, YOLO-seg ağırlıklarını (`yolo11n-seg.pt`, ~6 MB)
otomatik indirir. GPU gerekmez; her şey CPU'da dakikalar içinde koşar.

## Çalıştırma

```bash
# 1) Test görüntülerini hazırla (indirir + cgi/mixed türevlerini üretir)
python experiments/prepare_data.py

# 2) Tam deneyi koş (RD eğrileri + baseline-vs-semantic tablolar/görseller)
python experiments/run_comparison.py          # tam koşu (~10-15 dk, CPU)
python experiments/run_comparison.py --quick  # hızlı doğrulama (~2-3 dk)

# Ekstra araçlar
python experiments/demo_single.py [görüntü] --engine dct --bpp 0.4  # tek görüntü demosu
python experiments/ablation_coarseness.py                           # bit devri ablasyonu
python app.py                                 # tarayıcıda interaktif arayüz (Gradio)
```

Tüm çıktılar `outputs/` altına yazılır:

| Çıktı | İçerik |
|---|---|
| `rd_<kategori>_psnr.png`, `rd_<kategori>_ssim.png` | DCT + 4 filter bank RD eğrileri (uniform) |
| `semantic_rd_<motor>_<kategori>.png` | FG/BG PSNR vs bpp: baseline ve semantic birlikte |
| `mask_<görüntü>.png` | YOLO-seg önem maskesi overlay'i |
| `roi_<motor>_<görüntü>.png` | Aynı bpp'de baseline vs semantic, FG yakınlaştırmalı |
| `ablation_coarseness_<görüntü>.png` | Bit devri düğmesi: kabalık çarpanı taraması |
| `summary_table.md` | Ana tablo: kategori × motor × bpp başına FG kazancı (ort ± std) |
| `results_full.csv` | Tüm ham ölçümler |

Tüm ayarlanabilir parametreler `config.py`'dedir (quality/step taramaları,
filter bank listesi, hedef bpp noktaları, ROI eşikleri, YOLO ayarları).

## Yöntem

### İşlem hattı

```
görüntü ─┬─> YOLO-seg ──> piksel önem maskesi ──┐
         │                                      v
         └─> RGB→YCbCr ──> dönüşüm (DCT | DWT) ──> BÖLGESEL KUANTALAMA
                                                    │
             bit maliyeti tahmini (entropi) <───────┤
             ters kuantalama + ters dönüşüm <───────┘ ──> rekonstrüksiyon
```

### DCT motoru (`src/engines/dct_engine.py`)

- Görüntü 8×8 bloklara bölünür, her bloğa 2D DCT uygulanır.
- Standart JPEG luminance/chroma kuantalama matrisleri, IJG quality (1-100)
  ölçekleme formülüyle ölçeklenir; katsayılar matrise bölünüp yuvarlanır.
- **ROI:** piksel maskesi 8×8 blok ızgarasına indirgenir (bir blok, maskeyle
  ≥ %10 örtüşüyorsa "önemli"). Önemli bloklar ince matrisle, önemsizler
  `ROI_BG_COARSENESS` (=6) kat kaba matrisle kuantalanır.

### Wavelet motoru (`src/engines/wavelet_engine.py`)

- `pywt.wavedec2` ile 4 seviyeli DWT; filter bank'ler config'ten gelir ve
  deneylerde 4'ü kıyaslanır: `haar`, `db4`, `bior4.4` (≈ JPEG2000 CDF 9/7),
  `bior2.2` (≈ 5/3).
- **Subband'ler arası bit allocation:** her subband'in adımı, o subband'deki
  birim kuantalama hatasının piksel alanında yarattığı bozulmayla (sentez
  kazancı) ters orantılı seçilir: `step_b = base_step / gain_b`. Kazançlar,
  subband'e birim katsayı koyup `waverec2` çıktısının L2 normu ölçülerek bir
  kez hesaplanır ve önbelleklenir. Bu, "distortion'ı en çok düşüren subband'e
  daha çok bit" kuralının doğrudan uygulamasıdır (JPEG2000 Annex E'deki
  enerji ağırlıklarıyla aynı fikir).
- **ROI:** piksel maskesi her detay subband'inin çözünürlüğüne indirgenir
  (herhangi bir örtüşme → katsayı önemli) ve kuantalama adımı katsayı
  seviyesinde seçilir. En kaba LL bandına ROI uygulanmaz: katsayı/bit payı
  ihmal edilebilir, kaba LL ise tüm görüntüde parlaklık bloklaşması yaratır.

### Bit maliyeti tahmini (`src/engines/entropy.py`)

Gerçek bir Huffman/aritmetik kodlayıcı yerine Shannon alt sınırına dayalı,
kodek sembol yapısına sadık bir tahminci kullanılır (projenin odağı bit
*yönlendirme*; kanonik tablo yazmak bilimsel katkı eklemez):

- **DCT:** DC katsayıları bloklar arası DPCM farkı olarak; AC katsayıları
  zigzag sırada JPEG'in `(sıfır-koşu, boyut-kategorisi)` sembolleri + blok
  başına EOB olarak modellenir.
  `bit = H(semboller)·N + Σ genlik bitleri` (H = order-0 entropi).
- **Wavelet:** subband başına kuantalanmış katsayıların order-0 entropisi.

Tahminci her iki motora ve baseline/semantic varyantlara aynı şekilde
uygulandığı için kıyaslamalar tutarlıdır; mutlak bpp değerleri gerçek bir
kodlayıcının üreteceğinin hafifçe altında/üstünde olabilir ama sıralamayı
değiştirmez.

### Adil kıyaslama: bit bütçesi eşleme (`src/roi/bit_allocation.py`)

"Semantic vs baseline" ve "DCT vs wavelet" karşılaştırmalarının tamamı **aynı
gerçek bpp'de** yapılır: hedef bpp verilir, motorun quality/step parametresi
bisection ile hedefe %2 tolerans içinde yakınsatılır (`match_bpp`). Semantik
varyant arka plandan kazandığı bitleri foreground'a harcar; toplam bütçe
baseline ile aynıdır.

### Metrikler (`src/metrics/quality.py`)

- PSNR ve SSIM: global + **foreground (maske içi)** + **background (maske
  dışı)** ayrı ayrı.
- Maskeli SSIM, tam SSIM haritasının maske üzerindeki ortalamasıdır (bölgeyi
  kırpıp yeniden hesaplamak pencere sınır etkisi yüzünden yanlı olurdu).
- bpp: yukarıdaki entropi tahmininden gelen `toplam bit / piksel`.

## Kütüphane / kendi kodumuz sınırı

| Katman | Kaynak |
|---|---|
| Matematiksel dönüşümler: `scipy.fft.dctn/idctn`, `pywt.wavedec2/waverec2` | kütüphane |
| Segmentasyon: pretrained YOLO-seg (`ultralytics`), eğitim yok | kütüphane |
| SSIM haritası: `skimage.metrics.structural_similarity` | kütüphane |
| Quality→kuantalama matrisi ölçekleme, blok/subband kuantalama | **kendi kodumuz** |
| Subband'ler arası bit allocation (sentez kazancı) | **kendi kodumuz** |
| ROI bit yönlendirme (blok etiketleme, subband maske ölçekleme) | **kendi kodumuz** |
| Zigzag + DPCM/RLE sembol modeli ve entropi tabanlı bpp tahmini | **kendi kodumuz** |
| Bit bütçesi eşleme (bisection), FG/BG metrikleri, deney düzeneği | **kendi kodumuz** |

Hazır bir kodlayıcı (`PIL.save(quality=...)` vb.) hiçbir yerde kullanılmaz.

## Veri seti ve varsayımlar

`experiments/prepare_data.py` üç kategoriyi yeniden üretilebilir şekilde kurar:

- **natural:** gerçek fotoğraflar — skimage `astronaut`; ultralytics örnek
  görüntüleri `bus.jpg`, `zidane.jpg`; Kodak test setinden `kodim05`,
  `kodim15` (sıkıştırma literatürünün standardı); iki trafik sahnesi:
  `kenya_traffic` (Wikimedia Commons, CC BY-SA 4.0 — okunabilir plaka içerir,
  projenin motivasyon senaryosu) ve `singapore_jam` (Wikimedia Commons,
  epSos.de, CC BY 2.0 — 18 araçlı yoğun sahne).
- **cgi:** aynı fotoğrafların cel-shading (bilateral filtre + posterizasyon +
  kenar çizgisi) ile kartunlaştırılmış halleri. *Varsayım:* gerçek render
  çıktısı yerine bu dönüşüm, CGI içeriğin ayırt edici istatistiklerini (düz
  bölgeler, keskin kenarlar) temsil eder ve YOLO nesneleri tespit etmeye
  devam eder (9/9 görüntüde doğrulandı).
- **mixed:** doğal fotoğraf + sentetik HUD/metin katmanı (ekran görüntüsü /
  AR benzeri karma içerik).

## Sonuçların okunması

- `summary_table.md`: pozitif "FG PSNR kazancı", semantik yaklaşımın aynı
  bütçede önemli bölgeleri o kadar dB daha iyi kodladığını gösterir; "BG
  bedeli" bunun arka planda ödenen karşılığıdır. Kazanç, foreground küçük ve
  arka plan dokulu olduğunda büyür (bit devri daha kârlı); arka planı zaten
  "ucuz" (bulanık/düz) görüntülerde küçülür — ör. `zidane` (bulanık iç mekân
  arka planı) wavelet'te ~+0.1 dB, `bus` (dokulu bina/kaldırım) +1.5-1.8 dB.
- RD eğrileri: `bior4.4`/`db4`/`bior2.2` beklendiği gibi `haar`'dan iyi;
  wavelet motoru düşük bpp'de DCT'den önde (bloklaşma yok).

## Sınırlar ve Faz 3 (gelecek işi)

- Gerçek bit akışı yazılmaz (bpp entropi tahminidir; yöntem yukarıda).
- Segmentasyon hatası ROI'yi doğrudan etkiler: YOLO bir nesneyi kaçırırsa o
  bölge arka plan gibi kaba kodlanır. `MASK_DILATE_PX` bunu kısmen tamponlar.
- **Faz 3 (opsiyonel PoC, implemente edilmedi):** agresif sıkıştırılan arka
  planı pretrained bir restorasyon modeliyle (ör. Real-ESRGAN) alıcı tarafta
  geri kurmak ve LPIPS ile "PSNR düşse de algısal kalite artıyor" göstermek.
  Altyapı hazır: `config.LPIPS_NET` ve `requirements.txt` içindeki opsiyonel
  `lpips` satırı bu genişleme içindir.

## Referans çalışmalar

- Prakash et al., *Semantic Perceptual Image Compression using Deep
  Convolution Networks* (MSROI) — semantik ROI + JPEG fikrinin öncüsü.
- Akbari et al., *DSSLIC: Deep Semantic Segmentation-based Layered Image
  Compression*.
- JPEG (ITU-T T.81) ve JPEG2000 (ITU-T T.800) standartları — kuantalama
  matrisi, zigzag/RLE sembolleri ve subband ağırlıkları için.
