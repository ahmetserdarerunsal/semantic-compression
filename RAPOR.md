# Semantik-Farkında Görüntü Sıkıştırma: DCT ve Wavelet Kodlayıcılarda Segmentasyon Güdümlü Bit Yönlendirme

*Sinyal İşleme Ders Projesi — Ahmet Serdar Erünsal*

## Özet

Klasik dönüşüm tabanlı görüntü sıkıştırıcıları (JPEG, JPEG2000) bit bütçesini
görüntü içeriğinden bağımsız, uzamsal olarak eşit dağıtır. Bu çalışmada,
pretrained bir instance segmentation modelinin (YOLO11n-seg) ürettiği piksel
önem maskesini, elle yazılmış DCT ve DWT kodlayıcılarının kuantalama katmanına
bağlayan bir sistem sunuyoruz. Sistem, sabit toplam bit bütçesinde bitleri
semantik olarak önemli bölgelere (insan, araç, plaka, tabela) yönlendirir.
21 görüntülük üç kategorili bir test setinde (doğal, CGI-benzeri, karma),
iki kodlayıcıda ve dört bit hızında yapılan 168 karşılaştırmanın tamamında
yöntem, aynı bpp'de foreground PSNR'ını artırmaktadır (tipik +1–3 dB;
düşük maske kapsamalı trafik sahnelerinde +5 dB'ye kadar).
Bit devri şiddetinin (arka plan kabalık çarpanı) ablasyonu, DCT'de monoton
artan, wavelet'te ise doyuma ulaşan bir kazanç profili ortaya koymaktadır.

## 1. Giriş

Kayıplı görüntü sıkıştırmada temel değiş-tokuş, bit hızı (bpp) ile bozulma
arasındadır ve klasik kodekler bu değiş-tokuşu görüntünün her yerinde aynı
kuantalama sertliğiyle çözer. Oysa birçok uygulamada (araç kameraları,
güvenlik, video-konferans) görüntünün küçük bir bölümü görevin tamamını
taşır: plakanın okunması, yayanın seçilmesi, yüzün tanınması. Uniform
kodlayıcı, gökyüzüne ve asfalta, plakaya harcadığı özenle bit harcar.

Bu projenin tezi şudur: modern segmentasyon modelleri "neresi önemli"
sorusunu piksel hassasiyetinde yanıtlayabildiğine göre, sıkıştırıcının
kuantalama kararı bu bilgiyle koşullandırılabilir. Katkılarımız:

1. Kuantalama, bit tahsisi ve ROI yönlendirme mantığı tamamen tarafımızca
   yazılmış iki dönüşüm kodlayıcısı (8×8 blok DCT / çok seviyeli DWT);
2. YOLO-seg piksel maskesini iki farklı dönüşüm alanına (blok ızgarası /
   subband hiyerarşisi) eşleyen iki ROI mekanizması;
3. Eşit-bpp kısıtı altında adil kıyaslama düzeneği (bisection ile bit
   bütçesi eşleme) ve foreground/background ayrımlı metrik protokolü;
4. İçerik tipi (doğal/CGI/karma), filter bank ve bit devri şiddeti
   boyutlarında sistematik deneyler.

## 2. İlgili Çalışmalar

**Semantik ROI sıkıştırma.** Prakash ve ark. (MSROI), bir CNN'in çıkardığı
çok yapılı önem haritasıyla JPEG kuantalamasını uzamsal olarak modüle eden
öncü çalışmadır; bizim DCT hattımız bu fikrin blok-seviyesinde, ikili maskeli
bir uygulamasıdır. Akbari ve ark. (DSSLIC), segmentasyon haritasını ayrı bir
katman olarak kodlayıp derin bir ağla rekonstrüksiyon yapar; biz kodlayıcıyı
klasik tutup yalnız bit dağıtımını öğrenilmiş bilgiyle güderek katkıyı izole
ediyoruz. JPEG2000 standardının kendisi de ROI kodlamayı destekler (Maxshift
yöntemi, ITU-T T.800 §H); bizim wavelet ROI mekanizmamız bunun katsayı
seviyesinde, adım haritalı bir benzeridir.

**Kod tabanları.** DWT baseline'ı için NamitS27/Image-Compression-DWT'nin
genel akışı incelenmiş, kod alınmamıştır; YOLO-seg kullanımı ultralytics
resmi API'siyledir; generative-compression (Justin-Tan) yalnız Faz 3
bağlamı için referanstır.

## 3. Yöntem

### 3.1 Sistem mimarisi

```
girdi ─┬─> YOLO11n-seg ──> piksel önem maskesi M ──┐
       │                                           v
       └─> RGB→YCbCr ──> dönüşüm (DCT | DWT) ──> BÖLGESEL KUANTALAMA
                                                   │
             bit maliyeti (entropi modeli) <───────┤
             ters kuantalama + ters dönüşüm <──────┘──> rekonstrüksiyon
```

Kütüphane/kendi-kod sınırı nettir: `scipy.fft.dctn` ve `pywt.wavedec2`
yalnız matematiksel dönüşümü sağlar; kuantalama, bit tahsisi, ROI ve bit
sayımı katmanlarının tamamı proje kodudur. Hazır bir kodlayıcı
(`PIL.save(quality=…)` vb.) hiçbir aşamada kullanılmamıştır.

### 3.2 DCT kodlayıcısı

Görüntü YCbCr'ye çevrilir, her kanal 8×8 bloklara bölünür ve bloklara 2D DCT
uygulanır. Kuantalama matrisi, standart JPEG luminance/chroma tablolarının
IJG quality ölçeklemesiyle üretilir:

```
scale(q) = 5000/q        (q < 50)
           200 − 2q      (q ≥ 50)
Q_q = clip(⌊(Q50·scale + 50)/100⌋, 1, 255)
```

**ROI eşleme:** piksel maskesi 8×8 ızgaraya indirgenir; bir blok, alanının
≥ %10'u maskeyle örtüşüyorsa "önemli" etiketlenir. Önemli bloklar Q_q ile,
önemsizler c·Q_q ile kuantalanır (c: arka plan kabalık çarpanı, varsayılan 6).

### 3.3 Wavelet kodlayıcısı

Her kanala 4 seviyeli 2D DWT uygulanır (filter bank deneylerde değişken:
haar, db4, bior4.4 ≈ CDF 9/7, bior2.2 ≈ 5/3). Subband'ler arası bit tahsisi
sentez kazancına dayanır: subband b'ye birim katsayı hatası konduğunda piksel
alanında oluşan L2 bozulma g_b bir kez ölçülür ve adım `Δ_b = Δ_base / g_b`
seçilir. Bu, "birim bitin distortion'ı en çok düşürdüğü subband'e öncelik"
kuralının doğrudan uygulaması olup JPEG2000 Annex E'deki enerji
ağırlıklarıyla aynı ilkedir.

**ROI eşleme:** maske her detay subband'inin çözünürlüğüne, "herhangi bir
örtüşme → önemli" kuralıyla (max-pooling) indirgenir ve adım katsayı
seviyesinde Δ_b veya c·Δ_b olarak seçilir. En kaba LL bandına ROI
uygulanmaz: bit payı ihmal edilebilirken kaba LL tüm görüntüde parlaklık
bloklaşması yaratır.

### 3.4 Bit maliyeti modeli

Gerçek bir entropi kodlayıcısı yerine Shannon alt sınırına dayalı, kodek
sembol yapısına sadık bir tahminci kullanılır: DCT'de DC katsayıları DPCM
farkı kategorileri, AC katsayıları zigzag sırada (sıfır-koşu, boyut
kategorisi) sembolleri + EOB olarak modellenir ve
`bit = H(semboller)·N + Σ genlik biti` hesaplanır; wavelet'te subband başına
order-0 entropi kullanılır. Tahminci tüm yöntem ve varyantlara özdeş
uygulandığından kıyaslamalar tutarlıdır; mutlak bpp gerçek bir kodlayıcıya
göre birkaç yüzde sapabilir ancak sıralamayı etkilemez (bkz. §6).

### 3.5 Adil kıyaslama protokolü

Tüm "baseline vs semantic" ve "DCT vs wavelet" karşılaştırmaları aynı gerçek
bpp'de yapılır: hedef bpp verilir, kalite parametresi (q veya Δ_base)
bisection ile hedefe %2 göreli tolerans içinde yakınsatılır. Semantik
varyant, arka planda tasarruf ettiği bitleri foreground'a harcar; toplam
bütçe baseline ile eşittir.

### 3.6 Metrikler

PSNR ve SSIM; global değerlerin yanında maske içi (FG) ve maske dışı (BG)
için ayrı ayrı raporlanır. Maskeli SSIM, tam SSIM haritasının ilgili bölge
üzerindeki ortalamasıdır (bölge kırpma, pencere sınır etkisi nedeniyle yanlı
olurdu). bpp, §3.4'teki modelden gelir.

## 4. Deney Düzeneği

**Veri seti** (21 görüntü = 7 kaynak × 3 kategori, `prepare_data.py` ile
yeniden üretilebilir): *natural* — astronaut (skimage), bus ve zidane
(ultralytics), kodim05 ve kodim15 (Kodak), kenya_traffic (Wikimedia Commons
CC BY-SA 4.0; okunabilir plakalı trafik sahnesi, motivasyon senaryomuz),
singapore_jam (Wikimedia Commons CC BY 2.0, epSos.de; 18 araçlı yoğun sahne).
*cgi* — aynı kaynakların cel-shading (bilateral filtre + posterizasyon +
kenar çizgisi) ile kartunlaştırılmış halleri; gerçek render yerine CGI
içeriğin karakteristik istatistiklerini (düz bölgeler, keskin kenarlar)
temsil eder. *mixed* — doğal fotoğraf + sentetik HUD/metin katmanı.

**Konfigürasyon:** YOLO11n-seg, güven eşiği 0.30, maske 8 px dilate; hedef
bpp noktaları {0.25, 0.50, 0.80, 1.20}; kabalık çarpanı c = 6 (ablasyonla
gerekçelendirilir, §5.4). Tüm parametreler `config.py`'dedir; tüm deneyler
`python experiments/run_comparison.py` tek komutuyla yeniden üretilür.

## 5. Sonuçlar

> Bu bölümdeki sayılar `outputs/summary_table.md` ve `results_full.csv`
> dosyalarından alınmıştır; şekiller `outputs/` klasöründedir.

### 5.1 Uniform baseline: filter bank ve motor kıyası

(bkz. `rd_<kategori>_psnr.png`) Dört filter bank arasında bior4.4 (CDF 9/7)
ve bior2.2 tüm kategorilerde en iyi RD performansını verir; haar belirgin
biçimde geridedir; db4 arada kalır. DCT, orta-yüksek bpp'de bior ailesine
yaklaşır ancak düşük bpp'de bloklaşma nedeniyle geri düşer. Bu sıralamaların
literatürle (JPEG2000'in CDF 9/7 tercihi) örtüşmesi, kodlayıcıların doğru
kurulduğunun bir doğrulamasıdır.

### 5.2 Ana sonuç: eşit bpp'de semantik bit yönlendirme

168 karşılaştırmanın (21 görüntü × 2 motor × 4 bpp) **tamamında** FG kazancı
pozitiftir. Kategori ortalamaları (`summary_table.md`'den seçme):

| Kategori | Motor | FG kazancı @0.25 bpp | @0.50 | @0.80 | @1.20 |
|---|---|---|---|---|---|
| natural | dct | +1.42 ± 0.93 dB | +1.78 ± 1.01 | +2.04 ± 1.16 | +2.42 ± 1.32 |
| natural | wavelet | +1.08 ± 0.92 | +1.42 ± 1.04 | +1.64 ± 1.00 | +1.91 ± 0.99 |
| cgi | dct | +1.23 ± 0.89 | +1.89 ± 1.10 | +2.52 ± 1.36 | +3.25 ± 1.71 |
| cgi | wavelet | +1.97 ± 1.48 | +2.61 ± 1.71 | +2.91 ± 1.68 | +2.97 ± 1.62 |
| mixed | dct | +1.39 ± 1.05 | +1.82 ± 1.03 | +2.10 ± 1.27 | +2.44 ± 1.45 |
| mixed | wavelet | +1.14 ± 0.99 | +1.45 ± 1.12 | +1.66 ± 1.17 | +1.91 ± 1.13 |

Karşılık gelen BG bedeli −2.6 … −5.4 dB aralığındadır (tam tablo:
`summary_table.md`). Üç düzenli örüntü gözlenir:

1. **Kazanç bpp ile büyür:** yüksek bütçede arka planda "çalınabilecek" bit
   artar; kazanç doygunluğa ulaşmaz (test aralığında).
2. **Kazanç maske kapsamasıyla ters orantılıdır:** kapsaması %16 olan
   kenya_traffic doğal halde +3.4…+5.0 dB alırken, kapsaması %72 olan
   kodim15 +0.4 dB civarında kalır — devralınan bitler küçük bir alana
   yoğunlaştığında etki büyür. Motivasyon senaryomuz (küçük ama kritik
   nesneler) tam da bu verimli rejimdedir.
3. **Görsel kanıt:** `roi_dct_kenya_traffic.png` (0.80 bpp) ve
   `demo_dct_kenya_traffic.png` (0.30 bpp) karşılaştırmalarında `KCK 890Q`
   plakası semantik varyantta belirgin biçimde daha keskindir; 0.30 bpp'de
   FG kazancı +3.40 dB'dir.

### 5.3 Semantik RD eğrileri

(bkz. `semantic_rd_<motor>_<kategori>.png`) FG eğrisi tüm bit hızlarında
baseline'ın üzerinde, BG eğrisi altındadır; aradaki dikey mesafe her bit
hızında korunur. Bu, kazancın tek bir çalışma noktasına özgü olmadığını
gösterir.

### 5.4 Ablasyon: bit devri şiddeti

(bkz. `ablation_coarseness_kenya_traffic.png`; kenya_traffic @ 0.40 bpp)
Çarpan c=1 ROI'yi kapatır. DCT'de FG kazancı c ile monoton artar
(c=16'da +4.77 dB) ancak BG bedeli de hızlanarak büyür (−5.53 dB).
Wavelet'te kazanç c≈8'de doyar (+3.68 dB) ve c=16'da geriler (+3.18 dB):
arka plan katsayıları belli bir kabalıktan sonra zaten sıfırlandığından
devredilecek bit kalmaz; ek kabalık yalnız bozulmayı artırır. Varsayılan
c=6, iki motorda da eğrinin verimli bölgesindedir.

### 5.5 İçerik tipine göre davranış

CGI kategorisinde ortalama kazançlar doğal görüntülerin üzerine bile
çıkmıştır (ör. wavelet @0.80 bpp: +2.91'e karşı +1.64 dB). Bunun iki nedeni
vardır: (i) kartunlaştırılmış içerikte YOLO tespitleri kısmileşip maske
kapsaması düşmüş (kodim05: %50→%22), küçülen foreground'a yoğunlaşan bit
devri kazancı büyütmüştür; (ii) düz arka plan bölgeleri çok ucuz
kodlandığından kaba kuantalamanın BG bedeli görece düşüktür. Ancak (i)
aynı zamanda bir uyarıdır: kazanç, *tespit edilen* bölge üzerinden ölçülür;
model bir nesneyi kaçırırsa o nesne metriğe foreground olarak girmez ve
arka plan muamelesi görür (bkz. §6). Karma içerikte HUD katmanı keskin
kenarlarıyla bit tüketse de kazanç profili doğal içerikle benzerdir.

## 6. Sınırlılıklar

1. **bpp bir tahmindir:** gerçek bit akışı yazılmaz; Shannon-tabanlı model
   tüm varyantlara özdeş uygulandığından *göreli* sonuçlar güvenilirdir,
   mutlak bpp gerçek kodlayıcıdan birkaç yüzde sapabilir.
2. **Segmentasyon tek hata noktasıdır:** YOLO'nun kaçırdığı bir nesne arka
   plan gibi kaba kodlanır. 8 px maske genişletme sınır hatalarını
   tamponlar ama kavramsal kaçırmayı (ör. stilize içerikte tespit düşmesi)
   telafi edemez. CGI deneyindeki kısmi tespitler bunun somut örneğidir.
3. **İkili önem modeli:** maske içi/dışı iki seviyedir; MSROI'deki gibi
   dereceli önem haritası daha yumuşak geçişler sağlayabilirdi.
4. **Maske yan bilgisi kodlanmaz:** gerçek bir sistemde maskenin (veya blok
   etiketlerinin) alıcıya iletim maliyeti vardır; blok-seviyesi ikili harita
   için bu maliyet ≈ 0.01–0.02 bpp mertebesinde olup sonuçları değiştirmez,
   ancak tam bir codec'te hesaba katılmalıdır.

## 7. Sonuç ve Gelecek Çalışma

Pretrained segmentasyonun piksel maskesiyle güdülen bölgesel kuantalamanın,
iki farklı dönüşüm ailesinde ve üç içerik tipinde, eşit bit bütçesi kısıtı
altında foreground kalitesini tutarlı biçimde artırdığını gösterdik. Yöntem
kavramsal olarak basittir, mevcut kodek iskeletlerine eklenebilir ve
kazancı en çok, önemli bölgenin küçük olduğu tipik gözetim/araç senaryosunda
verir. Gelecek çalışma: (i) agresif sıkıştırılan arka planın alıcı tarafta
pretrained bir restorasyon modeliyle (Real-ESRGAN) geri kurulup LPIPS ile
algısal değerlendirilmesi (Faz 3 PoC); (ii) dereceli önem haritası;
(iii) maske yan bilgisinin gerçek kodlanması.

## Kaynakça

- A. Prakash vd., "Semantic Perceptual Image Compression using Deep
  Convolution Networks," DCC 2017 (MSROI).
- M. Akbari vd., "DSSLIC: Deep Semantic Segmentation-based Layered Image
  Compression," ICASSP 2019.
- ITU-T T.81 (JPEG) — Annex K kuantalama tabloları; ITU-T T.800 (JPEG2000)
  — Annex E kuantalama, Annex H ROI (Maxshift).
- G. Jocher vd., Ultralytics YOLO11, 2024.
- Z. Wang vd., "Image Quality Assessment: From Error Visibility to
  Structural Similarity," IEEE TIP 2004 (SSIM).
