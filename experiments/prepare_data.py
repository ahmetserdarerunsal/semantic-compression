# -*- coding: utf-8 -*-
"""Test görüntülerini hazırlar (yeniden üretilebilirlik için betikleştirildi).

Kategoriler:
- natural: gerçek fotoğraflar (skimage 'astronaut' + ultralytics örnek
  görüntüleri bus.jpg / zidane.jpg — her ikisi de segmentasyon literatüründe
  standart test görüntüsüdür).
- cgi: doğal fotoğrafların cel-shading (bilateral filtre + renk
  posterizasyonu + kenar çizgisi) ile kartunlaştırılmış halleri. Gerçek bir
  render motoru çıktısı yerine bu dönüşümü kullanıyoruz; CGI içeriğin ayırt
  edici istatistiklerini (düz/az dokulu bölgeler, keskin kenarlar) taşır ve
  YOLO nesneleri hâlâ tespit edebilir. (Varsayım README'de belirtilmiştir.)
- mixed: doğal fotoğraf + sentetik grafik katmanı (HUD çizgileri, metin
  kutuları) — ekran görüntüsü / artırılmış gerçeklik benzeri içerik.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
from skimage import data

from config import DATA_DIR

URLS = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    # Trafik sahnesi, okunabilir plaka (KCK 890Q) — Wikimedia Commons, CC BY-SA 4.0
    "kenya_traffic.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/7f/Cars_at_a_traffic_light.jpg",
    # Yoğun trafik, çok nesne — Wikimedia Commons, epSos.de, CC BY 2.0
    "singapore_jam.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Driving%20Cars%20in%20a%20Traffic%20Jam.jpg",
    # Kodak test seti (sıkıştırma literatürünün standardı, telifsiz)
    "kodim05.png": "http://r0k.us/graphics/kodak/kodak/kodim05.png",
    "kodim15.png": "http://r0k.us/graphics/kodak/kodak/kodim15.png",
}

# Kaynak bazlı ön kırpma (y0, y1, x0, x1) — tam çözünürlük pikselleriyle.
# kenya_traffic: araç içi karanlık çerçeve (ön cam/ayna) atılır.
PRE_CROP = {"kenya_traffic": (270, 2385, 585, 3456)}
MAX_SIDE = 768  # deneylerin makul sürede koşması için üst sınır


def _resize_cap(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def cartoonize(rgb: np.ndarray) -> np.ndarray:
    """Cel-shading: bilateral yumuşatma + posterizasyon + kenar çizgileri."""
    smooth = rgb
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=60, sigmaSpace=9)
    poster = (smooth // 32) * 32 + 16  # 8 seviyeli renk kuantalaması
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=5,
    )
    return cv2.bitwise_and(poster, poster, mask=edges)


def add_hud_overlay(rgb: np.ndarray) -> np.ndarray:
    """Doğal fotoğrafa sentetik HUD/grafik katmanı bindirir."""
    out = rgb.copy()
    h, w = out.shape[:2]
    # yarı saydam panel + metin
    panel = out[h - 90 : h - 10, 10 : w // 2].astype(np.int32)
    out[h - 90 : h - 10, 10 : w // 2] = np.clip(panel // 3 + 40, 0, 255).astype(np.uint8)
    cv2.putText(out, "SENSOR 07  FPS 60  REC", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
    cv2.putText(out, "LAT 41.0082 LON 28.9784", (20, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
    # köşe nişangahları ve ızgara çizgileri
    for (x, y) in [(30, 30), (w - 30, 30), (30, h - 120), (w - 30, h - 120)]:
        cv2.drawMarker(out, (x, y), (255, 60, 60), cv2.MARKER_CROSS, 24, 2)
    cv2.rectangle(out, (w // 2 - 60, h // 2 - 60), (w // 2 + 60, h // 2 + 60),
                  (255, 255, 0), 1)
    return out


def main() -> None:
    for cat in ["natural", "cgi", "mixed"]:
        (DATA_DIR / cat).mkdir(parents=True, exist_ok=True)

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (course-project image fetch)")]
    urllib.request.install_opener(opener)

    sources: dict[str, np.ndarray] = {"astronaut": data.astronaut()}
    for fname, url in URLS.items():
        dest = DATA_DIR / fname
        if not dest.exists():
            print(f"indiriliyor: {url}")
            urllib.request.urlretrieve(url, dest)
        rgb = np.array(Image.open(dest).convert("RGB"))
        stem = Path(fname).stem
        if stem in PRE_CROP:
            y0, y1, x0, x1 = PRE_CROP[stem]
            rgb = rgb[y0:y1, x0:x1]
        sources[stem] = rgb

    for name, rgb in sources.items():
        rgb = _resize_cap(rgb)
        Image.fromarray(rgb).save(DATA_DIR / "natural" / f"{name}.png")
        Image.fromarray(cartoonize(rgb)).save(DATA_DIR / "cgi" / f"{name}_cgi.png")
        Image.fromarray(add_hud_overlay(rgb)).save(DATA_DIR / "mixed" / f"{name}_mixed.png")
        print(f"hazır: {name} ({rgb.shape[1]}x{rgb.shape[0]})")

    print("Veri hazırlama tamam.")


if __name__ == "__main__":
    main()
