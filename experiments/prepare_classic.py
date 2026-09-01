# -*- coding: utf-8 -*-
""""Classic Test Images" kategorisini hazırlar: data/classic/*.png.

Klasik DSP/görüntü işleme test görüntüleri (Cameraman, Coins, Moon, Page,
Checkerboard) `scikit-image`'ın kendi örnek veri setinden (`skimage.data`)
üretilir — proje zaten `astronaut`/`bus`/`zidane` için aynı kaynağı
kullanıyor (bkz. prepare_data.py), bu nedenle aynı lisans/kaynak
sözleşmesine uyar; ek bir ağ isteği veya harici indirme YOKTUR.

Lena KASITLI OLARAK dahil edilmemiştir: klasik Lena görüntüsü artık
scikit-image'da bulunmuyor (rıza/telif tartışmaları nedeniyle proje
tarafından kaldırıldı) ve bu depoda başka bir yerde de mevcut değil.
Cameraman (`skimage.data.camera`, CC0 — fotoğrafçı Lav Varshney) modern DSP
literatüründe Lena'nın yerine geçen, tartışmasız lisanslı standart
alternatiftir ve bu kategoride "bayrak taşıyıcı" görüntü olarak kullanılır.

Lena'yı manuel eklemek isteyen biri için: `data/classic/lena.png` dosyasını
(kendi sahip olduğu/lisansladığı bir kopyayla) bu klasöre eklemesi yeterlidir
— uygulama data/classic/*.png dosyalarını otomatik olarak listeler.

Kullanım:  python experiments/prepare_classic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from skimage import data

import config

# (dosya adı, skimage.data fonksiyonu, kaynak notu)
CLASSIC_IMAGES = [
    ("cameraman", data.camera, "skimage.data.camera — CC0 (Lav Varshney); Lena'nın modern lisanssız muadili"),
    ("coins", data.coins, "skimage.data.coins — scikit-image örnek verisi (BSD)"),
    ("moon", data.moon, "skimage.data.moon — scikit-image örnek verisi (BSD)"),
    ("page", data.page, "skimage.data.page — scikit-image örnek verisi (BSD)"),
    ("checkerboard", data.checkerboard, "skimage.data.checkerboard — sentetik kalibrasyon deseni (BSD)"),
]


def main() -> None:
    out_dir = config.DATA_DIR / "classic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, loader, note in CLASSIC_IMAGES:
        img = loader()
        Image.fromarray(img).save(out_dir / f"{name}.png")
        print(f"[classic] {name}.png  ({img.shape})  — {note}")
    print(f"\nToplam {len(CLASSIC_IMAGES)} klasik test görüntüsü: {out_dir}")
    print("Not: Lena kasıtlı olarak eklenmedi (bkz. modül docstring'i). "
         "Manuel eklemek için data/classic/lena.png oluşturun.")


if __name__ == "__main__":
    main()
