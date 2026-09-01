# -*- coding: utf-8 -*-
"""PGM (Portable GrayMap, Netpbm) okuyucu — P2 (ASCII) ve P5 (binary).

PIL zaten .pgm'i açabiliyor, ancak Gradio'nun gr.Image bileşeni tarayıcı
tarafında sabit `accept="image/*"` filtresi kullanıyor ve .pgm dosyaları
çoğu tarayıcıda `application/octet-stream` olarak raporlandığı için upload
diyaloğunda seçilemiyor/reddediliyor (bkz. app.py'deki pgm_upload bileşeni).
Bu modül, dosya bir kez sunucuya ulaştığında header'ı (magic/width/height/
maxval) ve piksel verisini PIL'e bağımlı olmadan, doğrudan Netpbm
spesifikasyonuna göre ayrıştırır.
"""
from __future__ import annotations

import numpy as np

_WHITESPACE = b" \t\r\n\v\f"


def _next_token(data: bytes, pos: int) -> tuple[bytes, int]:
    """Yorumları (# ... satır sonu) ve boşlukları atlayıp bir sonraki
    token'ı döner. Dönen pos, token'ın SON karakterinden hemen sonrasıdır
    (izleyen boşluk atlanmamıştır — P5'te binary veri öncesi tek ayraç
    baytını doğru tespit edebilmek için gerekli)."""
    n = len(data)
    while pos < n:
        if data[pos:pos + 1] in (b" ", b"\t", b"\r", b"\n", b"\v", b"\f"):
            pos += 1
        elif data[pos:pos + 1] == b"#":
            nl = data.find(b"\n", pos)
            pos = n if nl == -1 else nl + 1
        else:
            break
    start = pos
    while pos < n and data[pos:pos + 1] not in (b" ", b"\t", b"\r", b"\n", b"\v", b"\f", b"#"):
        pos += 1
    return data[start:pos], pos


def parse_pgm(path: str) -> np.ndarray:
    """.pgm dosyasını gerçek 8-bit grayscale numpy dizisine (H, W) çevirir.

    P2 (ASCII) ve P5 (binary) desteklenir; header'daki yorum satırları
    (# ...) Netpbm spesifikasyonuna göre her token arasında atlanır.
    maxval == 255 ise bayt değerleri doğrudan kullanılır; farklıysa
    0-255 aralığına ölçeklenir (16-bit P5 dahil)."""
    with open(path, "rb") as f:
        data = f.read()

    magic, pos = _next_token(data, 0)
    if magic not in (b"P2", b"P5"):
        raise ValueError(
            f"Desteklenmeyen PGM magic number: {magic!r} — yalnız P2 (ASCII) "
            "ve P5 (binary) destekleniyor."
        )

    width_tok, pos = _next_token(data, pos)
    height_tok, pos = _next_token(data, pos)
    maxval_tok, pos = _next_token(data, pos)
    try:
        width, height, maxval = int(width_tok), int(height_tok), int(maxval_tok)
    except ValueError as exc:
        raise ValueError(f"PGM header bozuk (width/height/maxval okunamadı): {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Geçersiz PGM boyutu: {width}x{height}")
    if not (0 < maxval < 65536):
        raise ValueError(f"Geçersiz PGM maxval: {maxval}")

    n_pixels = width * height
    if magic == b"P5":
        pos += 1  # header/binary veri arasındaki TEK zorunlu ayraç baytı
        bytes_per_sample = 1 if maxval < 256 else 2
        needed = n_pixels * bytes_per_sample
        raw = data[pos:pos + needed]
        if len(raw) < needed:
            raise ValueError(
                f"PGM binary veri eksik: {len(raw)} bayt bulundu, {needed} bekleniyordu "
                f"({width}x{height}, {bytes_per_sample} bayt/piksel)."
            )
        if bytes_per_sample == 1:
            values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        else:
            values = np.frombuffer(raw, dtype=">u2").astype(np.float64)
    else:  # P2 — ASCII ondalık değerler, whitespace/yorum ile ayrılmış
        values_list: list[int] = []
        while len(values_list) < n_pixels:
            tok, pos = _next_token(data, pos)
            if not tok:
                raise ValueError(
                    f"PGM (P2) veri eksik: {len(values_list)}/{n_pixels} piksel okundu."
                )
            values_list.append(int(tok))
        values = np.array(values_list, dtype=np.float64)

    if maxval == 255:
        gray8 = values.astype(np.uint8).reshape(height, width)
    else:
        gray8 = np.clip(np.round(values * (255.0 / maxval)), 0, 255).astype(np.uint8).reshape(height, width)
    return gray8
