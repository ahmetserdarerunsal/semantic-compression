# -*- coding: utf-8 -*-
"""Tek görüntüde uçtan uca demo: segmentasyon + ROI + baseline-vs-semantic.

Kullanım:
    python experiments/demo_single.py [görüntü_yolu] [--engine dct|wavelet]
                                      [--bpp 0.40]

Görüntü verilmezse data/natural/bus.png kullanılır. Çıktılar outputs/ altına
demo_* öneki ile yazılır.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import config
from src.engines import dct_engine, wavelet_engine
from src.metrics.quality import evaluate
from src.roi.bit_allocation import mask_to_block_importance, match_bpp
from src.semantic.importance_map import get_importance_mask
from src.viz.plots import plot_comparison, plot_mask_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?",
                        default=str(config.DATA_DIR / "natural" / "bus.png"))
    parser.add_argument("--engine", choices=["dct", "wavelet"], default="dct")
    parser.add_argument("--bpp", type=float, default=0.40)
    args = parser.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    stem = Path(args.image).stem
    print(f"görüntü: {args.image} ({img.shape[1]}x{img.shape[0]})")

    mask, labels = get_importance_mask(img)
    if not mask.any():
        raise SystemExit("YOLO-seg hiç nesne bulamadı; ROI demo'su uygulanamaz.")
    print(f"maske kapsama: {mask.mean()*100:.1f}%  nesneler: {sorted(set(labels))}")

    if args.engine == "dct":
        block_imp = mask_to_block_importance(mask)
        base = match_bpp(lambda q: dct_engine.compress_image(img, q),
                         args.bpp, 1, 100, True)
        sem = match_bpp(
            lambda q: dct_engine.compress_image(
                img, q, block_imp, config.ROI_BG_COARSENESS
            ),
            args.bpp, 1, 100, True,
        )
        engine_name = "DCT motoru"
    else:
        wav = config.WAVELET_DEFAULT_FILTER
        base = match_bpp(lambda s: wavelet_engine.compress_image(img, s, wav),
                         args.bpp, 0.25, 512, False)
        sem = match_bpp(
            lambda s: wavelet_engine.compress_image(
                img, s, wav, importance_mask=mask,
                bg_coarseness=config.ROI_BG_COARSENESS,
            ),
            args.bpp, 0.25, 512, False,
        )
        engine_name = f"Wavelet motoru ({config.WAVELET_DEFAULT_FILTER})"

    mb = evaluate(img, base[0], mask)
    ms = evaluate(img, sem[0], mask)
    print(f"baseline: bpp={base[1]:.3f}  FG {mb['fg_psnr']:.2f} dB / BG {mb['bg_psnr']:.2f} dB")
    print(f"semantic: bpp={sem[1]:.3f}  FG {ms['fg_psnr']:.2f} dB / BG {ms['bg_psnr']:.2f} dB")
    print(f"FG kazancı: {ms['fg_psnr']-mb['fg_psnr']:+.2f} dB "
          f"(BG bedeli: {ms['bg_psnr']-mb['bg_psnr']:+.2f} dB)")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    plot_mask_overlay(img, mask, labels,
                      config.OUTPUT_DIR / f"demo_mask_{stem}.png")
    plot_comparison(img, base[0], sem[0], mask, args.bpp, mb, ms,
                    config.OUTPUT_DIR / f"demo_{args.engine}_{stem}.png",
                    engine_name)
    print(f"görseller: outputs/demo_mask_{stem}.png, outputs/demo_{args.engine}_{stem}.png")


if __name__ == "__main__":
    main()
