# -*- coding: utf-8 -*-
"""Ablasyon: arka plan kabalık çarpanı (ROI_BG_COARSENESS) taraması.

Soru: arka planı ne kadar kaba kuantalarsak foreground ne kazanır, arka plan
ne kaybeder? Çarpan 1 = ROI kapalı (baseline); büyüdükçe bit devri artar ama
bir noktadan sonra arka plandan çalınacak bit kalmaz (doyma) ve BG bedeli
büyümeye devam eder.

Kullanım:  python experiments/ablation_coarseness.py [görüntü] [--bpp 0.40]
Çıktı:     outputs/ablation_coarseness_<görüntü>.png + konsol tablosu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from src.engines import dct_engine, wavelet_engine
from src.metrics.quality import evaluate
from src.roi.bit_allocation import mask_to_block_importance, match_bpp
from src.semantic.importance_map import get_importance_mask
from src.viz.plots import PALETTE

COARSENESS_SWEEP = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]


def sweep_engine(
    engine: str, img: np.ndarray, mask: np.ndarray, target_bpp: float
) -> tuple[list[float], list[float]]:
    """Bir motor için çarpan taraması; (FG kazançları, BG bedelleri) döner."""
    block_imp = mask_to_block_importance(mask)
    wav = config.WAVELET_DEFAULT_FILTER

    def encode(coarseness: float):
        if engine == "dct":
            return match_bpp(
                lambda q: dct_engine.compress_image(img, q, block_imp, coarseness),
                target_bpp, 1, 100, True,
            )
        return match_bpp(
            lambda s: wavelet_engine.compress_image(
                img, s, wav, importance_mask=mask, bg_coarseness=coarseness
            ),
            target_bpp, 0.25, 512, False,
        )

    base = evaluate(img, encode(1.0)[0], mask)
    fg_gains, bg_costs = [], []
    for c in COARSENESS_SWEEP:
        m = evaluate(img, encode(c)[0], mask)
        fg_gains.append(m["fg_psnr"] - base["fg_psnr"])
        bg_costs.append(m["bg_psnr"] - base["bg_psnr"])
        print(f"  {engine} c={c:5.1f}  FG {fg_gains[-1]:+.2f} dB  BG {bg_costs[-1]:+.2f} dB")
    return fg_gains, bg_costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?",
                        default=str(config.DATA_DIR / "natural" / "kenya_traffic.png"))
    parser.add_argument("--bpp", type=float, default=0.40)
    args = parser.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    stem = Path(args.image).stem
    mask, labels = get_importance_mask(img)
    if not mask.any():
        raise SystemExit("Nesne bulunamadı.")
    print(f"{stem}: kapsama %{mask.mean()*100:.1f}, hedef {args.bpp} bpp")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, engine in zip(axes, ["dct", "wavelet"]):
        fg, bg = sweep_engine(engine, img, mask, args.bpp)
        ax.plot(COARSENESS_SWEEP, fg, color=PALETTE[1], marker="s",
                linewidth=1.8, label="Foreground kazancı")
        ax.plot(COARSENESS_SWEEP, bg, color=PALETTE[0], marker="o",
                linestyle="--", alpha=0.65, linewidth=1.8,
                label="Background bedeli")
        ax.axhline(0, color="#454247", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(COARSENESS_SWEEP)
        ax.set_xticklabels([f"{c:g}" for c in COARSENESS_SWEEP])
        ax.set_xlabel("Arka plan kabalık çarpanı")
        ax.set_title(f"{engine} motoru", fontsize=10)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Baseline'a göre ΔPSNR (dB)")
    fig.suptitle(
        f"Bit devri düğmesi: {stem} @ {args.bpp} bpp (çarpan 1 = ROI kapalı)",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.OUTPUT_DIR / f"ablation_coarseness_{stem}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"grafik: {out}")


if __name__ == "__main__":
    main()
