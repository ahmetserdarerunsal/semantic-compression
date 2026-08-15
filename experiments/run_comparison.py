# -*- coding: utf-8 -*-
"""Tam deney: iki motor x üç görüntü tipi, baseline vs semantic.

Tek komutla koşar ve outputs/ altına üretir:
- rd_<kategori>_psnr.png / rd_<kategori>_ssim.png : DCT + 4 filter bank RD eğrileri
- mask_<görüntü>.png                              : YOLO-seg maske overlay'i
- roi_<motor>_<görüntü>.png                       : aynı bpp'de baseline vs semantic
- results_full.csv                                : tüm ham sonuçlar
- summary_table.md                                : FG/BG kazanç özet tablosu

Kullanım:  python experiments/run_comparison.py  [--quick]
--quick: tek hedef bpp ve seyrek RD taramasıyla hızlı doğrulama koşusu.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import config
from src.engines import dct_engine, wavelet_engine
from src.metrics.quality import evaluate
from src.roi.bit_allocation import mask_to_block_importance, match_bpp
from src.semantic.importance_map import get_importance_mask
from src.viz.plots import (
    plot_comparison,
    plot_mask_overlay,
    plot_rd_curves,
    plot_semantic_rd,
)


def load_dataset() -> dict[str, list[tuple[str, np.ndarray]]]:
    """Kategori -> [(görüntü adı, RGB dizi), ...] eşlemesi."""
    dataset: dict[str, list[tuple[str, np.ndarray]]] = {}
    for cat in config.IMAGE_CATEGORIES:
        items = []
        for p in sorted((config.DATA_DIR / cat).glob("*.png")):
            items.append((p.stem, np.array(Image.open(p).convert("RGB"))))
        if not items:
            raise SystemExit(
                f"data/{cat} boş — önce 'python experiments/prepare_data.py' çalıştırın."
            )
        dataset[cat] = items
    return dataset


# ---------------------------------------------------------------------------
# Faz 1: uniform RD eğrileri (DCT + filter bank kıyaslaması)
# ---------------------------------------------------------------------------
def run_rd_sweep(
    dataset: dict[str, list[tuple[str, np.ndarray]]],
    quick: bool,
    writer: csv.writer,
) -> None:
    """Her kategori için DCT + tüm filter bank'lerin RD eğrilerini üretir."""
    q_sweep = config.DCT_QUALITY_SWEEP[::2] if quick else config.DCT_QUALITY_SWEEP
    s_sweep = config.WAVELET_STEP_SWEEP[::2] if quick else config.WAVELET_STEP_SWEEP

    for cat, items in dataset.items():
        psnr_curves: dict[str, list[tuple[float, float]]] = {}
        ssim_curves: dict[str, list[tuple[float, float]]] = {}

        # DCT motoru
        pts_p, pts_s = [], []
        for q in q_sweep:
            bpps, psnrs, ssims = [], [], []
            for name, img in items:
                rec, bpp = dct_engine.compress_image(img, q)
                m = evaluate(img, rec)
                bpps.append(bpp), psnrs.append(m["psnr"]), ssims.append(m["ssim"])
                writer.writerow([cat, name, "dct", "-", q, f"{bpp:.4f}",
                                 f"{m['psnr']:.3f}", f"{m['ssim']:.4f}", "", "", "", ""])
            pts_p.append((np.mean(bpps), np.mean(psnrs)))
            pts_s.append((np.mean(bpps), np.mean(ssims)))
        psnr_curves["DCT (JPEG mantığı)"] = pts_p
        ssim_curves["DCT (JPEG mantığı)"] = pts_s

        # Wavelet motoru, her filter bank için
        for wav in config.WAVELET_FILTER_BANKS:
            pts_p, pts_s = [], []
            for step in s_sweep:
                bpps, psnrs, ssims = [], [], []
                for name, img in items:
                    rec, bpp = wavelet_engine.compress_image(img, step, wav)
                    m = evaluate(img, rec)
                    bpps.append(bpp), psnrs.append(m["psnr"]), ssims.append(m["ssim"])
                    writer.writerow([cat, name, "wavelet", wav, step, f"{bpp:.4f}",
                                     f"{m['psnr']:.3f}", f"{m['ssim']:.4f}", "", "", "", ""])
                pts_p.append((np.mean(bpps), np.mean(psnrs)))
                pts_s.append((np.mean(bpps), np.mean(ssims)))
            psnr_curves[f"DWT {wav}"] = pts_p
            ssim_curves[f"DWT {wav}"] = pts_s

        plot_rd_curves(psnr_curves, config.OUTPUT_DIR / f"rd_{cat}_psnr.png",
                       f"RD eğrisi — {cat} (uniform sıkıştırma)", "PSNR (dB)")
        plot_rd_curves(ssim_curves, config.OUTPUT_DIR / f"rd_{cat}_ssim.png",
                       f"RD eğrisi — {cat} (uniform sıkıştırma)", "SSIM")
        print(f"[RD] {cat}: eğriler çizildi ({len(psnr_curves)} seri)")


# ---------------------------------------------------------------------------
# Faz 2: aynı bpp'de baseline vs semantic
# ---------------------------------------------------------------------------
def _encode_pair(
    engine: str, img: np.ndarray, mask: np.ndarray, target_bpp: float
) -> tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]] | None:
    """Bir motor için (baseline, semantic) çiftini aynı hedef bpp'ye eşler."""
    if engine == "dct":
        block_imp = mask_to_block_importance(mask)
        base = match_bpp(lambda q: dct_engine.compress_image(img, q),
                         target_bpp, 1, 100, True)
        sem = match_bpp(
            lambda q: dct_engine.compress_image(
                img, q, block_imp, config.ROI_BG_COARSENESS
            ),
            target_bpp, 1, 100, True,
        )
    else:
        wav = config.WAVELET_DEFAULT_FILTER
        base = match_bpp(lambda s: wavelet_engine.compress_image(img, s, wav),
                         target_bpp, 0.25, 512, False)
        sem = match_bpp(
            lambda s: wavelet_engine.compress_image(
                img, s, wav, importance_mask=mask,
                bg_coarseness=config.ROI_BG_COARSENESS,
            ),
            target_bpp, 0.25, 512, False,
        )
    return (base[0], base[1]), (sem[0], sem[1])


def run_semantic_comparison(
    dataset: dict[str, list[tuple[str, np.ndarray]]],
    quick: bool,
    writer: csv.writer,
) -> list[dict]:
    """Baseline vs semantic deneyini koşar; özet satırlarını döner."""
    targets = [0.40] if quick else config.TARGET_BPP_POINTS
    rows: list[dict] = []

    for cat, items in dataset.items():
        for name, img in items:
            mask, labels = get_importance_mask(img)
            if not mask.any():
                print(f"[ROI] {cat}/{name}: nesne bulunamadı, atlanıyor")
                continue
            plot_mask_overlay(img, mask, labels, config.OUTPUT_DIR / f"mask_{name}.png")

            for engine in ["dct", "wavelet"]:
                for target in targets:
                    (b_rec, b_bpp), (s_rec, s_bpp) = _encode_pair(engine, img, mask, target)
                    mb = evaluate(img, b_rec, mask)
                    ms = evaluate(img, s_rec, mask)
                    row = dict(
                        category=cat, image=name, engine=engine, target_bpp=target,
                        base_bpp=b_bpp, sem_bpp=s_bpp,
                        fg_gain_psnr=ms["fg_psnr"] - mb["fg_psnr"],
                        bg_cost_psnr=ms["bg_psnr"] - mb["bg_psnr"],
                        fg_gain_ssim=ms["fg_ssim"] - mb["fg_ssim"],
                        base=mb, sem=ms,
                    )
                    rows.append(row)
                    for tag, bpp, m in [("baseline", b_bpp, mb), ("semantic", s_bpp, ms)]:
                        writer.writerow([
                            cat, name, engine, tag, target, f"{bpp:.4f}",
                            f"{m['psnr']:.3f}", f"{m['ssim']:.4f}",
                            f"{m['fg_psnr']:.3f}", f"{m['fg_ssim']:.4f}",
                            f"{m['bg_psnr']:.3f}", f"{m['bg_ssim']:.4f}",
                        ])
                    # temsilî görsel: orta hedef bpp noktasında
                    if target == targets[len(targets) // 2]:
                        plot_comparison(
                            img, b_rec, s_rec, mask, target, mb, ms,
                            config.OUTPUT_DIR / f"roi_{engine}_{name}.png",
                            "DCT motoru" if engine == "dct" else
                            f"Wavelet motoru ({config.WAVELET_DEFAULT_FILTER})",
                        )
                print(f"[ROI] {cat}/{name} {engine}: "
                      + ", ".join(f"{r['target_bpp']:.2f}bpp FG{r['fg_gain_psnr']:+.2f}dB"
                                  for r in rows[-len(targets):]))
    return rows


def plot_semantic_rd_curves(rows: list[dict]) -> None:
    """Kategori x motor başına FG/BG PSNR vs bpp eğrileri (baseline vs semantic)."""
    for cat in config.IMAGE_CATEGORIES:
        for engine in ["dct", "wavelet"]:
            sel = [r for r in rows if r["category"] == cat and r["engine"] == engine]
            targets = sorted({r["target_bpp"] for r in sel})
            if len(targets) < 2:
                continue  # tek noktayla eğri çizilmez (--quick koşusu)
            base_fg, sem_fg, base_bg, sem_bg = [], [], [], []
            for t in targets:
                pts = [r for r in sel if r["target_bpp"] == t]
                base_fg.append((t, np.mean([p["base"]["fg_psnr"] for p in pts])))
                sem_fg.append((t, np.mean([p["sem"]["fg_psnr"] for p in pts])))
                base_bg.append((t, np.mean([p["base"]["bg_psnr"] for p in pts])))
                sem_bg.append((t, np.mean([p["sem"]["bg_psnr"] for p in pts])))
            plot_semantic_rd(
                base_fg, sem_fg, base_bg, sem_bg,
                config.OUTPUT_DIR / f"semantic_rd_{engine}_{cat}.png",
                f"Semantik RD — {cat} / {engine} (aynı bpp'de FG/BG kalitesi)",
            )
    print("[RD] semantik FG/BG eğrileri çizildi")


def write_summary(rows: list[dict]) -> None:
    """Ana çıktı tablosu: aynı bpp'de semantic'in FG kazancı (markdown)."""
    lines = [
        "# Özet: aynı bit bütçesinde semantik bit yönlendirmenin kazancı",
        "",
        "Pozitif FG kazancı = semantik yaklaşım, baseline ile AYNI bpp'de",
        "foreground'u (YOLO-seg maskesi) bu kadar daha iyi kodluyor demektir.",
        "",
        "| Kategori | Motor | Hedef bpp | FG PSNR kazancı (dB) | FG SSIM kazancı | BG PSNR bedeli (dB) |",
        "|---|---|---|---|---|---|",
    ]
    for cat in config.IMAGE_CATEGORIES:
        for engine in ["dct", "wavelet"]:
            for target in sorted({r["target_bpp"] for r in rows}):
                sel = [r for r in rows
                       if r["category"] == cat and r["engine"] == engine
                       and r["target_bpp"] == target]
                if not sel:
                    continue
                fg = [r["fg_gain_psnr"] for r in sel]
                fs = [r["fg_gain_ssim"] for r in sel]
                bg = [r["bg_cost_psnr"] for r in sel]
                lines.append(
                    f"| {cat} | {engine} | {target:.2f} "
                    f"| {np.mean(fg):+.2f} ± {np.std(fg):.2f} "
                    f"| {np.mean(fs):+.4f} ± {np.std(fs):.4f} "
                    f"| {np.mean(bg):+.2f} ± {np.std(bg):.2f} |"
                )
    path = config.OUTPUT_DIR / "summary_table.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ÖZET] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="seyrek tarama + tek hedef bpp ile hızlı koşu")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = load_dataset()
    t0 = time.time()

    with open(config.OUTPUT_DIR / "results_full.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "image", "engine", "variant/filter", "param",
                         "bpp", "psnr", "ssim", "fg_psnr", "fg_ssim",
                         "bg_psnr", "bg_ssim"])
        run_rd_sweep(dataset, args.quick, writer)
        rows = run_semantic_comparison(dataset, args.quick, writer)

    plot_semantic_rd_curves(rows)
    write_summary(rows)
    print(f"Toplam süre: {time.time()-t0:.0f} s — tüm çıktılar: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
