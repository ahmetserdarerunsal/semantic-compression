# -*- coding: utf-8 -*-
"""Görselleştirme: maske overlay, baseline-vs-semantic yan yana, RD eğrileri.

Renkler CVD-güvenli Okabe-Ito paletinden sabit sırayla atanır; seriler ayrıca
marker şekliyle de ayrışır (renk körlüğü için ikincil kodlama).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sabit kategorik palet + marker eşlemesi (sıra asla döngüye sokulmaz)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Önem maskesini görüntü üzerine yarı saydam renkle bindirir."""
    out = image.astype(np.float64).copy()
    color = np.array([0, 114, 178], dtype=np.float64)  # palet mavisi
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color
    return out.astype(np.uint8)


def plot_mask_overlay(
    image: np.ndarray, mask: np.ndarray, labels: list[str], out_path: Path
) -> None:
    """Orijinal | maske | overlay üçlüsünü kaydeder."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(image)
    axes[0].set_title("Orijinal")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("YOLO-seg önem maskesi")
    axes[2].imshow(overlay_mask(image, mask))
    axes[2].set_title(f"Overlay ({', '.join(sorted(set(labels))) or 'nesne yok'})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _mask_bbox(mask: np.ndarray, pad: int = 12) -> tuple[int, int, int, int]:
    """Maskenin sınırlayıcı kutusu (y0, y1, x0, x1), kenar payıyla."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (
        max(int(ys.min()) - pad, 0),
        min(int(ys.max()) + pad, h),
        max(int(xs.min()) - pad, 0),
        min(int(xs.max()) + pad, w),
    )


def plot_comparison(
    original: np.ndarray,
    baseline: np.ndarray,
    semantic: np.ndarray,
    mask: np.ndarray,
    bpp: float,
    metrics_base: dict[str, float],
    metrics_sem: dict[str, float],
    out_path: Path,
    engine_name: str,
) -> None:
    """Aynı bpp'de baseline vs semantic: tam kare + foreground yakınlaştırma."""
    y0, y1, x0, x1 = _mask_bbox(mask)
    panels = [
        ("Orijinal", original, None),
        (f"Baseline (uniform) — {bpp:.2f} bpp", baseline, metrics_base),
        (f"Semantic (ROI) — {bpp:.2f} bpp", semantic, metrics_sem),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for col, (title, img, m) in enumerate(panels):
        axes[0, col].imshow(img)
        sub = title
        if m is not None:
            sub += f"\nFG PSNR {m['fg_psnr']:.2f} dB · BG PSNR {m['bg_psnr']:.2f} dB"
        axes[0, col].set_title(sub, fontsize=9)
        axes[1, col].imshow(img[y0:y1, x0:x1])
        axes[1, col].set_title("foreground yakınlaştırma", fontsize=8)
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(
        f"{engine_name}: aynı bit bütçesinde uniform vs semantik bit yönlendirme",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_rd(
    base_fg: list[tuple[float, float]],
    sem_fg: list[tuple[float, float]],
    base_bg: list[tuple[float, float]],
    sem_bg: list[tuple[float, float]],
    out_path: Path,
    title: str,
) -> None:
    """Semantik RD eğrisi: FG/BG PSNR vs bpp, baseline ve semantic birlikte.

    Renk = yöntem (baseline mavi, semantic turuncu); çizgi stili = bölge
    (düz = foreground, kesikli = background).
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    series = [
        ("Baseline FG", base_fg, PALETTE[0], "-", "o"),
        ("Semantic FG", sem_fg, PALETTE[1], "-", "s"),
        ("Baseline BG", base_bg, PALETTE[0], "--", "o"),
        ("Semantic BG", sem_bg, PALETTE[1], "--", "s"),
    ]
    for label, pts, color, ls, mk in series:
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                linestyle=ls, marker=mk, markersize=5, linewidth=1.8,
                label=label, alpha=1.0 if ls == "-" else 0.65)
    ax.set_xlabel("bpp (bit / piksel)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def rd_curve_figure(
    curves: dict[str, list[tuple[float, float]]],
    title: str,
    ylabel: str = "PSNR (dB)",
    highlight: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    """plot_rd_curves ile aynı görsel dil, ama dosyaya yazmak yerine
    interaktif UI (gr.Plot) için bir Figure döner; isteğe bağlı olarak
    güncel çalışma noktasını (hedef bpp'de gerçekleşen bpp/PSNR) yıldızla
    işaretler — slider ile teori arasındaki bağı doğrudan gösterir."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for slot, (label, points) in enumerate(curves.items()):
        pts = sorted(points)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        color = PALETTE[slot % len(PALETTE)]
        marker = MARKERS[slot % len(MARKERS)]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5,
                linewidth=1.8, label=label)
        if highlight and label in highlight:
            hx, hy = highlight[label]
            ax.scatter([hx], [hy], color=color, marker="*", s=220,
                       edgecolor="black", linewidth=0.8, zorder=5)
    ax.set_xlabel("bpp (bit / piksel)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_rd_curves(
    curves: dict[str, list[tuple[float, float]]],
    out_path: Path,
    title: str,
    ylabel: str = "PSNR (dB)",
) -> None:
    """RD eğrileri: {seri adı: [(bpp, kalite), ...]}. Sıra sabit palete eşlenir."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    slot = 0  # palet asla döngüye sokulmaz; DCT referansı palet dışıdır
    for label, points in curves.items():
        pts = sorted(points)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if label.startswith("DCT"):
            # Referans seri: kesikli koyu gri (wavelet ailesinden ayrışsın)
            ax.plot(xs, ys, color="#454247", linestyle="--", marker="x",
                    markersize=6, linewidth=1.6, label=label)
        else:
            ax.plot(xs, ys, color=PALETTE[slot], marker=MARKERS[slot],
                    markersize=5, linewidth=1.8, label=label)
            slot += 1
    ax.set_xlabel("bpp (bit / piksel)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
