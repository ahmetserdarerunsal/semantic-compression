# -*- coding: utf-8 -*-
"""Laboratuvar tasarım sistemi: renk paleti + Matplotlib karanlık tema
yardımcısı. Tüm grafikler (RD eğrisi, dyadic ağaç, filtre bankası, DCT/DWT
ısı haritaları) bu tek noktadan stillendirilir — 20 satırlık stil kodunu
her plot fonksiyonunda tekrarlamamak için (mega-spec Part 67).

Bilimsel okunabilirlik önceliklidir: karanlık tema kontrastı DÜŞÜRMEZ,
yalnız arka planı UI ile tutarlı hale getirir (Part 68).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Renk paleti (mega-spec Part 20) — tek kaynak, hem CSS hem Matplotlib bunu
# kullanır (bkz. app.py APP_CSS).
# ---------------------------------------------------------------------------
BG_MAIN = "#070B14"
BG_SECONDARY = "#0B1120"
BG_PANEL = "#0F1726"
BG_ELEVATED = "#151E30"
BORDER = "#243247"
BORDER_LIGHT = "#334155"

ACCENT_CYAN = "#31C8FF"       # JPEG / DCT kimliği
ACCENT_PURPLE = "#8B5CF6"     # Wavelet / DWT kimliği
ACCENT_GREEN = "#22C55E"      # semantik ROI / pozitif doğrulama
POSITIVE = "#22C55E"
WARNING = "#F59E0B"           # trade-off (hata değil)
CRITICAL = "#EF4444"          # gerçek hatalar

TEXT_PRIMARY = "#F3F6FA"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"
TEXT_PLOT = "#CBD5E1"

# Yöntem kimlikleri (metrik kartları, grafik serileri, rozetlerde TUTARLI)
METHOD_JPEG = ACCENT_CYAN
METHOD_WAVELET = ACCENT_PURPLE
METHOD_SEMANTIC = ACCENT_GREEN
METHOD_BASELINE = TEXT_SECONDARY
METHOD_JPEG2000 = "#F0ABFC"  # ayrık, gerçek kodek için kısıtlı bir vurgu
METHOD_REAL_JPEG = "#FB923C"  # gerçek libjpeg çapraz-doğrulaması (turuncu — cyan/mor/pembe'den ayrık)


def lab_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    """Karanlık laboratuvar temalı tek eksenli bir Figure/Axes çifti döner."""
    fig, ax = plt.subplots(figsize=figsize)
    apply_lab_style(fig, ax)
    return fig, ax


def apply_lab_style(fig: plt.Figure, ax_or_axes) -> None:
    """Var olan bir Figure + (tek ya da çoklu) Axes'e karanlık tema uygular.

    Tüm plot fonksiyonları (rd_curve_figure, dyadic_tree_figure,
    filter_bank_figure, block_pipeline_figure) bunu çağırır — tek doğruluk
    kaynağı, dağınık kopya-yapıştır stil kodu yerine.
    """
    fig.patch.set_facecolor(BG_PANEL)
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(TEXT_SECONDARY)
        ax.yaxis.label.set_color(TEXT_SECONDARY)
        ax.title.set_color(TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_color(BORDER_LIGHT)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)


def style_legend(ax: plt.Axes, **kwargs) -> None:
    leg = ax.legend(frameon=False, labelcolor=TEXT_PLOT, fontsize=8, **kwargs)
    if leg is not None:
        leg.get_frame().set_facecolor(BG_PANEL)
