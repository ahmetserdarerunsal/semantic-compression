# -*- coding: utf-8 -*-
"""İnteraktif Rate–Distortion grafiği (Plotly) — gerçek zamanlı hover
tooltip'leri için. Matplotlib statik bir görüntüye render edildiğinden
tarayıcıda hover sağlayamaz; Gradio'nun gr.Plot bileşeni Plotly Figure'ları
NATİF olarak (tam JS etkileşimiyle) destekler, bu yüzden yalnız BU grafik
için Plotly kullanılır — geri kalan tüm görselleştirmeler (DCT/DWT ısı
haritaları, dyadic ağaç, filtre bankası) hâlâ Matplotlib'tir.

KRİTİK: Hiçbir tooltip değeri uydurulmaz. Her nokta src/compare.rd_sweep
tarafından GERÇEKTEN ölçülmüş (bpp, psnr, ssim, sıkıştırma oranı, motora
özgü parametre) değerleri taşır; eksik alan varsa gösterilmez.
"""
from __future__ import annotations

import plotly.graph_objects as go

from src.viz.style import (ACCENT_CYAN, ACCENT_PURPLE, BG_PANEL, BORDER,
                           METHOD_JPEG2000, TEXT_PLOT, TEXT_PRIMARY,
                           TEXT_SECONDARY)

_METHOD_COLORS = [ACCENT_CYAN, ACCENT_PURPLE, METHOD_JPEG2000, "#F59E0B"]


def _is_jpeg2000(label: str) -> bool:
    """Ana Compare RD grafiğindeki ikinci seri artık gerçek JPEG2000/OpenJPEG
    (mega-spec "ANA COMPARE TERMİNOLOJİ GÜNCELLEMESİ") — bu yüzden burada
    "Wavelet" değil "JPEG2000" aranır; extra alanları da buna göre (aşağıda)
    JPEG2000'in gerçekten taşıdığı size_bytes'a göre gösterilir."""
    return "JPEG2000" in label


def _fmt(v, decimals=2, suffix="") -> str:
    return f"{v:.{decimals}f}{suffix}" if v is not None else "—"


def _hover_html(label: str, p: dict, *, is_current: bool = False) -> str:
    """Kompakt tooltip metni: Yöntem/BPP/PSNR birincil; SSIM/parametre
    ikincil. mega-spec: sadece GERÇEKTEN ölçülmüş alanlar gösterilir."""
    lines = [f"<b>{label}</b>"]
    lines.append(f"BPP: {_fmt(p['bpp'], 3)}")
    lines.append(f"PSNR: {_fmt(p['psnr'], 2, ' dB')}")
    if p.get("ssim") is not None:
        lines.append(f"SSIM: {_fmt(p['ssim'], 4)}")
    if p.get("compression_ratio") is not None:
        lines.append(f"Sıkıştırma Oranı: {_fmt(p['compression_ratio'], 2)} : 1")
    extra = p.get("extra") or {}
    if _is_jpeg2000(label):
        if "size_bytes" in extra:
            lines.append(f"Boyut: {extra['size_bytes']} bayt")
    else:
        if "quality" in extra:
            lines.append(f"Quality: {extra['quality']}")
        if "block_size" in extra:
            lines.append(f"Blok: {extra['block_size']}×{extra['block_size']}")
    if is_current:
        lines.insert(1, "<b>— GÜNCEL HEDEF —</b>")
        lines.append(f"Hedef BPP: {_fmt(p.get('target_bpp'), 3)}")
    return "<br>".join(lines)


def rd_interactive_figure(
    curves: dict[str, list[dict]], target_bpp: float,
    current: dict[str, dict] | None = None,
) -> go.Figure:
    """curves: {yöntem_etiketi: [nokta_sözlüğü, ...]} — src.compare.rd_sweep
    çıktısı (gerçek ölçümler, sıralı). current: {yöntem_etiketi: nokta} —
    güncel hedefteki gerçek çalışma noktaları (yıldızla vurgulanır)."""
    fig = go.Figure()

    for i, (label, pts) in enumerate(curves.items()):
        if not pts:
            continue
        color = _METHOD_COLORS[i % len(_METHOD_COLORS)]
        xs = [p["bpp"] for p in pts]
        ys = [p["psnr"] for p in pts]

        # Görünen çizgi + küçük belirteçler (hover KAPALI — asıl hover
        # aşağıdaki daha büyük görünmez katmandan gelir)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8, line=dict(color=BG_PANEL, width=1)),
            hoverinfo="skip",
        ))

        # Görünmez, GENİŞ isabet alanı — küçük belirteçleri büyütmeden
        # hover'ı kolaylaştırır (mega-spec: "slightly larger invisible
        # hit area around each measured marker").
        hovertext = [_hover_html(label, p) for p in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", showlegend=False,
            marker=dict(size=26, color=color, opacity=0.001),
            hovertext=hovertext, hoverinfo="text",
        ))

    # Güncel hedef: yıldız işaretleyici + zenginleştirilmiş tooltip
    if current:
        for i, (label, p) in enumerate(current.items()):
            color = _METHOD_COLORS[i % len(_METHOD_COLORS)]
            fig.add_trace(go.Scatter(
                x=[p["bpp"]], y=[p["psnr"]], mode="markers", showlegend=False,
                marker=dict(symbol="star", size=18, color=color,
                           line=dict(color=TEXT_PRIMARY, width=1.2)),
                hovertext=[_hover_html(f"{label} (güncel hedef)", p, is_current=True)],
                hoverinfo="text",
            ))

    fig.add_vline(x=target_bpp, line_width=1, line_dash="dot", line_color=TEXT_SECONDARY)
    fig.add_annotation(x=target_bpp, y=0, yref="paper", yanchor="bottom",
                       text=f"HEDEF {target_bpp:.2f} bpp", showarrow=False,
                       font=dict(size=10, color=TEXT_SECONDARY), xanchor="left",
                       xshift=4, yshift=4)

    fig.update_layout(
        paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PLOT, family="Inter, ui-sans-serif, system-ui, sans-serif", size=12),
        xaxis=dict(title="bpp (bit / piksel)", gridcolor=BORDER, zeroline=False,
                  color=TEXT_SECONDARY, linecolor=BORDER,
                  showspikes=True, spikemode="across", spikesnap="cursor",
                  spikecolor=TEXT_SECONDARY, spikethickness=1, spikedash="dot"),
        yaxis=dict(title="PSNR (dB)", gridcolor=BORDER, zeroline=False,
                  color=TEXT_SECONDARY, linecolor=BORDER,
                  showspikes=True, spikemode="across", spikesnap="cursor",
                  spikecolor=TEXT_SECONDARY, spikethickness=1, spikedash="dot"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                   font=dict(color=TEXT_PLOT), orientation="h", y=1.1, x=0),
        margin=dict(l=55, r=20, t=30, b=45),
        height=420,
        hovermode="closest",
        hoverdistance=30,
        hoverlabel=dict(bgcolor=BG_PANEL, bordercolor=BORDER,
                        font=dict(color=TEXT_PRIMARY, size=11.5)),
    )
    return fig
