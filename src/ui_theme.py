# -*- coding: utf-8 -*-
"""Laboratuvar tasarım sistemi: Gradio theme (desteklenen theming API'si
üzerinden — buton/slider/panel renkleri) + yalnız kart bileşenleri için
gerekli özel CSS (mega-spec Part 64: "prefer theme configuration", "avoid
fragile deeply-nested selectors").

Renk sabitleri src/viz/style.py ile birebir aynıdır (grafikler ve arayüz
aynı paletten beslenir).
"""
from __future__ import annotations

import gradio as gr

from src.viz.style import (ACCENT_CYAN, BG_ELEVATED, BG_MAIN, BG_PANEL,
                           BG_SECONDARY, BORDER, BORDER_LIGHT, CRITICAL,
                           POSITIVE, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY,
                           WARNING)

_cyan = gr.themes.Color(
    c50="#E6F9FF", c100="#B8EFFF", c200="#8AE5FF", c300="#5CDBFF",
    c400="#31C8FF", c500="#31C8FF", c600="#1FA8DE", c700="#1585B3",
    c800="#0D6488", c900="#08415C", c950="#052A3D",
)
_slate = gr.themes.Color(
    c50="#F3F6FA", c100="#CBD5E1", c200="#94A3B8", c300="#64748B",
    c400="#475569", c500="#334155", c600="#243247", c700="#151E30",
    c800="#0F1726", c900="#0B1120", c950="#070B14",
)

LAB_THEME = gr.themes.Base(
    primary_hue=_cyan, secondary_hue=_cyan, neutral_hue=_slate,
    # text_size varsayılanı (text_md: 14/16/22px) tüm native bileşenleri
    # (buton, slider etiketi, dropdown, checkbox) bir kademe küçük
    # gösteriyordu — "%80 zoom" hissinin asıl kaynaklarından biri buydu.
    # text_lg (16/20/24px) tek satırla TÜM native metinleri global olarak
    # büyütür (mega-spec Part 14: "consistent scale system", tek tek değil).
    text_size=gr.themes.sizes.text_lg,
    font=("Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"),
    font_mono=("SFMono-Regular", "Consolas", "Liberation Mono", "monospace"),
).set(
    body_background_fill=BG_MAIN, body_background_fill_dark=BG_MAIN,
    background_fill_primary=BG_PANEL, background_fill_primary_dark=BG_PANEL,
    background_fill_secondary=BG_SECONDARY, background_fill_secondary_dark=BG_SECONDARY,
    block_background_fill=BG_PANEL, block_background_fill_dark=BG_PANEL,
    block_border_color=BORDER, block_border_color_dark=BORDER,
    block_label_background_fill=BG_ELEVATED, block_label_background_fill_dark=BG_ELEVATED,
    block_label_text_color=TEXT_SECONDARY, block_label_text_color_dark=TEXT_SECONDARY,
    block_title_text_color=TEXT_PRIMARY, block_title_text_color_dark=TEXT_PRIMARY,
    panel_background_fill=BG_PANEL, panel_background_fill_dark=BG_PANEL,
    panel_border_color=BORDER, panel_border_color_dark=BORDER,
    body_text_color=TEXT_PRIMARY, body_text_color_dark=TEXT_PRIMARY,
    body_text_color_subdued=TEXT_SECONDARY, body_text_color_subdued_dark=TEXT_SECONDARY,
    border_color_primary=BORDER, border_color_primary_dark=BORDER,
    input_background_fill=BG_ELEVATED, input_background_fill_dark=BG_ELEVATED,
    input_border_color=BORDER_LIGHT, input_border_color_dark=BORDER_LIGHT,
    input_border_color_focus=ACCENT_CYAN, input_border_color_focus_dark=ACCENT_CYAN,
    slider_color=ACCENT_CYAN, slider_color_dark=ACCENT_CYAN,
    button_primary_background_fill=ACCENT_CYAN, button_primary_background_fill_dark=ACCENT_CYAN,
    button_primary_background_fill_hover="#5CDBFF", button_primary_background_fill_hover_dark="#5CDBFF",
    button_primary_text_color=BG_MAIN, button_primary_text_color_dark=BG_MAIN,
    button_secondary_background_fill=BG_ELEVATED, button_secondary_background_fill_dark=BG_ELEVATED,
    button_secondary_border_color=BORDER_LIGHT, button_secondary_border_color_dark=BORDER_LIGHT,
    button_secondary_text_color=TEXT_PRIMARY, button_secondary_text_color_dark=TEXT_PRIMARY,
    # Birincil eylem (KARŞILAŞTIR/BLOĞU İNCELE) ikincilden (Örnek Seç/
    # Değiştir) görsel olarak daha güçlü olmalı (Part 10) — kalın metin +
    # rahat yükseklik. button_small_* de büyütülür (size="sm" hâlâ okunur
    # kalsın diye — Değiştir/kategori butonları).
    button_large_text_weight="700", button_large_padding="14px 22px",
    button_small_text_weight="600", button_small_padding="10px 16px",
    checkbox_background_color=BG_ELEVATED, checkbox_background_color_dark=BG_ELEVATED,
    checkbox_background_color_selected=ACCENT_CYAN, checkbox_background_color_selected_dark=ACCENT_CYAN,
    checkbox_border_color=BORDER_LIGHT, checkbox_border_color_dark=BORDER_LIGHT,
    table_border_color=BORDER, table_border_color_dark=BORDER,
    table_even_background_fill=BG_PANEL, table_even_background_fill_dark=BG_PANEL,
    table_odd_background_fill=BG_ELEVATED, table_odd_background_fill_dark=BG_ELEVATED,
    error_background_fill=BG_PANEL, error_background_fill_dark=BG_PANEL,
    error_border_color=CRITICAL, error_border_color_dark=CRITICAL,
    shadow_drop="none", shadow_drop_lg="0 2px 8px rgba(0,0,0,0.25)",
    block_shadow="none", block_shadow_dark="none",
    # YAPISAL DÜZELTME: Gradio varsayılanında HER Group/Row/Column bir
    # ".block" olarak render edilir ve temanın block_border_color'ıyla
    # otomatik kenarlık alır — önceki tasarımın "kutu içinde kutu" (Gradio
    # demo) hissinin asıl kaynağı buydu. Kenarlığı global olarak SIFIRLAYIP
    # yalnız gerçekten gruplama gerektiren yerlerde (görüntü çerçeveleri,
    # quality-strip, seçili kartlar) özel CSS sınıflarıyla geri ekliyoruz.
    block_border_width="0px", block_border_width_dark="0px",
    panel_border_width="0px", panel_border_width_dark="0px",
)

# Sayfayı Gradio'nun kendi desteklediği ?__theme=dark URL parametresiyle
# karanlık moda zorlar — "büyük/kırılgan JS hack" değil, çerçevenin resmi
# mekanizması (mega-spec Part 66: dekoratif efekt değil, tasarım kimliğinin
# gerektirdiği tek işlevsel adım).
FORCE_DARK_JS = """
function() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

LAB_CSS = f"""
:root {{
  --lab-bg-main: {BG_MAIN};
  --lab-bg-secondary: {BG_SECONDARY};
  --lab-bg-panel: {BG_PANEL};
  --lab-bg-elevated: {BG_ELEVATED};
  --lab-border: {BORDER};
  --lab-border-light: {BORDER_LIGHT};
  --lab-text-primary: {TEXT_PRIMARY};
  --lab-text-secondary: {TEXT_SECONDARY};
  --lab-text-muted: {TEXT_MUTED};
  --lab-accent: {ACCENT_CYAN};
  --lab-positive: {POSITIVE};
  --lab-warning: {WARNING};
  --lab-critical: {CRITICAL};
  --lab-sans: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
  --lab-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;

  /* ---- Global tipografi ölçeği (mega-spec "FINAL STATE/CALLBACK/
     TYPOGRAPHY FIX" Part 23) — tek kaynak; her kural aşağıda bunlardan
     birini kullanır, rastgele px atanmaz.
     Main title 28-32 / Page section 22-26 / Panel title 18-20 /
     Body 15-16 / Secondary 13-14 / Metric major 26-32 / Metric secondary
     18-22 — bu değişkenler bu hedeflerle eşlenir (bkz. her değişkenin
     kullanım yeri için aşağıdaki .section-label/.page-section-label/
     .metric-value/.frame-caption kuralları). ---- */
  --text-2xs: 11px;    /* yalnız rozet / mikro büyük-harf etiket (gövde metni DEĞİL) */
  --text-xs: 13px;     /* ikincil metin alt sınırı */
  --text-sm: 14px;     /* ikincil metin üst sınırı / açıklama */
  --text-md: 15.5px;   /* gövde metni, giriş/etiket varsayılanı */
  --text-lg: 19px;     /* metrik ikincil değerler (panel navigasyonundan AYRIK — tab-nav kendi 17px'ini kullanır) */
  --text-xl: 27px;     /* metrik ana (birincil) değerler */
  --text-2xl: 24px;    /* boş-durum başlığı */
  --text-3xl: 30px;    /* ürün ana başlığı */
}}

.gradio-container {{
  max-width: 1650px !important;
  font-family: var(--lab-sans) !important;
  background: var(--lab-bg-main) !important;
}}

/* Gradio marka altbilgisini gizle (Part 24 minimal chrome) */
footer {{ display: none !important; }}
.gradio-container .prose a[href*="gradio.app"] {{ display: none !important; }}

/* ---------------- Header (2 satır) ---------------- */
.lab-header {{
  padding: 18px 4px 18px 4px;
  border-bottom: 1px solid var(--lab-border);
  margin-bottom: 4px;
}}
.lab-header .lab-title {{
  font-size: var(--text-3xl); font-weight: 700; letter-spacing: 0.01em;
  line-height: 1.2; color: var(--lab-text-primary); margin: 0;
}}
.lab-header .lab-subtitle {{
  font-size: var(--text-sm); font-weight: 500; color: var(--lab-text-secondary);
  margin-top: 7px;
}}

/* ---------------- Tabs (ANA NAVİGASYON — tüm sayfalarda) ----------------
   Önceki sürüm 13px/10px padding ile ikincil bir etiket gibi görünüyordu;
   bu artık uygulamanın BİRİNCİL gezinmesi olarak okunmalı (Part 4/27). */
.tab-nav {{ border-bottom: 1px solid var(--lab-border) !important; gap: 30px; padding: 0 2px; }}
.tab-nav button {{
  color: var(--lab-text-muted) !important; font-weight: 600 !important;
  font-size: 17px !important; letter-spacing: 0.01em !important;
  border: none !important; border-bottom: 3px solid transparent !important;
  background: transparent !important; padding: 13px 2px !important;
  height: 50px !important; text-transform: uppercase;
}}
.tab-nav button.selected {{
  color: var(--lab-text-primary) !important;
  border-bottom: 3px solid var(--lab-accent) !important;
  background: transparent !important;
}}

/* ---------------- Section labels — İKİ kademe (mega-spec Part 23):
   .page-section-label = sayfa düzeyi bölüm başlığı (SAME-RATE COMPARISON,
   QUALITY COMPARISON, DCT TRANSFORM EXPLORER, ...) 22-26px;
   .section-label = panel/alt-panel başlığı (KAYNAK GÖRÜNTÜ, BLOK AYARLARI,
   ROI SEÇİMİ, ...) 18-20px. Küçük metrik etiketleriyle KARIŞTIRILMAZ, onlar
   kendi sınıflarını kullanır: .metric-label, .measurement-label, ... */
.page-section-label {{
  font-size: 23px; font-weight: 700; letter-spacing: 0.02em;
  text-transform: uppercase; color: var(--lab-text-primary);
  margin-bottom: 12px;
}}
.section-label {{
  font-size: 19px; font-weight: 700; letter-spacing: 0.03em;
  text-transform: uppercase; color: var(--lab-text-primary);
  margin-bottom: 10px;
}}
/* Görüntü çerçevelerinin ÜSTÜNDEKİ küçük ortalanmış etiketler (ORİJİNAL,
   JPEG / DCT, JPEG2000 / DWT, ROI MASKESİ, ...) — bir sayfa/panel başlığı
   değil, kompakt bir çerçeve altyazısıdır; kendi (daha küçük) kademesi. */
.frame-caption {{
  font-size: 14px; font-weight: 700; letter-spacing: 0.03em;
  text-transform: uppercase; color: var(--lab-text-primary);
  margin-bottom: 8px;
}}

/* ---------------- Cards ---------------- */
.lab-card {{
  background: var(--lab-bg-panel);
  border: 1px solid var(--lab-border);
  border-radius: 11px;
  padding: 16px 18px;
  margin-bottom: 4px;
}}

.metric-card-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
.method-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.metric-title {{ font-size: var(--text-md); font-weight: 700; color: var(--lab-text-primary); flex-grow: 1; }}
.size-badge {{
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--lab-text-muted); border: 1px solid var(--lab-border-light);
  border-radius: 5px; padding: 2px 7px;
}}
.metric-param {{ font-size: var(--text-xs); color: var(--lab-text-muted); margin-bottom: 10px; }}

.metric-row {{
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 7px 0; border-top: 1px solid var(--lab-border);
}}
.metric-row:first-of-type {{ border-top: none; }}
.metric-label {{ font-size: var(--text-xs); color: var(--lab-text-secondary); letter-spacing: 0.02em; }}
.metric-value-wrap {{ display: flex; align-items: baseline; gap: 8px; }}
.metric-value {{ font-family: var(--lab-mono); font-size: var(--text-xl); font-weight: 600; color: var(--lab-text-primary); }}
.metric-value.mono {{ font-family: var(--lab-mono); }}
.metric-delta {{ font-family: var(--lab-mono); font-size: var(--text-sm); }}

/* ---------------- Target / RD summary card ---------------- */
.target-card {{ text-align: left; }}
.target-value-hero {{
  font-family: var(--lab-mono); font-size: 32px; font-weight: 700;
  color: var(--lab-accent); margin: 4px 0 6px 0;
}}
.target-value-hero .unit {{ font-size: var(--text-sm); color: var(--lab-text-secondary); }}
.target-caption {{ font-size: var(--text-xs); color: var(--lab-text-muted); margin-bottom: 12px; line-height: 1.5; }}
.target-row {{ display: flex; align-items: center; gap: 8px; padding: 6px 0; border-top: 1px solid var(--lab-border); }}
.target-row:first-of-type {{ border-top: none; }}
.target-label {{ font-size: var(--text-sm); color: var(--lab-text-secondary); flex-grow: 1; }}
.target-value {{ font-family: var(--lab-mono); font-size: var(--text-md); color: var(--lab-text-primary); }}
.target-diff .target-value {{ color: var(--lab-text-muted); }}

/* ---------------- Status badge ---------------- */
.status-badge {{
  display: inline-block; font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.06em;
  border: 1px solid; border-radius: 6px; padding: 4px 11px; text-transform: uppercase;
}}
.validation-head {{ margin-bottom: 10px; }}

/* ---------------- Sparsity card ---------------- */
.sparsity-compare {{ display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }}
.sparsity-col {{ flex: 1; text-align: center; }}
.sparsity-col-title {{ font-size: var(--text-2xs); color: var(--lab-text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 5px; }}
.sparsity-big {{ font-family: var(--lab-mono); font-size: 27px; font-weight: 700; color: var(--lab-text-primary); }}
.sparsity-arrow {{ color: var(--lab-text-muted); font-size: var(--text-lg); }}

/* ---------------- Trade-off / same-budget cards ---------------- */
.tradeoff-row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: var(--text-sm); }}
.tradeoff-delta {{ font-family: var(--lab-mono); font-size: var(--text-xl); font-weight: 700; margin-top: 7px; text-align: right; }}
.tradeoff-metric {{ padding: 9px 0; border-top: 1px solid var(--lab-border); }}
.tradeoff-metric:first-of-type {{ border-top: none; padding-top: 0; }}
.budget-row {{ display: flex; gap: 16px; }}
.budget-col {{ flex: 1; text-align: center; }}

/* ---------------- Empty state (küçük kart içi — DCT/DWT Lab "önce görüntü
   seçin" vb; TAM SAYFA boş durum için .hero-empty kullanılır) ---------- */
.empty-state {{ text-align: center; padding: 38px 18px; border-style: dashed; }}
.empty-title {{ font-size: var(--text-md); font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; color: var(--lab-text-secondary); }}
.empty-caption {{ font-size: var(--text-sm); color: var(--lab-text-muted); margin-top: 8px; line-height: 1.5; }}

/* ---------------- Summary bar (DCT/DWT Lab) ---------------- */
.summary-bar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }}
.summary-chip {{
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border);
  border-radius: 8px; padding: 8px 14px; min-width: 96px;
}}
.summary-chip-label {{ display: block; font-size: var(--text-2xs); color: var(--lab-text-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
.summary-chip-value {{ display: block; font-size: var(--text-lg); font-weight: 700; color: var(--lab-text-primary); margin-top: 3px; }}

/* ---------------- Compact target-rate hero (Level-1 view) ---------------- */
.target-rate-header {{ display: flex; align-items: baseline; gap: 10px; padding: 4px 0 12px 0; }}
.target-rate-label {{ font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--lab-text-muted); }}
.target-rate-value {{ font-family: var(--lab-mono); font-size: 28px; font-weight: 700; color: var(--lab-accent); }}
.target-rate-value .unit {{ font-size: var(--text-sm); color: var(--lab-text-secondary); font-weight: 400; }}

/* ---------------- Category badge (image picker) ---------------- */
.category-badge {{
  display: inline-block; font-size: var(--text-xs); font-weight: 600; color: var(--lab-text-secondary);
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 6px; padding: 4px 10px; margin-top: 5px;
}}

/* ---------------- Ölçüm şeridi (DCT Lab — kart yığını yerine tek şerit,
   mega-spec Part 16: "should look like a scientific instrument readout") */
.measurement-strip {{
  display: flex; flex-wrap: wrap; border: 1px solid var(--lab-border);
  border-radius: 8px; overflow: hidden; margin: 14px 0;
}}
.measurement-item {{
  flex: 1; min-width: 120px; padding: 12px 18px;
  border-right: 1px solid var(--lab-border);
}}
.measurement-item:last-child {{ border-right: none; }}
.measurement-label {{
  display: block; font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--lab-text-muted);
}}
.measurement-value {{
  display: block; font-family: var(--lab-mono); font-size: var(--text-xl); font-weight: 700;
  color: var(--lab-text-primary); margin-top: 4px;
}}

/* ---------------- Kaynak görüntü bölümü (DCT/DWT Lab) ---------------- */
.source-image-frame img {{
  object-fit: contain !important; max-height: 420px;
}}
.selected-block-line {{
  font-family: var(--lab-mono); font-size: var(--text-sm); color: var(--lab-text-secondary);
  margin: 8px 0 0 0;
}}
.selected-block-line .sb-label {{
  font-family: var(--lab-sans); font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--lab-text-muted); margin-right: 8px;
}}

/* ---------------- Theory tab grid ---------------- */
.theory-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
@media (max-width: 900px) {{ .theory-grid {{ grid-template-columns: 1fr; }} }}

/* ---------------- Method-colored image frames ---------------- */
.frame-jpeg {{ border: 1.5px solid {ACCENT_CYAN} !important; border-radius: 8px; }}
.frame-wavelet {{ border: 1.5px solid #8B5CF6 !important; border-radius: 8px; }}
.frame-jp2k {{ border: 1.5px solid #F0ABFC !important; border-radius: 8px; }}
.frame-real-jpeg {{ border: 1.5px solid #FB923C !important; border-radius: 8px; }}
.frame-neutral {{ border: 1px solid var(--lab-border-light) !important; border-radius: 8px; }}

/* ============================================================
   YAPISAL: boş durum / deney çubuğu / görüntü çipi / kalite şeridi
   (mega-spec "structural redesign" — kart-içinde-kart yerine)
   ============================================================ */

/* ---------------- Boş durum (Karşılaştır ilk açılış) ----------------
   İçerik dar tutulur (520-620px) ki geniş iş alanında "dev dikdörtgen +
   ortada minik yazı" hissi oluşmasın; üstte biraz daha fazla boşlukla
   geometrik merkezin az üstünde dursun (Part 7/8). */
.hero-empty {{
  text-align: center;
  padding: 84px 24px 64px 24px;
  max-width: 600px;
  margin: 0 auto;
}}
.hero-empty .hero-title {{
  font-size: var(--text-2xl); font-weight: 700; letter-spacing: 0.01em;
  color: var(--lab-text-primary); margin-bottom: 14px;
}}
.hero-empty .hero-caption {{
  font-size: var(--text-md); color: var(--lab-text-secondary); line-height: 1.55;
  margin-bottom: 8px;
}}

/* ---------------- Deney çubuğu (yatay toolbar) ----------------
   NOT: "display" BİLEREK burada tanımlanmaz. gr.Row zaten varsayılan
   flex'tir; bu sınıf "display:flex" eklerse, Gradio'nun visible=False
   durumunu gizlemek için kullandığı eşit-özgüllükteki kendi CSS kuralını
   ezer (bu tam olarak yaşanan gerçek bir hataydı: deney çubuğu görüntü
   seçilmeden ÖNCE de görünür kalıyordu). Yalnız boşluk/kenarlık eklenir. */
.experiment-bar {{
  align-items: center; flex-wrap: wrap; gap: 22px;
  padding: 14px 0 20px 0; border-bottom: 1px solid var(--lab-border);
  margin-bottom: 22px;
}}
.experiment-bar .bar-group {{ display: flex; flex-direction: column; gap: 3px; }}
.experiment-bar .bar-label {{
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--lab-text-muted);
}}

/* ---------------- Görüntü çipi (seçili örnek/yüklenen görüntü) ------- */
.image-chip {{
  display: flex; align-items: center; gap: 9px;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 8px; padding: 8px 14px;
}}
.image-chip .chip-name {{ font-size: var(--text-md); font-weight: 600; color: var(--lab-text-primary); }}
.image-chip .chip-category {{
  font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--lab-accent);
}}

/* ---------------- Oran adaleti başlığı (TARGET/JPEG/JPEG2000/Δ) ------- */
.rate-fairness {{ display: flex; gap: 30px; align-items: baseline; padding: 4px 0 6px 0; }}
.rate-fairness .rf-item {{ display: flex; flex-direction: column; gap: 2px; }}
.rate-fairness .rf-label {{ font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--lab-text-muted); }}
.rate-fairness .rf-value {{ font-family: var(--lab-mono); font-size: var(--text-lg); font-weight: 700; color: var(--lab-text-primary); }}
.rate-fairness .rf-value.accent-jpeg {{ color: {ACCENT_CYAN}; }}
.rate-fairness .rf-value.accent-wavelet {{ color: #8B5CF6; }}

/* ---------------- Kalite karşılaştırma şeridi (kart yerine) ---------- */
.quality-strip {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
.quality-strip th {{
  text-align: right; font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--lab-text-muted); padding: 0 0 10px 0;
}}
.quality-strip th:first-child, .quality-strip td:first-child {{ text-align: left; }}
.quality-strip th.th-jpeg {{ color: {ACCENT_CYAN}; }}
.quality-strip th.th-wavelet {{ color: #8B5CF6; }}
.quality-strip td {{
  font-family: var(--lab-mono); font-size: var(--text-lg); text-align: right;
  padding: 10px 0; border-top: 1px solid var(--lab-border); color: var(--lab-text-primary);
}}
.quality-strip td:first-child {{
  font-family: var(--lab-sans); font-size: var(--text-sm); color: var(--lab-text-secondary);
  text-align: left; text-transform: uppercase; letter-spacing: 0.03em;
}}
.quality-strip td.delta {{ font-size: var(--text-sm); color: var(--lab-text-muted); }}

/* ---------------- Filtre bankası akış şeması (DWT Lab, özgün) --------
   CSS-only hover tooltip (JS yok): .flow-tip varsayılan gizli, hover'da
   görünür. mega-spec Part 13/22: "hover edilen aşama kısa açıklama versin",
   100-180ms subtle transition. */
.filter-flow {{ display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px 0; }}
.flow-row {{ display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }}
.flow-row-4 {{ gap: 10px; }}
.flow-arrow {{ color: var(--lab-text-muted); font-size: 15px; line-height: 1; }}
.flow-node {{
  position: relative; background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 7px; padding: 8px 14px; font-size: var(--text-sm); color: var(--lab-text-primary);
  cursor: default; transition: border-color 140ms ease, background 140ms ease;
}}
.flow-node:hover {{ border-color: var(--lab-accent); background: var(--lab-bg-panel); }}
.flow-node.flow-lo, .flow-node.flow-ll {{ border-left: 3px solid {ACCENT_CYAN}; }}
.flow-node.flow-hi, .flow-node.flow-hh {{ border-left: 3px solid #8B5CF6; }}
.flow-node.flow-lh {{ border-left: 3px solid #4ADE80; }}
.flow-node.flow-hl {{ border-left: 3px solid #FBBF24; }}
.flow-tip {{
  position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 6px; padding: 5px 10px; font-size: var(--text-xs); color: var(--lab-text-secondary);
  white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 140ms ease; z-index: 5;
}}
.flow-node:hover .flow-tip {{ opacity: 1; }}
.flow-band {{
  display: inline-block; padding: 6px 16px; border-radius: 6px; font-weight: 700;
  font-size: var(--text-sm); color: #0B1120;
}}
.flow-band.flow-ll {{ background: {ACCENT_CYAN}; }}
.flow-band.flow-lh {{ background: #4ADE80; }}
.flow-band.flow-hl {{ background: #FBBF24; }}
.flow-band.flow-hh {{ background: #8B5CF6; }}

/* ---------------- Aktif görüntü çubuğu (mega-spec "FINAL STATE/CALLBACK
   FIX" Part 21 — TEK global görüntü seçici, tüm sekmelerin ÜSTÜNDE, SINGLE
   SOURCE OF TRUTH) ---------------- */
.active-image-bar {{
  align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 10px 16px; margin: 4px 0 18px 0;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 10px;
}}
.active-image-bar-content {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.active-image-tag {{
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--lab-text-muted);
}}
.active-image-bar-content .chip-name {{ font-size: var(--text-lg); font-weight: 700; color: var(--lab-text-primary); }}
.active-image-dims {{ font-family: var(--lab-mono); font-size: var(--text-sm); color: var(--lab-text-secondary); }}
.active-image-resized {{
  font-family: var(--lab-mono); font-size: var(--text-xs); color: var(--lab-text-muted);
  border-left: 1px solid var(--lab-border-light); padding-left: 10px; cursor: help;
}}
.active-image-filesize {{
  font-family: var(--lab-mono); font-size: var(--text-xs); font-weight: 600;
  color: var(--lab-text-secondary);
  border-left: 1px solid var(--lab-border-light); padding-left: 10px; cursor: help;
}}

/* ---------------- DWT Ayrıştırma Gezgini — sabit bant açıklama şeridi
   (hover'a bağlı değil, her zaman görünür — Part 13) ---------------- */
.dwt-legend {{
  display: flex; flex-wrap: wrap; gap: 6px 18px;
  padding: 8px 14px; margin: 6px 0 12px 0;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border);
  border-radius: 8px; font-size: var(--text-xs); color: var(--lab-text-secondary);
}}
.dwt-legend-chip b {{ color: var(--lab-text-primary); }}
.dwt-tree-image img {{ cursor: pointer; }}

/* ---------------- Otomatik tam analiz durumu (mega-spec "AUTOMATIC FULL
   ANALYSIS PIPELINE" Part 3) — kompakt, geçici; bitince küçük bir rozete
   çöker (.analysis-status-ready), kalıcı büyük bir panel DEĞİLDİR. -------- */
.analysis-status {{
  display: flex; flex-wrap: wrap; gap: 4px 18px;
  padding: 8px 16px; margin: -4px 0 14px 0;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 8px; font-size: var(--text-sm);
}}
.analysis-status-row {{ display: flex; align-items: center; gap: 6px; color: var(--lab-text-muted); }}
.analysis-status-row.done {{ color: var(--lab-text-secondary); }}
.analysis-status-mark {{
  font-family: var(--lab-mono); width: 14px; text-align: center; color: var(--lab-text-muted);
}}
.analysis-status-row.done .analysis-status-mark {{ color: var(--lab-positive); font-weight: 700; }}
.analysis-status-ready {{
  display: inline-block; padding: 6px 14px; margin: -4px 0 14px 0;
  background: var(--lab-bg-elevated); border: 1px solid {ACCENT_CYAN};
  border-radius: 999px; font-size: var(--text-xs); font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase; color: {ACCENT_CYAN};
}}

/* Section başlığından sonra görüntü satırına kadar boşluk (editorial spacing) */
.section-gap-lg {{ margin-top: 40px; }}
.section-gap-sm {{ margin-top: 10px; }}

/* ---------------- SIKIŞTIRMA ÖZETİ — kompakt KPI satırı (FINAL FEATURE
   PASS Part 2-9). Mevcut .lab-card/.metric-card-head/.size-badge/.metric-
   title dilini yeniden kullanır; sadece kutu ızgarası + boyut oku yeni. */
.kpi-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 2px; }}
.kpi-box {{
  flex: 1 1 92px; min-width: 84px;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border);
  border-radius: 8px; padding: 8px 10px;
}}
.kpi-label {{
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--lab-text-muted);
}}
.kpi-value {{
  font-family: var(--lab-mono); font-size: var(--text-lg); font-weight: 700;
  color: var(--lab-text-primary); margin-top: 3px;
}}
.size-arrow-row {{
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--lab-border);
}}
.size-arrow-value {{ font-family: var(--lab-mono); font-size: var(--text-lg); font-weight: 700; color: var(--lab-text-primary); }}
.size-arrow-sep {{ color: var(--lab-text-muted); font-size: var(--text-lg); }}
.size-reduction-pct {{ font-family: var(--lab-mono); font-size: var(--text-sm); color: {POSITIVE}; font-weight: 700; margin-left: auto; }}

/* ---------------- Önce / Sonra kaydırıcısı (FINAL FEATURE PASS Part
   10-16) — saf CSS + tek native <input type=range>; JS yalnız inline
   oninput ile iki zaten-hesaplanmış görüntü arasında clip-path taşır,
   backend çağrısı YOK. object-fit gerekmez: konteyner aspect-ratio'su
   gerçek (w/h) oranından hesaplanır, iki görüntü de birebir aynı (H,W)
   şeklindedir (bkz. Part 11 pixel-alignment). ---- */
.ba-wrap {{ display: flex; flex-direction: column; gap: 8px; }}
.ba-frame {{
  position: relative; width: 100%; aspect-ratio: var(--ba-ar, 4/3);
  border-radius: 10px; overflow: hidden; border: 1px solid var(--lab-border);
  background: var(--lab-bg-elevated);
}}
.ba-frame img {{
  position: absolute; inset: 0; width: 100%; height: 100%;
  display: block; user-select: none; pointer-events: none;
}}
.ba-frame .ba-clip {{
  position: absolute; inset: 0; overflow: hidden;
  clip-path: inset(0 calc(100% - var(--ba-pos, 50%)) 0 0);
}}
.ba-frame .ba-divider {{
  position: absolute; top: 0; bottom: 0; width: 2px; left: var(--ba-pos, 50%);
  background: var(--lab-text-primary); transform: translateX(-1px);
  box-shadow: 0 0 0 1px rgba(0,0,0,0.35); pointer-events: none;
}}
.ba-frame .ba-tag {{
  position: absolute; top: 10px; font-family: var(--lab-mono);
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; padding: 3px 9px; border-radius: 5px;
  background: rgba(11,15,20,0.72); color: var(--lab-text-primary);
  border: 1px solid var(--lab-border-light); pointer-events: none;
}}
.ba-tag-left {{ left: 10px; }}
.ba-tag-right {{ right: 10px; }}
.ba-range {{
  -webkit-appearance: none; appearance: none; width: 100%; height: 24px;
  background: transparent; margin: 0; cursor: ew-resize;
}}
.ba-range::-webkit-slider-runnable-track {{ height: 4px; background: var(--lab-border); border-radius: 2px; }}
.ba-range::-webkit-slider-thumb {{
  -webkit-appearance: none; appearance: none; width: 18px; height: 18px;
  border-radius: 50%; margin-top: -7px; cursor: ew-resize;
  background: var(--lab-text-primary); border: 3px solid var(--ba-accent, {ACCENT_CYAN});
}}
.ba-range::-moz-range-track {{ height: 4px; background: var(--lab-border); border-radius: 2px; }}
.ba-range::-moz-range-thumb {{
  width: 18px; height: 18px; border-radius: 50%; border: 3px solid var(--ba-accent, {ACCENT_CYAN});
  background: var(--lab-text-primary); cursor: ew-resize;
}}
.ba-range:focus-visible {{ outline: 2px solid var(--ba-accent, {ACCENT_CYAN}); outline-offset: 3px; border-radius: 4px; }}

/* Yan Yana / Önce-Sonra görünüm seçici (mevcut Radio bileşenine binmez,
   sadece konteyner aralığı — Radio'nun kendi stili değişmez, Part 23:
   mevcut tasarım dilini koru). */
.view-toggle-row {{ margin: 4px 0 18px; }}

/* ---------------- DWT subband tıklanan-katsayı kutusu (DWT Lab —
   LH/HL/HH görüntülerine tıklayınca çıkan küçük, okunabilir kutu). ---- */
.dwt-pixel-box {{
  margin-top: 10px; padding: 8px 12px;
  background: var(--lab-bg-elevated); border: 1px solid {ACCENT_CYAN};
  border-radius: 8px;
}}
.dwt-pixel-title {{
  font-family: var(--lab-mono); font-size: var(--text-2xs); font-weight: 700;
  letter-spacing: 0.05em; text-transform: uppercase; color: {ACCENT_CYAN};
  margin-bottom: 4px;
}}
.dwt-pixel-row {{ display: flex; justify-content: space-between; padding: 1px 0; }}
.dwt-pixel-key {{ font-size: var(--text-xs); color: var(--lab-text-secondary); }}
.dwt-pixel-val {{ font-family: var(--lab-mono); font-size: var(--text-sm); font-weight: 600; color: var(--lab-text-primary); }}

/* ---------------- Genel gövde/açıklama metni (Part 13) ---------------- */
.gradio-container p, .gradio-container li {{ font-size: var(--text-md); line-height: 1.55; }}

/* ---------------- Responsive (Part 20) — masaüstü birincil hedef,
   dizüstü genişliklerinde ölçek hafifçe küçülür, mobil'de yığılır. ---- */
@media (max-width: 1439px) and (min-width: 900px) {{
  .lab-header .lab-title {{ font-size: 25px; }}
  .tab-nav button {{ font-size: 16px !important; }}
  .page-section-label {{ font-size: 20px; }}
  .section-label {{ font-size: 17px; }}
}}
"""
