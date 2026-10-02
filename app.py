import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from utils.config import CLASSES, CLASS_COLORS, CLASS_HEX

st.set_page_config(
    page_title="Seg-app",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Path to the hero image: a segmentation mask whose pixel values are class
# indices (0-4). Relative to the folder you run `streamlit run` from.
PREVIEW_IMAGE_PATH = Path("assets/preview.png")


@st.cache_data
def load_preview_html() -> str:
    """Return the HTML for the hero image, with the mask colored by class.

    What: reads the mask, turns class indices (0..4) into the RGB colors from
    CLASS_COLORS, and embeds the result as a base64 <img>.
    Why: a raw mask stores classes as tiny pixel values (0, 1, 2, ...), which look
    almost black on screen. Coloring it with the same palette as the rest of the
    app makes it readable. Cached, so it runs once instead of on every rerun.
    """
    if not PREVIEW_IMAGE_PATH.exists():
        # Simple placeholder so the page still loads if the file is missing
        return (
            '<div style="aspect-ratio:3/2;display:flex;align-items:center;'
            'justify-content:center;border-radius:8px;'
            'background:rgba(128,128,128,0.12);font-size:13px;opacity:0.6">'
            "Brak obrazka</div>"
        )

    img = Image.open(PREVIEW_IMAGE_PATH)
    arr = np.array(img)  # palette ("P") images give class indices directly

    # RGB/RGBA masks: all channels hold the same index, so one channel is enough
    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.max() < len(CLASS_COLORS):
        # Looks like a class-index mask -> map each index to its color
        rgb = CLASS_COLORS[arr]
    else:
        # Values too large to be class indices: the image is already colored
        rgb = np.array(img.convert("RGB"))

    out = Image.fromarray(rgb)

    # Downscale big masks to keep the page light. NEAREST keeps class borders
    # sharp (smoother filters would blend colors into classes that don't exist).
    max_width = 1200
    if out.width > max_width:
        new_height = round(out.height * max_width / out.width)
        out = out.resize((max_width, new_height), Image.NEAREST)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<img src="data:image/png;base64,{data}" alt="Wynik segmentacji" '
        'style="width:100%;display:block;border-radius:8px">'
    )


# ---------------------------------------------------------------------------
# Global styles
# Why: one stylesheet keeps the look consistent. Colors are semi-transparent so
# the page works in both light and dark Streamlit themes.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Hide Streamlit's default chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Wider content column so the page uses more of the screen */
.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1200px !important;
    padding-top: 4rem !important;
    padding-bottom: 4rem !important;
}

.eyebrow {
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.55;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.15;
    margin: 0 0 1.1rem 0;
}
.hero-text {
    font-size: 17px;
    line-height: 1.7;
    opacity: 0.7;
    margin-bottom: 1.8rem;
}

/* Primary button */
.stButton > button {
    background-color: #2563EB;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.8rem;
    font-weight: 500;
}
.stButton > button:hover,
.stButton > button:focus {
    background-color: #1D4ED8;
    color: #fff;
    border: none;
}

/* Preview of the fake segmentation mask (right side of the hero) */
.preview {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 14px;
    padding: 10px;
}
.preview svg { display: block; border-radius: 8px; }
.preview .caption {
    font-size: 12px;
    opacity: 0.55;
    margin: 8px 2px 0 2px;
}

/* Row of key numbers */
.stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 3.5rem;
    padding: 1.6rem 0;
    border-top: 1px solid rgba(128, 128, 128, 0.18);
    border-bottom: 1px solid rgba(128, 128, 128, 0.18);
}
.stat .num   { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.stat .label { font-size: 13px; opacity: 0.6; margin-top: 4px; }

.section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.55;
    margin: 3.5rem 0 1rem 0;
}

/* CSS grid => the three cards always have equal height */
.cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}
.card {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 14px;
    padding: 1.6rem 1.5rem 1.7rem 1.5rem;
    min-height: 190px;
}
.card .icon  { font-size: 26px; margin-bottom: 12px; }
.card .title { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.card .text  { font-size: 14px; line-height: 1.65; opacity: 0.7; }

/* Class chips */
.chips { display: flex; flex-wrap: wrap; gap: 10px; }
.chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 14px;
    opacity: 0.9;
}
.dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    border: 1px solid rgba(128, 128, 128, 0.45);  /* keeps dark/light colors visible */
}

.footer {
    margin-top: 3.5rem;
    padding-top: 1.4rem;
    border-top: 1px solid rgba(128, 128, 128, 0.18);
    font-size: 13px;
    line-height: 1.9;
    opacity: 0.6;
}

/* Narrow screens: stack things vertically */
@media (max-width: 800px) {
    .cards, .stats { grid-template-columns: 1fr 1fr; }
    .hero-title { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero: text + button on the left, mask preview on the right
# ---------------------------------------------------------------------------
left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")

with left:
    st.markdown("""
<div class="eyebrow">Projekt inżynierski</div>
<div class="hero-title">Segmentacja semantyczna obrazów teledetekcyjnych</div>
<div class="hero-text">
Aplikacja do demonstracji modeli CNN i Transformer wytrenowanych na
wysokorozdzielczych zdjęciach lotniczych i satelitarnych. Wybierz zdjęcie,
modele i rozdzielczość, a następnie porównaj wyniki.
</div>
""", unsafe_allow_html=True)
    if st.button("Przejdź do segmentacji →", type="primary"):
        st.switch_page("pages/segmentation.py")

with right:
    st.markdown(
        f'<div class="preview">{load_preview_html()}'
        f'<div class="caption">Przykładowy wynik segmentacji</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Key numbers (class count comes from config so it stays in sync)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="stats">
<div class="stat"><div class="num">6</div><div class="label">modeli do porównania</div></div>
<div class="stat"><div class="num">2</div><div class="label">zbiory danych</div></div>
<div class="stat"><div class="num">{len(CLASSES)}</div><div class="label">klas segmentacji</div></div>
<div class="stat"><div class="num">25 cm – 10 m</div><div class="label">zakres rozdzielczości</div></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Feature cards (single HTML block; no indentation, or Markdown treats it as code)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="section-title">Co możesz zrobić</div>
<div class="cards">
<div class="card">
<div class="icon">🖼️</div>
<div class="title">Wybierz zdjęcie</div>
<div class="text">Przykłady z LandCover.ai lub BigEarthNet, własny plik JPG/PNG/TIF albo obszar pobrany z mapy Polski.</div>
</div>
<div class="card">
<div class="icon">🤖</div>
<div class="title">Wybierz model</div>
<div class="text">UNet, DeepLabV3+ lub SegFormer. Jeden model albo kilka naraz, żeby porównać wyniki.</div>
</div>
<div class="card">
<div class="icon">🔍</div>
<div class="title">Wybierz rozdzielczość</div>
<div class="text">Od 25 cm/px do 10 m/px. Zobacz, jak pogorszenie rozdzielczości wpływa na jakość segmentacji.</div>
</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Segmentation classes
# ---------------------------------------------------------------------------
chips = "".join(
    f'<span class="chip"><span class="dot" style="background:{CLASS_HEX[i]}"></span>{name}</span>'
    for i, name in enumerate(CLASSES)
)
st.markdown(
    f'<div class="section-title">Klasy segmentacji</div><div class="chips">{chips}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
<b>Zbiory danych:</b> LandCover.ai (25 cm/px, Polska) · BigEarthNet reBEN v2.0 (10 m/px, Europa)<br>
<b>Modele:</b> UNet + ResNet34 · DeepLabV3+ + ResNet34 · SegFormer-B2 · UNet BigEarthNet · DeepLabV3+ BigEarthNet · SegFormer BigEarthNet
</div>
""", unsafe_allow_html=True)