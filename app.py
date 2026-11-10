import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from utils.config import CLASSES, CLASS_COLORS, CLASS_HEX, MODELS, RESOLUTIONS, SAMPLE_IMAGES

st.set_page_config(
    page_title="Seg-app",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# sample image and corresponding mask
PREVIEW_MASK_PATH: Path = Path("assets/preview_mask.png")
PREVIEW_IMAGE_PATH: Path = Path("assets/preview_img.jpg")


# gray placeholder for image
def _placeholder(text: str) -> str:
    return (
        '<div style="aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;'
        'border-radius:8px;background:rgba(128,128,128,0.12);font-size:13px;opacity:0.6">'
        f"{text}</div>"
    )


@st.cache_data
def _to_data_uri(path: str, mtime: float, colorize: bool) -> str:
    """Read an image file and return it as a base64 data URI for an <img> tag.

    - (colorize=True) it turns class indices (0..4) into the RGB colors
    from CLASS_COLORS; a normal photo (colorize=False) is only converted to RGB.
    - mtime: it is not used inside, it is only part of the cache key, so replacing
    the file on disk refreshes the cached result instead of showing the old image.
    """
    img = Image.open(path)

    if colorize:
        arr = np.array(img)
        if arr.ndim == 3:
            # RGB/RGBA masks: all channels hold the same index, one channel is enough
            arr = arr[..., 0]
        if arr.max() < len(CLASS_COLORS):
            # map each index to its color
            rgb = CLASS_COLORS[arr]
        else:
            # values too large: already colored
            rgb = np.array(img.convert("RGB"))
        out = Image.fromarray(rgb)
    else:
        out = img.convert("RGB")

    max_width = 800
    if out.width > max_width:
        new_height = round(out.height * max_width / out.width)
        # nearest - keeps class borders sharp, LANCZOS gives the best quality for a photo
        out = out.resize((max_width, new_height), Image.NEAREST if colorize else Image.LANCZOS)

    buf = io.BytesIO()
    if colorize:
        out.save(buf, format="PNG")
        mime = "image/png"
    else:
        out.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _preview_item(path: Path | None, caption: str, colorize: bool, missing: str) -> str:
    """One image with its caption."""
    if path is None or not path.exists():
        body = _placeholder(missing)
    else:
        uri = _to_data_uri(str(path), path.stat().st_mtime, colorize)
        body = f'<img class="pair-img" src="{uri}" alt="{caption}">'
    return f'<div class="pair-item">{body}<div class="pair-cap">{caption}</div></div>'


def load_preview_html() -> str:
    """Return the HTML of the preview: input photo -> segmentation mask."""
    photo = _preview_item(PREVIEW_IMAGE_PATH, "Zdjęcie wejściowe", colorize=False, missing="Brak zdjęcia")
    mask = _preview_item(PREVIEW_MASK_PATH, "Wynik segmentacji", colorize=True, missing="Brak maski")
    return f'<div class="preview"><div class="pair">{photo}<div class="arrow">→</div>{mask}</div></div>'


# global styles
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
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #2563EB !important;
    color: #ffffff !important;
    border: none !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
}

/* Hero preview (right side): input photo -> segmentation mask */
.preview {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 14px;
    padding: 12px;
}
.pair { display: flex; align-items: center; gap: 10px; }
.pair-item { flex: 1; min-width: 0; }
/* Square + cover => both images always have exactly the same size */
.pair-img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    display: block;
    border-radius: 8px;
}
.pair-cap { font-size: 12px; opacity: 0.55; margin-top: 8px; text-align: center; }
/* bottom padding ~ caption height, so the arrow sits at the middle of the images */
.arrow { font-size: 20px; opacity: 0.45; padding-bottom: 24px; }

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

# text + button on the left, mask preview on the right
left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")

with left:
    st.markdown("""
<div class="eyebrow">Projekt inżynierski</div>
<div class="hero-title">Segmentacja semantyczna obrazów teledetekcyjnych</div>
<div class="hero-text">
Aplikacja do demonstracji modeli CNN i Transformer wytrenowanych na
zdjęciach lotniczych oraz satelitarnych. Wybierz zdjęcie,
model i rozdzielczość, a następnie porównaj wyniki.
</div>
""", unsafe_allow_html=True)
    if st.button("Przejdź do segmentacji →", type="primary"):
        st.switch_page("pages/segmentation.py")

with right:
    st.markdown(load_preview_html(), unsafe_allow_html=True)

# key numbers (model, dataset, class and resolution counts come from config)
st.markdown(f"""
<div class="stats">
<div class="stat"><div class="num">{len(MODELS)}</div><div class="label">modeli do porównania</div></div>
<div class="stat"><div class="num">{len(SAMPLE_IMAGES)}</div><div class="label">zbiory danych</div></div>
<div class="stat"><div class="num">{len(CLASSES)}</div><div class="label">klas segmentacji</div></div>
<div class="stat"><div class="num">{len(RESOLUTIONS)}</div><div class="label">poziomów rozdzielczości do testów</div></div>
</div>
""", unsafe_allow_html=True)

# feature cards
st.markdown("""
<div class="section-title"></div>
<div class="cards">
<div class="card">
<div class="title">Wybierz zdjęcie</div>
<div class="text">Przykłady z LandCover.ai lub BigEarthNet, własny plik JPG/PNG/TIF albo obszar pobrany z mapy Polski.</div>
</div>
<div class="card">
<div class="title">Wybierz model</div>
<div class="text">UNet, DeepLabV3+ lub SegFormer. Porównaj różne architektury.</div>
</div>
<div class="card">
<div class="title">Wybierz rozdzielczość</div>
<div class="text">Od 25 cm/px do 10 m/px. Zobacz, jak pogorszenie rozdzielczości wpływa na jakość segmentacji.</div>
</div>
</div>
""", unsafe_allow_html=True)

# segmentation classes
chips = "".join(
    f'<span class="chip"><span class="dot" style="background:{CLASS_HEX[i]}"></span>{name}</span>'
    for i, name in enumerate(CLASSES)
)
st.markdown(
    f'<div class="section-title">Klasy segmentacji</div><div class="chips">{chips}</div>',
    unsafe_allow_html=True,
)

# footer
st.markdown("""
<div class="footer">
<b>Zbiory danych:</b> LandCover.ai (25 cm/px, Polska) · BigEarthNet reBEN v2.0 (10 m/px, Europa)<br>
<b>Modele:</b> UNet + ResNet34 · DeepLabV3+ + ResNet34 · SegFormer-B2
</div>
""", unsafe_allow_html=True)