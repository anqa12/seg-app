import streamlit as st

st.set_page_config(
    page_title="Seg-app",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 3rem; padding-left: 4rem; padding-right: 4rem; }
.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    font-weight: 500;
}
.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# header (left col with available pages, right col with content)
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown("""
    <div style="display:inline-flex;align-items:center;gap:6px;
         background:rgba(128,128,128,0.08);border:0.5px solid rgba(128,128,128,0.2);
         border-radius:20px;padding:4px 12px;font-size:12px;color:gray;
         margin-bottom:1.5rem">
       Projekt inżynierski
    </div>
    """, unsafe_allow_html=True)

    st.markdown("# Segmentacja semantyczna obrazów teledetekcyjnych")
    st.markdown("""
    <p style="font-size:16px;color:gray;line-height:1.8;margin-bottom:2rem">
    Aplikacja służąca do demonstracji modeli CNN i Transformer wytrenowanych
    na wysokorozdzielczych zdjęciach lotniczych i satelitarnych.
    Możliwość wyboru zdjęcia z bazy, wgrania własnego, wybrania modeli
    oraz rozdzielczości do testów.
    </p>
    """, unsafe_allow_html=True)

    if st.button("Przejdź do segmentacji →", type="primary"):
        st.switch_page("pages/segmentation.py")

st.markdown("---")

# three cards describing the functionalities of the app
st.markdown("**Co możesz zrobić**")
c1, c2, c3 = st.columns(3, gap="large")

_card_style = (
    "border:1px solid rgba(100,100,100,0.25);"
    "border-radius:12px;"
    "padding:1.2rem 1rem;"
    "height:100%;"
)

with c1:
    st.markdown(f"""
    <div style="{_card_style}">
      <div style="font-size:22px;margin-bottom:10px">🖼️</div>
      <div style="font-size:14px;font-weight:500;margin-bottom:6px">Wybierz zdjęcie</div>
      <div style="font-size:13px;color:gray;line-height:1.6">
        Przykładowe zdjęcia ze zbioru testowego LandCover.ai lub BigEarthNet,
        własny plik JPG/PNG/TIF, lub pobierz obszar z mapy Polski.
      </div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div style="{_card_style}">
      <div style="font-size:22px;margin-bottom:10px">🤖</div>
      <div style="font-size:14px;font-weight:500;margin-bottom:6px">Wybierz model</div>
      <div style="font-size:13px;color:gray;line-height:1.6">
        UNet, DeepLabV3+ lub SegFormer — jeden lub wiele architektur
        do porównania wyników segmentacji.
      </div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="{_card_style}">
      <div style="font-size:22px;margin-bottom:10px">🔍</div>
      <div style="font-size:14px;font-weight:500;margin-bottom:6px">Wybierz rozdzielczość</div>
      <div style="font-size:13px;color:gray;line-height:1.6">
        Od 25 cm/px do 10 m/px — porównaj jak degradacja
        rozdzielczości wpływa na jakość segmentacji.
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("---")

# segmentation classes
st.markdown("**Klasy segmentacji**")
CLASS_HEX = ["#000000", "#8B5A2B", "#228B22", "#4169E1", "#C0C0C0"]
CLASSES = ["Tło", "Zabudowa", "Lasy", "Woda", "Drogi"]

dots = "".join(
    f'<span style="display:inline-flex;align-items:center;gap:5px;'
    f'background:rgba(128,128,128,0.10);border:1px solid rgba(100,100,100,0.22);'
    f'border-radius:20px;padding:5px 12px;font-size:13px;color:gray;margin-right:8px">'
    f'<span style="width:9px;height:9px;border-radius:50%;background:{CLASS_HEX[i]};'
    f'{"border:1px solid rgba(200,200,200,.4)" if i in [0, 4] else ""}"></span>'
    f'{c}</span>'
    for i, c in enumerate(CLASSES)
)
st.markdown(f'<div style="margin:8px 0 1.5rem">{dots}</div>', unsafe_allow_html=True)

st.markdown("---")

# footer - info about the Data and Models
st.markdown("""
<div style="font-size:12px;color:gray;line-height:2.2">
<b>Zbiory danych:</b>
LandCover.ai (25 cm/px, Polska) &nbsp;·&nbsp;
BigEarthNet reBEN v2.0 (10 m/px, Europa)<br>
<b>Modele:</b>
UNet + ResNet34 &nbsp;·&nbsp;
DeepLabV3+ + ResNet34 &nbsp;·&nbsp;
SegFormer-B2 &nbsp;·&nbsp;
UNet BigEarthNet &nbsp;·&nbsp;
DeepLabV3+ BigEarthNet &nbsp;·&nbsp;
SegFormer BigEarthNet
</div>
""", unsafe_allow_html=True)