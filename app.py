import streamlit as st

st.set_page_config(
    page_title="Seg-app",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container { padding-top: 3rem; max-width: 720px; }
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# badge
st.markdown("""
<div style="display:inline-flex;align-items:center;gap:6px;
     background:rgba(128,128,128,0.08);border:0.5px solid rgba(128,128,128,0.2);
     border-radius:20px;padding:4px 12px;font-size:12px;color:gray;margin-bottom:1.5rem">
  🛰️ &nbsp; Praca inżynierska 2026
</div>
""", unsafe_allow_html=True)

# title and description
st.markdown("## Segmentacja semantyczna obrazów teledetekcyjnych")
st.markdown("""
<p style="font-size:15px;color:gray;line-height:1.7;max-width:560px;margin-bottom:2rem">
Aplikacja służąca do demonstracji modeli CNN i Transformer wytrenowanych na
wysokorozdzielczych zdjęciach lotniczych i satelitarnych.
Możliwość wyboru zdjęcia z bazy/ wgrania własnego, wybrania modeli oraz rozdzielczości do testów.
</p>
""", unsafe_allow_html=True)

# button
if st.button("Przejdź do segmentacji", type="primary"):
    st.switch_page("pages/segmentation.py")

st.markdown("---")

# three cards
st.markdown("**Co możesz zrobić**")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div style="
    border:1px solid rgba(100,100,100,0.25);
    border-radius:12px;
    padding:1rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
    ">
      <div style="font-size:24px;margin-bottom:8px"/>
      <div style="font-size:13px;font-weight:500;margin-bottom:4px">Wybierz zdjęcie</div>
      <div style="font-size:12px;color:gray;line-height:1.5">
        Przykładowe zdjęcia ze zbioru testowego lub własny plik JPG/PNG/TIF
      </div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="border:0.5px solid rgba(128,128,128,0.2);border-radius:12px;padding:1rem">
      <div style="font-size:24px;margin-bottom:8px"/>
      <div style="font-size:13px;font-weight:500;margin-bottom:4px">Wybierz model</div>
      <div style="font-size:12px;color:gray;line-height:1.5">
        UNet, DeepLabV3+ lub SegFormer — jeden lub wiele do porównania
      </div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div style="border:0.5px solid rgba(128,128,128,0.2);border-radius:12px;padding:1rem">
      <div style="font-size:24px;margin-bottom:8px"/>
      <div style="font-size:13px;font-weight:500;margin-bottom:4px">Wybierz rozdzielczość</div>
      <div style="font-size:12px;color:gray;line-height:1.5">
        Od 25 cm/px do 10 m/px — porównaj jak skala wpływa na jakość
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("")
st.markdown("---")

# classes
st.markdown("**Klasy segmentacji**")
CLASS_HEX = ["#000000", "#8B5A2B", "#228B22", "#4169E1", "#C0C0C0"]
CLASSES   = ["Tło", "Zabudowa", "Lasy", "Woda", "Drogi"]

dots = "".join(
    f'<span style="display:inline-flex;align-items:center;gap:5px;'
    f'background:rgba(128,128,128,0.10);'f'border:1px solid rgba(100,100,100,0.22);'
    f'border-radius:20px;padding:4px 10px;font-size:12px;color:gray;margin-right:6px">'
    f'<span style="width:8px;height:8px;border-radius:50%;background:{CLASS_HEX[i]};'
    f'{"border:1px solid rgba(128,128,128,.3)" if c in ["Tło","Drogi"] else ""}"></span>'
    f'{c}</span>'
    for i, c in enumerate(CLASSES)
)
st.markdown(f'<div style="margin:6px 0 1.5rem">{dots}</div>', unsafe_allow_html=True)

st.markdown("---")

# footer
st.markdown("""
<div style="font-size:11px;color:gray;line-height:2">
<b>Zbiory danych:</b> LandCover.ai (25 cm/px, Polska) &nbsp;·&nbsp; BigEarthNet reBEN v2.0 (10 m/px, Europa)<br>
<b>Modele:</b> UNet + ResNet34 &nbsp;·&nbsp; DeepLabV3+ + ResNet50 &nbsp;·&nbsp; SegFormer-B2 &nbsp;·&nbsp; UNet BigEarthNet
</div>
""", unsafe_allow_html=True)