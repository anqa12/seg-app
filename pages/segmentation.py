import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODELS, RESOLUTIONS, CLASS_HEX, CLASSES, LANDCOVER_MODELS, BIGEARTHNET_MODELS
from utils.models import load_model
from utils.inference import rescale, predict
from utils.display import (render_result_block, render_resolution_comparison,
                           render_model_comparison, render_gt_comparison)
from components.sample_tab import render_sample_tab
from components.upload_tab import render_upload_tab
from components.map_tab import render_map_tab
from utils.session import set_native_resolution

st.set_page_config(page_title="Segmentacja", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="stExpander"] { border: 1px solid rgba(128,128,128,0.2); border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.title("Segmentacja semantyczna")
st.markdown("---")

# Domyślna rozdzielczość i model dopasowane automatycznie do datasetu
# zdjęcia testowego (LandCover/BigEarthNet mają zupełnie inną skalę GSD,
# więc "pierwszy z listy" jako domyślny nie ma sensu dla BigEarthNet).
_DATASET_DEFAULT_RES = {
    "LandCover":   "25 cm/px",
    "BigEarthNet": "10 m/px",
}


def _default_models_for(dataset_type: str | None) -> list[str]:
    """Domyślny model UNet dla danego datasetu (a dla mapy/uploadu — LandCover,
    bo to generyczne ortofoto lotnicze, bliższe domenowo LandCover niż BigEarthNet)."""
    if dataset_type == "BigEarthNet" and BIGEARTHNET_MODELS:
        return [BIGEARTHNET_MODELS[0]]
    if LANDCOVER_MODELS:
        return [LANDCOVER_MODELS[0]]
    return [list(MODELS.keys())[0]]

# session state
if "source_res"  not in st.session_state: st.session_state["source_res"]  = 0.25
if "img_np" not in st.session_state: st.session_state["img_np"] = None
if "mask_gt" not in st.session_state: st.session_state["mask_gt"] = None
if "resolutions" not in st.session_state: st.session_state["resolutions"] = dict(RESOLUTIONS)

# choose image
st.subheader("Wybierz zdjęcie")

tab_sample, tab_map, tab_upload = st.tabs([
    "📂 Zdjęcia testowe", "🗺️ Pobierz z mapy", "⬆️ Wgraj własne"
])

with tab_sample:
    render_sample_tab()

with tab_upload:
    render_upload_tab()

with tab_map:
    render_map_tab()

img_np = st.session_state.get("img_np")
mask_gt = st.session_state.get("mask_gt")
source_res = st.session_state.get("source_res", 0.25)
img_source = st.session_state.get("img_source")

# Ustawienia i wyniki mają się pojawiać TYLKO w tej zakładce, z której
# faktycznie pochodzi aktualnie wybrane zdjęcie — nie na całej stronie.
# Obiekty zakładek zwrócone przez st.tabs() można ponownie użyć jako
# kontener (`with tab_x:`) w dalszej części skryptu, więc renderujemy
# resztę strony wewnątrz WŁAŚCIWEJ zakładki zamiast poniżej wszystkich.
_TAB_CONTAINERS = {"sample": tab_sample, "map": tab_map, "upload": tab_upload}
active_container = _TAB_CONTAINERS.get(img_source)

if img_np is None or active_container is None:
    st.info("Wybierz zdjęcie testowe lub wgraj własne aby przejść dalej.")
    st.stop()

with active_container:
    st.markdown("---")

    st.subheader("Wybierz ustawienia")

    # Rozdzielczość źródłowa NIEZNANA (np. wgrany plik bez georeferencji —
    # patrz utils/inference.py) — user musi ją podać TUTAJ, w panelu ustawień,
    # zamiast dodatkowego osobnego przycisku/pola w zakładce wyboru zdjęcia.
    # Zwykły selectbox od razu ma jakąś wartość (domyślną), więc nie trzeba
    # żadnego dodatkowego potwierdzenia — wystarczy, że się pojawi.
    manual_label = None
    if source_res is None:
        st.warning(
            "⚠️ Nie wykryto rozdzielczości źródłowej tego zdjęcia — "
            "wybierz ją poniżej, żeby móc uruchomić segmentację."
        )
        manual_label = st.selectbox(
            "Rozdzielczość źródłowa tego zdjęcia:",
            list(st.session_state["resolutions"].keys()),
            key=f"manual_source_res_{st.session_state.get('selection_gen', 0)}",
        )
        source_res = st.session_state["resolutions"][manual_label]
        st.session_state["source_res"] = source_res

    available_resolutions = list(st.session_state["resolutions"].keys())
    default_res = [available_resolutions[0]]

    chosen_dataset_type = st.session_state.get("chosen_dataset_type")

    if img_source == "sample" and chosen_dataset_type in _DATASET_DEFAULT_RES:
        wanted = _DATASET_DEFAULT_RES[chosen_dataset_type]
        if wanted in available_resolutions:
            default_res = [wanted]

    if img_source in ("upload", "map"):
        # utils/session.py gwarantuje co najwyżej jeden wpis "Natywna (...)"
        # w resolutions — ustawiony przez upload_tab.py (metadane) albo
        # map_tab.py (rozdzielczość pobranego fragmentu ortofotomapy).
        native_label = next(
            (l for l in st.session_state["resolutions"] if l.startswith("Natywna (")),
            None,
        )
        if native_label:
            available_resolutions = list(st.session_state["resolutions"].keys())
            if native_label not in available_resolutions:
                available_resolutions.insert(0, native_label)
            default_res = [native_label]

    # Jeśli user właśnie ręcznie podał rozdzielczość źródłową powyżej, niech
    # ta sama wartość będzie od razu domyślnie zaznaczona jako rozdzielczość
    # docelowa — logicznie to ta sama liczba, więc nie ma sensu wymuszać
    # dodatkowego, osobnego wyboru.
    if manual_label:
        default_res = [manual_label]

    col_m, col_r, col_v = st.columns([2, 2, 1])

    with col_m:
        st.markdown("**Model / Modele**")
        st.caption("Wybierz jeden (porównanie rozdzielczości) lub wiele (porównanie modeli)")
        selected_models = st.multiselect(
            "Modele", list(MODELS.keys()),
            default=_default_models_for(chosen_dataset_type),
            label_visibility="collapsed",
            key=f"selected_models_{st.session_state.get('selection_gen', 0)}"
        )

        # warning if model does not support current dataset
        if chosen_dataset_type:
            mismatched = [
                m for m in selected_models
                if chosen_dataset_type.lower() not in m.lower()
            ]
            if mismatched:
                st.caption(
                    f"{', '.join(mismatched)} {'nie był' if len(mismatched)==1 else 'nie były'} "
                    f"trenowane na zbiorze **{chosen_dataset_type}** — wyniki mogą być słabe."
                )

    with col_r:
        st.markdown("**Rozdzielczość / Rozdzielczości**")
        st.caption("Wybierz jedną lub więcej do porównania")
        selected_res = st.multiselect(
            "Rozdzielczości", available_resolutions,
            default=default_res,
            label_visibility="collapsed",
            key=f"selected_res_{st.session_state.get('selection_gen', 0)}"
        )

    with col_v:
        st.markdown("**Nakładka**")
        alpha = st.slider("Przezroczystość:", 0.0, 1.0, 0.45, 0.05,
                          label_visibility="collapsed")
        global_overlap = st.slider("Overlap kafelków:", 0.0, 0.5, 0.0, 0.1,
                                   help="Używane jeśli model nie narzuca własnego overlapu w config.py")
        # use overlap checkbox
        use_voting = st.checkbox(
            "Gaussian voting (gładsze granice)", value=(global_overlap > 0),
            help="Włącz razem z overlap > 0, żeby faktycznie uśredniać nakładające się kafelki."
        )

    n_m, n_r = len(selected_models), len(selected_res)
    if n_m == 0 or n_r == 0:
        st.warning("Wybierz co najmniej jeden model i jedną rozdzielczość.")
        st.stop()
    elif n_m > 1 and n_r > 1:
        st.warning("Wybrano wiele modeli **i** wiele rozdzielczości. "
                   "Wybierz **jeden model** (porównanie rozdzielczości) "
                   "lub **jedną rozdzielczość** (porównanie modeli).")
        st.stop()
    elif n_m == 1 and n_r > 1:
        st.info(f"Tryb: **porównanie rozdzielczości** — model *{selected_models[0]}*, "
                f"{n_r} rozdzielczości")
    elif n_m > 1 and n_r == 1:
        st.info(f"Tryb: **porównanie modeli** — {n_m} modeli, "
                f"rozdzielczość *{selected_res[0]}*")
    else:
        st.info(f"Tryb: **pojedynczy wynik** — *{selected_models[0]}*, *{selected_res[0]}*")

    st.markdown("---")

    st.subheader("Wyniki segmentacji")

    # "Zatrzymaj" musi być wyrenderowany i klikalny ZANIM zacznie się ciężka
    # pętla predykcji — Streamlit z automatu przerywa trwający przebieg
    # skryptu, gdy user kliknie cokolwiek nowego, więc wystarczy, że przycisk
    # jest już w DOM-ie (dlatego ustawiamy seg_running i sprawdzamy je od razu,
    # a nie dopiero po zakończeniu obliczeń).
    if st.button("Uruchom segmentację", type="primary", width="stretch"):
        st.session_state["seg_running"] = True

    running = st.session_state.get("seg_running", False)

    if running:
        if st.button("🛑 Zatrzymaj", width="stretch"):
            st.session_state["seg_running"] = False
            st.warning("Segmentacja została zatrzymana przez użytkownika.")
            st.stop()

    if not running:
        st.stop()


    def calculate_actual_scale(res_label: str, source_resolution_m: float) -> float:
        target_res_m = st.session_state["resolutions"][res_label]
        return source_resolution_m / target_res_m


    # one model, one or more resolutions
    if n_m == 1 and n_r >= 1:
        model_name = selected_models[0]
        with st.spinner(f"Ładowanie modelu {model_name}…"):
            model = load_model(model_name)
            tile_size = MODELS[model_name]["tile"]
            overlap = MODELS[model_name].get("overlap", global_overlap)
            voting = MODELS[model_name].get("use_voting", use_voting)

        if model is None:
            st.stop()

        results = []
        for res_label in selected_res:
            actual_scale = calculate_actual_scale(res_label, source_res)

            # warning when resizing an image
            if actual_scale > 1.5:
                st.caption(
                    f"{res_label}: powiększenie {actual_scale:.1f}× "
                    f"(zdjęcie źródłowe ma niższą rozdzielczość) — jakość może być słaba."
                )

            img_s = rescale(img_np, actual_scale, tile_size)
            mask = predict(model, img_s, tile_size=tile_size, overlap=overlap,
                           use_voting=voting, progress_label=f"[{res_label}]")
            if mask is None:
                continue

            results.append({"label": res_label, "img": img_s, "mask": mask, "metrics": None})

        if not results:
            st.stop()

        if n_r == 1:
            r = results[0]
            if mask_gt is not None:
                render_gt_comparison(r["img"], r["mask"], mask_gt, alpha,
                                     label=f"{model_name} · {r['label']} · "
                                           f"{r['img'].shape[1]}×{r['img'].shape[0]} px")
            else:
                render_result_block(r["img"], r["mask"], alpha,
                                    label=f"{model_name} · {r['label']} · "
                                          f"{r['img'].shape[1]}×{r['img'].shape[0]} px")
        else:
            render_resolution_comparison(results, alpha, mask_gt=mask_gt)

    # more models, one resolution
    elif n_r == 1 and n_m > 1:
        res_label = selected_res[0]
        actual_scale = calculate_actual_scale(res_label, source_res)

        if actual_scale > 1.5:
            st.caption(f"{res_label}: powiększenie {actual_scale:.1f}× — jakość może być słaba.")

        model_results = {}
        for model_name in selected_models:
            ts = MODELS[model_name]["tile"]
            overlap = MODELS[model_name].get("overlap", global_overlap)
            voting = MODELS[model_name].get("use_voting", use_voting)

            with st.spinner(f"Ładowanie {model_name}…"):
                model = load_model(model_name)
            if model is None:
                continue

            img_s = rescale(img_np, actual_scale, ts)
            mask = predict(model, img_s, tile_size=ts, overlap=overlap,
                           use_voting=voting, progress_label=f"[{model_name}]")
            if mask is not None:
                model_results[model_name] = mask

        if model_results:
            first_ts = MODELS[selected_models[0]]["tile"]
            img_input = rescale(img_np, actual_scale, first_ts)
            render_model_comparison(img_input, model_results, alpha, mask_gt=mask_gt)

    # obliczenia zakończone (nie przerwane) — wracamy do stanu początkowego
    st.session_state["seg_running"] = False