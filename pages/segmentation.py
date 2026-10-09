import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODELS, RESOLUTIONS, LANDCOVER_MODELS, BIGEARTHNET_MODELS
from utils.models import load_model
from utils.inference import rescale, predict, check_rescale, count_tiles
from utils.display import (render_result_block, render_resolution_comparison,
                           render_model_comparison, render_gt_comparison)
from components.sample_tab import render_sample_tab
from components.upload_tab import render_upload_tab
from components.map_tab import render_map_tab

st.set_page_config(page_title="Seg-app", page_icon="🛰", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="stExpander"] { border: 1px solid rgba(128,128,128,0.2); border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.title("Segmentacja semantyczna obrazów teledetekcyjnych")
st.markdown("---")

_DATASET_DEFAULT_RES = {
    "LandCover":   "0.25 m/px",
    "BigEarthNet": "10 m/px",
}


def _default_models_for(dataset_type: str | None) -> list[str]:
    """
    Choose the right base model for LC and BEN datasets
    :param dataset_type:
    :return: list of available model names
    """
    if dataset_type == "BigEarthNet" and BIGEARTHNET_MODELS:
        return [BIGEARTHNET_MODELS[0]]
    if LANDCOVER_MODELS:
        return [LANDCOVER_MODELS[0]]
    return [list(MODELS.keys())[0]]


def calculate_actual_scale(res_label: str, source_resolution_m: float) -> float:
    """
    how much the source image has to be scaled to reach the chosen resolution
    :param res_label:
    :param source_resolution_m:
    :return: float
    """
    target_res_m = st.session_state["resolutions"][res_label]
    return source_resolution_m / target_res_m


def _tile_counts(img_shape: tuple, models: list[str], resolutions: list[str],
                 source_resolution_m: float, overlap_value: float) -> list[int]:
    """
    how many tiles each selected model or resolution needs to be cut into
    :param img_shape:
    :param models:
    :param resolutions:
    :param source_resolution_m:
    :param overlap_value:
    :return: list[int] counts
    """
    counts = []
    for model_name in models:
        for res_label in resolutions:
            scale = calculate_actual_scale(res_label, source_resolution_m)
            if check_rescale(img_shape, scale)[0]:  # skip resolutions that will be rejected anyway
                counts.append(count_tiles(img_shape, scale, MODELS[model_name]["tile"], overlap_value))
    return counts


# Session state
if "source_res" not in st.session_state: st.session_state["source_res"] = 0.25
if "img_np" not in st.session_state: st.session_state["img_np"] = None
if "mask_gt" not in st.session_state: st.session_state["mask_gt"] = None
if "resolutions" not in st.session_state: st.session_state["resolutions"] = dict(RESOLUTIONS)

# Choose image
st.subheader("Wybierz zdjęcie")

tab_sample, tab_map, tab_upload = st.tabs([
    "📂 Zdjęcia testowe", "🗺️ Pobierz z mapy", "⬆️ Wgraj własne"
])

with tab_sample:
    render_sample_tab()

with tab_map:
    render_map_tab()

with tab_upload:
    render_upload_tab()

img_np = st.session_state.get("img_np")
mask_gt = st.session_state.get("mask_gt")
source_res = st.session_state.get("source_res", 0.25)
img_source = st.session_state.get("img_source")

_TAB_CONTAINERS = {"sample": tab_sample, "map": tab_map, "upload": tab_upload}
active_container = _TAB_CONTAINERS.get(img_source)

if img_np is None or active_container is None:
    st.info("Wybierz zdjęcie testowe lub wgraj własne, aby przejść dalej.")
    st.stop()

with active_container:
    st.markdown("---")
    st.subheader("Wybierz ustawienia")

    manual_label = None
    if source_res is None:
        st.warning(
            "Nie wykryto rozdzielczości źródłowej zdjęcia — "
            "wybierz ją poniżej, żeby móc uruchomić segmentację."
        )
        manual_label = st.selectbox(
            "Rozdzielczość źródłowa zdjęcia:",
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

    if img_source == "map":
        native_label = st.session_state.get("_map_native_label")
    elif img_source == "upload":
        native_label = st.session_state.get("_upload_native_label")
    else:
        native_label = None

    if native_label and native_label in st.session_state["resolutions"]:
        default_res = [native_label]

    if manual_label:
        default_res = [manual_label]

    col_m, col_r, col_v = st.columns([2, 2, 1])

    # Choose model(s)
    with col_m:
        st.markdown("**Model / Modele**")
        st.caption("Wybierz jeden lub wiele do porównania")
        selected_models = st.multiselect(
            "Modele", list(MODELS.keys()),
            default=_default_models_for(chosen_dataset_type),
            label_visibility="collapsed",
            key=f"selected_models_{st.session_state.get('selection_gen', 0)}"
        )

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

    # Choose resolution(s)
    with col_r:
        st.markdown("**Rozdzielczość / Rozdzielczości**")
        st.caption("Wybierz jedną lub więcej do porównania")
        selected_res = st.multiselect(
            "Rozdzielczości", available_resolutions,
            default=default_res,
            label_visibility="collapsed",
            key=f"selected_res_{st.session_state.get('selection_gen', 0)}"
        )

    # advanced settings - gaussian blending
    # The panel is shown only when the image really gets cut into tiles (large photos from
    # the map or uploads). Ready-made test images fit in one tile, so there is nothing to set.
    overlap, use_voting = 0.25, True  # used when the panel is hidden
    with col_v:
        st.markdown("**Zaawansowane**")
        if max(_tile_counts(img_np.shape, selected_models, selected_res, source_res, 0.0), default=1) > 1:
            with st.expander("⚙️ Podział na mniejsze fragmenty", expanded=False):
                overlap = st.slider(
                    "Nakładanie się wycinków:", 0.0, 0.5, 0.25, 0.05,
                    help="Większe = gładsze granice między wycinkami, ale dłuższe liczenie. "
                         "0 = najszybciej, 0.5 = najgładziej.",
                )
                use_voting = st.checkbox(
                    "Łączenie wycinków (Gaussian voting)",
                    value=True,
                    disabled=(overlap == 0),
                    help="Łączy wyniki z nakładających się wycinków (im bliżej brzegu wycinka, "
                         "tym mniejsza waga). "
                         "Przy nakładaniu 0 nieaktywne.",
                )
                n_tiles = max(_tile_counts(img_np.shape, selected_models, selected_res, source_res, overlap))
                st.caption(f"Przy tych ustawieniach: do {n_tiles} wycinków na zdjęcie.")
        else:
            st.caption("Zdjęcie mieści się w rozmiarze wejściowym modelu, więc podział na mniejsze fragmenty nie jest potrzebny.")

    # skip resolutions this image cannot be downscaled to without being deformed
    # (e.g. a small 25 cm/px crop cannot become a 10 m/px image)
    unavailable = {}
    for res_label in selected_res:
        ok, reason = check_rescale(img_np.shape, calculate_actual_scale(res_label, source_res))
        if not ok:
            unavailable[res_label] = reason

    if unavailable:
        for res_label, reason in unavailable.items():
            st.warning(f"**{res_label}** — pominięto: {reason}")
        selected_res = [r for r in selected_res if r not in unavailable]
        if not selected_res:
            st.stop()

    # handle possible errors
    n_m, n_r = len(selected_models), len(selected_res)
    if n_m == 0 or n_r == 0:
        st.warning("Wybierz co najmniej jeden model i jedną rozdzielczość.")
        st.stop()
    elif n_m > 1 and n_r > 1:
        st.warning("Wybrano wiele modeli **i** wiele rozdzielczości. "
                   "Wybierz **jeden model** lub **jedną rozdzielczość**.")
        st.stop()
    elif n_m == 1 and n_r > 1:
        st.info(f"Tryb: **porównanie rozdzielczości** — model *{selected_models[0]}*, {n_r} rozdzielczości")
    elif n_m > 1 and n_r == 1:
        st.info(f"Tryb: **porównanie modeli** — {n_m} modeli, rozdzielczość *{selected_res[0]}*")
    else:
        st.info(f"Tryb: **pojedynczy wynik** — *{selected_models[0]}*, *{selected_res[0]}*")

    st.markdown("---")
    st.subheader("Wyniki segmentacji")

    if st.button("Uruchom segmentację", type="primary", width="stretch"):
        st.session_state["seg_running"] = True

    running = st.session_state.get("seg_running", False)

    if running:
        if st.button("Zatrzymaj", width="stretch"):
            st.session_state["seg_running"] = False
            st.warning("Segmentacja została zatrzymana przez użytkownika.")
            st.stop()

    if not running:
        st.stop()

    # one model, one or more resolutions
    if n_m == 1 and n_r >= 1:
        model_name = selected_models[0]
        is_sat = "bigearthnet" in model_name.lower() or "ben" in model_name.lower()

        with st.spinner(f"Ładowanie modelu {model_name}…"):
            model = load_model(model_name)
            tile_size = MODELS[model_name]["tile"]
            model_overlap = MODELS[model_name].get("overlap", overlap)
            model_voting = MODELS[model_name].get("use_voting", use_voting)

        if model is None:
            st.stop()

        results = []
        for res_label in selected_res:
            actual_scale = calculate_actual_scale(res_label, source_res)

            if actual_scale > 1.5:
                st.caption(
                    f"{res_label}: powiększenie {actual_scale:.1f}× "
                    f"(zdjęcie źródłowe ma niższą rozdzielczość) — jakość może być słaba."
                )

            # rescale & predict
            img_s = rescale(img_np, actual_scale)
            mask = predict(
                model, img_s,
                tile_size=tile_size,
                overlap=model_overlap,
                use_voting=model_voting,
                progress_label=f"[{res_label}]",
            )
            if mask is None:
                continue

            results.append({"label": res_label, "img": img_s, "mask": mask})

        if not results:
            st.stop()

        if n_r == 1:
            r = results[0]
            if mask_gt is not None:
                render_gt_comparison(
                    r["img"], r["mask"], mask_gt,
                    label=f"{model_name} · {r['label']} · {r['img'].shape[1]}×{r['img'].shape[0]} px"
                )
            else:
                render_result_block(
                    r["img"], r["mask"],
                    label=f"{model_name} · {r['label']} · {r['img'].shape[1]}×{r['img'].shape[0]} px"
                )
        else:
            render_resolution_comparison(results, mask_gt=mask_gt)

    # more models, one resolution
    elif n_r == 1 and n_m > 1:
        res_label = selected_res[0]
        actual_scale = calculate_actual_scale(res_label, source_res)

        if actual_scale > 1.5:
            st.caption(f"{res_label}: powiększenie {actual_scale:.1f}× — jakość może być słaba.")

        model_results = {}
        for model_name in selected_models:
            ts = MODELS[model_name]["tile"]
            model_overlap = MODELS[model_name].get("overlap", overlap)
            model_voting = MODELS[model_name].get("use_voting", use_voting)
            is_sat = "bigearthnet" in model_name.lower() or "ben" in model_name.lower()

            with st.spinner(f"Ładowanie {model_name}…"):
                model = load_model(model_name)
            if model is None:
                continue

            # rescale & predict
            img_s = rescale(img_np, actual_scale)
            mask = predict(
                model, img_s,
                tile_size=ts,
                overlap=model_overlap,
                use_voting=model_voting,
                progress_label=f"[{model_name}]"
            )
            if mask is not None:
                model_results[model_name] = mask

        if model_results:
            img_input = rescale(img_np, actual_scale)
            render_model_comparison(img_input, model_results, mask_gt=mask_gt)

    st.session_state["seg_running"] = False
