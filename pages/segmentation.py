import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODELS, RESOLUTIONS, LANDCOVER_MODELS, BIGEARTHNET_MODELS, TILE_OVERLAP, USE_GAUSSIAN_VOTING
from utils.models import load_model
from utils.inference import rescale, predict, check_rescale
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

/* 1. Wyrównanie komunikatów do lewej */
div[data-testid="stAlert"] {
    max-width: 700px;
    margin: 0 auto 1rem 0 !important;
    border-radius: 10px;
}

/* 2. Równe przyciski (Wyśrodkowanie i szerokość) */
div[data-testid="stButton"],
div[data-testid="stDownloadButton"] {
    display: flex;
    justify-content: center;
    width: 100%;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    width: 100% !important;
    min-height: 46px !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease;
}

/* 3. Przycisk główny (Primary) - PÓŁPRZEZROCZYSTY NIEBIESKI (Zapisz, Uruchom) */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stDownloadButton"] > button[kind="primary"] {
    background-color: rgba(59, 130, 246, 0.15) !important; /* Półprzezroczysty niebieski */
    color: #60a5fa !important; /* Jasnoniebieski tekst */
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
    background-color: #2563eb !important; /* Pełny niebieski po najechaniu */
    color: white !important;
}

/* 4. Przycisk poboczny (Secondary) - PRZEZROCZYSTY (Wybierz) */
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stDownloadButton"] > button[kind="secondary"] {
    background-color: transparent !important;
    color: inherit !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover,
div[data-testid="stDownloadButton"] > button[kind="secondary"]:hover {
    border-color: #8bb4f7 !important;
    color: #8bb4f7 !important;
}

/* 5. Przycisk trzeciorzędny (Tertiary) - CZERWONY (Usuń, Zatrzymaj) */
div[data-testid="stButton"] > button[kind="tertiary"],
div[data-testid="stDownloadButton"] > button[kind="tertiary"] {
    background-color: rgba(239, 68, 68, 0.15) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
}
div[data-testid="stButton"] > button[kind="tertiary"]:hover,
div[data-testid="stDownloadButton"] > button[kind="tertiary"]:hover {
    background-color: #ef4444 !important;
    color: white !important;
}
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
    st.info("Wybierz zdjęcie, aby przejść dalej.")
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

    # available_resolutions = list(st.session_state["resolutions"].keys())
    available_resolutions = sorted(st.session_state["resolutions"], key=st.session_state["resolutions"].get)
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

    col_m, col_r = st.columns(2)

    # Choose model(s)
    with col_m:
        st.markdown("**Model**")
        st.caption("Wybierz jeden lub kilka do porównania")
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
        st.markdown("**Rozdzielczość**")
        st.caption("Wybierz jedną lub kilka do porównania")
        selected_res = st.multiselect(
            "Rozdzielczości", available_resolutions,
            default=default_res,
            label_visibility="collapsed",
            key=f"selected_res_{st.session_state.get('selection_gen', 0)}"
        )
    selected_res = sorted(selected_res, key=lambda label: st.session_state["resolutions"][label])

    # --- Validate the selection ------------------------------------------
    # Every problem found is collected in `blockers`. While the list is not
    # empty the "Uruchom segmentację" button is drawn but disabled. This
    # replaces the old approach (silently dropping options + st.stop() before
    # the button was even drawn).
    n_m, n_r = len(selected_models), len(selected_res)
    blockers = []

    if n_m == 0 or n_r == 0:
        blockers.append("Wybierz co najmniej jeden model i jedną rozdzielczość.")

    if n_m > 1 and n_r > 1:
        blockers.append(
            "Wybrano wiele modeli **i** wiele rozdzielczości. "
            "Wybierz **jeden model** lub **jedną rozdzielczość**."
        )

    # Checked on the ORIGINAL selection, before anything is removed. Before,
    # unreachable resolutions were dropped first, so "2 models + 2 resolutions"
    # (one of them unreachable) shrank to "2 models + 1 resolution" and slipped
    # through the check above.
    for res_label in selected_res:
        ok, reason = check_rescale(img_np.shape, calculate_actual_scale(res_label, source_res))
        if not ok:
            blockers.append(
                f"**{res_label}** — nie można przeskalować zdjęcia: {reason} "
                f"Odznacz tę rozdzielczość, aby uruchomić segmentację."
            )

    for msg in blockers:
        st.warning(msg)

    # mode info only makes sense for a valid selection
    if not blockers:
        if n_m == 1 and n_r > 1:
            st.info(f"Tryb: **porównanie rozdzielczości** — model *{selected_models[0]}*, {n_r} rozdzielczości")
        elif n_m > 1 and n_r == 1:
            st.info(f"Tryb: **porównanie modeli** — {n_m} modeli, rozdzielczość *{selected_res[0]}*")
        else:
            st.info(f"Tryb: **pojedynczy wynik** — *{selected_models[0]}*, *{selected_res[0]}*")

    st.markdown("---")
    st.subheader("Wyniki segmentacji")

    # Clear old results if parameters change (uses the original selection)
    current_params = f"{selected_models}_{selected_res}"
    if st.session_state.get("last_params") != current_params:
        st.session_state.pop("seg_results", None)
        st.session_state["last_params"] = current_params

    # Start button logic - disabled while the selection is invalid
    if st.button("Uruchom segmentację", type="primary",
                 disabled=bool(blockers)):
        st.session_state["seg_running"] = True
        st.session_state.pop("seg_results", None)

    if blockers:
        # Invalid selection: make sure nothing keeps "running" and no result
        # computed for an older, valid selection stays on screen.
        st.session_state["seg_running"] = False
        st.session_state.pop("seg_results", None)
        st.stop()

    running = st.session_state.get("seg_running", False)

    # Placeholder for the Stop button (only visible during execution)
    stop_button_placeholder = st.empty()

    if running and "seg_results" not in st.session_state:
        with stop_button_placeholder:
            if st.button("Zatrzymaj", type="tertiary"):
                st.session_state["seg_running"] = False
                st.warning("Segmentacja została zatrzymana przez użytkownika.")
                st.stop()

    # Stop execution if not running and no results available
    if not running and "seg_results" not in st.session_state:
        st.stop()

    # ==========================================
    # 1. INFERENCE EXECUTION
    # ==========================================
    if "seg_results" not in st.session_state:

        # one model, one or more resolutions
        if n_m == 1 and n_r >= 1:
            model_name = selected_models[0]

            with st.spinner(f"Ładowanie modelu {model_name}…"):
                model = load_model(model_name)
                tile_size = MODELS[model_name]["tile"]
                model_overlap = MODELS[model_name].get("overlap", TILE_OVERLAP)
                model_voting = MODELS[model_name].get("use_voting", USE_GAUSSIAN_VOTING)

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
                if mask is not None:
                    results.append({"label": res_label, "img": img_s, "mask": mask})

            if not results:
                st.stop()

            # Save results to session state
            st.session_state["seg_results"] = {"type": "res_compare", "model": model_name, "data": results}

        # more models, one resolution
        elif n_r == 1 and n_m > 1:
            res_label = selected_res[0]
            actual_scale = calculate_actual_scale(res_label, source_res)

            if actual_scale > 1.5:
                st.caption(f"{res_label}: powiększenie {actual_scale:.1f}× — jakość może być słaba.")

            model_results = {}
            for model_name in selected_models:
                ts = MODELS[model_name]["tile"]
                model_overlap = MODELS[model_name].get("overlap", TILE_OVERLAP)
                model_voting = MODELS[model_name].get("use_voting", USE_GAUSSIAN_VOTING)

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
                # Save results to session state
                st.session_state["seg_results"] = {"type": "model_compare", "img_input": img_input,
                                                   "data": model_results}

        # Clear running flag and remove the stop button immediately after inference
        st.session_state["seg_running"] = False
        stop_button_placeholder.empty()

    # ==========================================
    # 2. RENDERING RESULTS & EXPORT
    # ==========================================
    if "seg_results" in st.session_state:
        res = st.session_state["seg_results"]

        # --- Display Results ---
        if res["type"] == "res_compare":
            data = res["data"]
            model_name = res["model"]
            if len(data) == 1:
                r = data[0]
                label = f"{model_name} · {r['label']} · {r['img'].shape[1]}×{r['img'].shape[0]} px"
                if mask_gt is not None:
                    render_gt_comparison(r["img"], r["mask"], mask_gt, label=label)
                else:
                    render_result_block(r["img"], r["mask"], label=label)
            else:
                render_resolution_comparison(data, mask_gt=mask_gt)

        elif res["type"] == "model_compare":
            render_model_comparison(res["img_input"], res["data"], mask_gt=mask_gt)
