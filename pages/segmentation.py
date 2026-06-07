import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODELS, RESOLUTIONS, SAMPLE_IMAGES, CLASS_HEX, CLASSES
from utils.models import load_model
from utils.inference import rescale, predict, load_image_from_upload
from utils.display import (render_result_block, render_resolution_comparison,
                           render_model_comparison, render_gt_comparison,
                           load_mask_from_path)

st.set_page_config(page_title="Segmentacja", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="stExpander"] { border: 1px solid rgba(128,128,128,0.2); border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.title("Segmentacja semantyczna")
st.markdown("---")

# session state
if "source_res"  not in st.session_state: st.session_state["source_res"]  = 0.25
if "img_np" not in st.session_state: st.session_state["img_np"] = None
if "mask_gt" not in st.session_state: st.session_state["mask_gt"] = None
if "resolutions" not in st.session_state: st.session_state["resolutions"] = dict(RESOLUTIONS)

# choose image
st.subheader("Wybierz zdjęcie")

tab_sample, tab_upload = st.tabs(["📂 Zdjęcia testowe", "📂 Wgraj własne"])

# ready images tab
with tab_sample:
    # datasets division (lc/ben)
    dataset_names = list(SAMPLE_IMAGES.keys())
    inner_tabs = st.tabs([f"🗂 {ds}" for ds in dataset_names])

    for inner_tab, dataset_name in zip(inner_tabs, dataset_names):
        with inner_tab:
            samples = SAMPLE_IMAGES[dataset_name]
            st.caption(f"Zdjęcia testowe ze zbioru **{dataset_name}**")
            cols = st.columns(len(samples))

            for col, (name, info) in zip(cols, samples.items()):
                img_path = info["img"]
                with col:
                    if os.path.exists(img_path):
                        st.image(img_path, width="stretch")
                    else:
                        st.markdown(
                            '<div style="height:100px;background:rgba(128,128,128,0.08);'
                            'border-radius:8px;display:flex;align-items:center;'
                            'justify-content:center;font-size:11px;color:gray">'
                            'brak pliku</div>', unsafe_allow_html=True)
                    st.caption(info["desc"])
                    if st.button("Wybierz", key=f"sample_{dataset_name}_{name}",
                                 width="stretch"):
                        st.session_state["chosen"] = name
                        st.session_state["chosen_ds"] = dataset_name
                        st.session_state["img_source"] = "sample"

    # import image and gt
    chosen = st.session_state.get("chosen")
    chosen_ds = st.session_state.get("chosen_ds")
    if chosen and chosen_ds and st.session_state.get("img_source") == "sample":
        info = SAMPLE_IMAGES[chosen_ds][chosen]
        if os.path.exists(info["img"]):
            st.session_state["img_np"] = np.array(Image.open(info["img"]).convert("RGB"))
            st.session_state["source_res"] = info.get("source_res", 0.25)
            # import gt if exists
            mask_path = info.get("mask")
            if mask_path and os.path.exists(mask_path):
                st.session_state["mask_gt"] = load_mask_from_path(mask_path)
            else:
                st.session_state["mask_gt"] = None
            st.success(f"Wybrano: **{chosen}** ({chosen_ds})")
        else:
            st.error(f"Brak pliku `{info['img']}`.")

# own image tab
with tab_upload:
    uploaded = st.file_uploader(
        "Wgraj zdjęcie (JPG, PNG, TIF):",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    if uploaded:
        img_np, detected_res = load_image_from_upload(uploaded)
        if img_np is not None:
            st.session_state["img_source"] = "upload"
            st.session_state["img_np"] = img_np
            st.session_state["mask_gt"] = None   # no gt for own images

            c_img, c_info = st.columns([3, 1])
            with c_img:
                st.image(img_np,
                         caption=f"{uploaded.name} — {img_np.shape[1]}×{img_np.shape[0]} px",
                         width="stretch")
            with c_info:
                if detected_res:
                    st.success(f"Wykryta rozdzielczość:\n**{detected_res:.4f} m/px**")
                    st.session_state["source_res"] = detected_res
                else:
                    st.warning("Nie wykryto rozdzielczości z metadanych.")
                    res_label = st.selectbox(
                        "Podaj rozdzielczość źródłową:",
                        list(st.session_state["resolutions"].keys()),
                        key="manual_res"
                    )
                    st.session_state["source_res"] = st.session_state["resolutions"][res_label]

img_np = st.session_state.get("img_np")
mask_gt = st.session_state.get("mask_gt")
source_res = st.session_state.get("source_res", 0.25)

if img_np is not None and st.session_state.get("img_source") == "sample":
    st.image(img_np,
             caption=f"Wybrane zdjęcie — {img_np.shape[1]}×{img_np.shape[0]} px "
                     f"({source_res} m/px)",
             width=400)
    if mask_gt is not None:
        st.success("Maska Ground Truth dostępna — wyniki będą zawierać metryki IoU/F1.")
    else:
        st.info("Brak maski GT dla tego zdjęcia.")

if img_np is None:
    st.info("Wybierz zdjęcie testowe lub wgraj własne aby przejść dalej.")
    st.stop()

st.markdown("---")

# settings/parameters
st.subheader("Wybierz ustawienia")

available_resolutions = list(st.session_state["resolutions"].keys())
default_res = [available_resolutions[0]]

if st.session_state.get("img_source") == "upload" and source_res != 0.25:
    native_label = f"Natywna ({source_res:.4f} m/px)"
    if native_label not in st.session_state["resolutions"]:
        st.session_state["resolutions"][native_label] = source_res
        available_resolutions = list(st.session_state["resolutions"].keys())
    if native_label not in available_resolutions:
        available_resolutions.insert(0, native_label)
    default_res = [native_label]

col_m, col_r, col_v = st.columns([2, 2, 1])

with col_m:
    st.markdown("**Model / Modele**")
    st.caption("Wybierz jeden (porównanie rozdzielczości) lub wiele (porównanie modeli)")
    selected_models = st.multiselect(
        "Modele", list(MODELS.keys()),
        default=[list(MODELS.keys())[0]],
        label_visibility="collapsed"
    )

with col_r:
    st.markdown("**Rozdzielczość / Rozdzielczości**")
    st.caption("Wybierz jedną lub więcej do porównania")
    selected_res = st.multiselect(
        "Rozdzielczości", available_resolutions,
        default=default_res,
        label_visibility="collapsed"
    )

with col_v:
    st.markdown("**Nakładka**")
    alpha = st.slider("Przezroczystość:", 0.0, 1.0, 0.45, 0.05,
                      label_visibility="collapsed")
    global_overlap = st.slider("Overlap kafelków:", 0.0, 0.5, 0.0, 0.1,
                               help="Używane jeśli model nie narzuca własnego overlapu w config.py")

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

# segmentation
st.subheader("Wyniki segmentacji")

if not st.button("Uruchom segmentację", type="primary", width="stretch"):
    st.stop()

def calculate_actual_scale(res_label: str, source_resolution_m: float) -> float:
    target_res_m = st.session_state["resolutions"][res_label]
    return source_resolution_m / target_res_m

# one model, few scales
if n_m == 1 and n_r >= 1:
    model_name = selected_models[0]
    with st.spinner(f"Ładowanie modelu {model_name}…"):
        model = load_model(model_name)
        tile_size = MODELS[model_name]["tile"]
        overlap = MODELS[model_name].get("overlap", global_overlap)

    if model is None:
        st.stop()

    results = []
    for res_label in selected_res:
        actual_scale = calculate_actual_scale(res_label, source_res)
        img_s = rescale(img_np, actual_scale, tile_size)
        mask = predict(model, img_s, tile_size=tile_size, overlap=overlap,
                        progress_label=f"[{res_label}]")
        if mask is None:
            continue
        results.append({"label": res_label, "img": img_s, "mask": mask, "metrics": None})

    if not results:
        st.stop()

    if n_r == 1:
        r = results[0]
        if mask_gt is not None:
            # compare result with gt
            render_gt_comparison(r["img"], r["mask"], mask_gt, alpha,
                                 label=f"{model_name} · {r['label']} · "
                                       f"{r['img'].shape[1]}×{r['img'].shape[0]} px")
        else:
            render_result_block(r["img"], r["mask"], alpha,
                                label=f"{model_name} · {r['label']} · "
                                      f"{r['img'].shape[1]}×{r['img'].shape[0]} px")
    else:
        # more scales
        render_resolution_comparison(results, alpha, mask_gt=mask_gt)

# a couple of models, one scale
elif n_r == 1 and n_m > 1:
    res_label = selected_res[0]
    actual_scale = calculate_actual_scale(res_label, source_res)

    model_results = {}
    for model_name in selected_models:
        ts = MODELS[model_name]["tile"]
        overlap = MODELS[model_name].get("overlap", global_overlap)

        with st.spinner(f"Ładowanie {model_name}…"):
            model = load_model(model_name)
        if model is None:
            continue

        img_s = rescale(img_np, actual_scale, ts)
        mask = predict(model, img_s, tile_size=ts, overlap=overlap,
                        progress_label=f"[{model_name}]")
        if mask is not None:
            model_results[model_name] = mask

    if model_results:
        first_ts = MODELS[selected_models[0]]["tile"]
        img_input = rescale(img_np, actual_scale, first_ts)
        render_model_comparison(img_input, model_results, alpha, mask_gt=mask_gt)
