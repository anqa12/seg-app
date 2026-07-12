"""
upload_tab.py
-------------
Fragment do wklejenia do segmentation.py jako zakładka wgrywania własnego zdjęcia.

Użycie w segmentation.py:
    from pages.upload_tab import render_upload_tab

    with tab_upload:
        render_upload_tab()
"""

import streamlit as st

from utils.inference import load_image_from_upload


def render_upload_tab():
    """
    Renderuje zakładkę do wgrywania własnego zdjęcia (JPG/PNG/TIF) i wykrywania
    jego rozdzielczości źródłowej z metadanych (lub ręcznego jej podania).
    """
    uploaded = st.file_uploader(
        "Wgraj zdjęcie (JPG, PNG, TIF):",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    if not uploaded:
        return

    img_np, detected_res = load_image_from_upload(uploaded)
    if img_np is None:
        return

    st.session_state["img_source"] = "upload"
    st.session_state["img_np"] = img_np
    st.session_state["mask_gt"] = None
    st.session_state["chosen_dataset_type"] = None

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