"""
sample_tab.py
-------------
Fragment do wklejenia do segmentation.py jako pierwsza zakładka.

Użycie w segmentation.py:
    from pages.sample_tab import render_sample_tab

    with tab_sample:
        render_sample_tab()
"""

import os
import numpy as np
import streamlit as st
from PIL import Image

from utils.config import SAMPLE_IMAGES
from utils.display import load_mask_from_path


def render_sample_tab():
    """
    Renderuje zakładkę z gotowymi zdjęciami testowymi pogrupowanymi wg zbioru danych.
    """
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

    chosen = st.session_state.get("chosen")
    chosen_ds = st.session_state.get("chosen_ds")
    if chosen and chosen_ds and st.session_state.get("img_source") == "sample":
        info = SAMPLE_IMAGES[chosen_ds][chosen]
        if os.path.exists(info["img"]):
            st.session_state["img_np"] = np.array(Image.open(info["img"]).convert("RGB"))
            st.session_state["source_res"] = info.get("source_res", 0.25)
            st.session_state["chosen_dataset_type"] = chosen_ds

            mask_path = info.get("mask")
            if mask_path and os.path.exists(mask_path):
                st.session_state["mask_gt"] = load_mask_from_path(mask_path)
            else:
                st.session_state["mask_gt"] = None
            st.success(f"Wybrano: **{chosen}** ({chosen_ds})")
        else:
            st.error(f"Brak pliku `{info['img']}`.")