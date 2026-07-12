"""
upload_tab.py
-------------
Fragment do wklejenia do segmentation.py jako zakładka wgrywania własnego zdjęcia.

Użycie w segmentation.py:
    from components.upload_tab import render_upload_tab

    with tab_upload:
        render_upload_tab()
"""

import streamlit as st

from utils.inference import load_image_from_upload
from utils.session import reset_image_state, set_native_resolution


def _clear_upload():
    """
    Usuwa wgrane zdjęcie: dane robocze + sam widget file_uploader.
    Streamlit nie ma metody "clear" dla file_uploader — jedyny sposób to
    podbicie klucza widgetu, żeby dostał świeżą, pustą instancję.
    """
    if st.session_state.get("img_source") == "upload":
        reset_image_state()
    st.session_state.pop("_upload_img_np", None)
    st.session_state.pop("_upload_raw_name", None)
    st.session_state.pop("_upload_detected_res", None)
    st.session_state.pop("_upload_processed_file_id", None)
    st.session_state["_upload_widget_key"] = st.session_state.get("_upload_widget_key", 0) + 1


def render_upload_tab():
    """
    Renderuje zakładkę do wgrywania własnego zdjęcia (JPG/PNG/TIF) i wykrywania
    jego rozdzielczości źródłowej z metadanych (lub ręcznego jej podania).
    """
    widget_key = f"upload_file_{st.session_state.get('_upload_widget_key', 0)}"
    uploaded = st.file_uploader(
        "Wgraj zdjęcie (JPG, PNG, TIF):",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key=widget_key,
    )

    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{uploaded.size}"

        # Przetwarzamy plik TYLKO gdy to naprawdę nowe wgranie — inaczej ten sam
        # plik, który zostaje "przyklejony" do widgetu, wgrywałby się od nowa
        # przy KAŻDYM rerunie strony (np. kliknięciu czegoś w innej zakładce)
        # i przejmował aktywne zdjęcie z powrotem na siebie.
        if st.session_state.get("_upload_processed_file_id") != file_id:
            img_np, detected_res = load_image_from_upload(uploaded)
            if img_np is not None:
                reset_image_state()
                st.session_state["img_source"] = "upload"
                st.session_state["img_np"] = img_np
                st.session_state["_upload_img_np"] = img_np
                st.session_state["_upload_raw_name"] = uploaded.name
                st.session_state["_upload_detected_res"] = detected_res
                st.session_state["_upload_processed_file_id"] = file_id
                if detected_res:
                    st.session_state["source_res"] = detected_res
                    set_native_resolution(detected_res)

    # ── Trwały podgląd wgranego zdjęcia ────────────────────────────────────
    # Renderowany zawsze, dopóki dane są w session_state — znika, gdy user
    # wybierze zdjęcie testowe / pobierze coś z mapy (reset_image_state()
    # czyści "_upload_img_np"), albo kliknie "Usuń wgrane zdjęcie" poniżej.
    if st.session_state.get("_upload_img_np") is not None:
        img_np = st.session_state["_upload_img_np"]

        c_img, c_info = st.columns([3, 1])
        with c_img:
            st.image(
                img_np,
                caption=f"{st.session_state.get('_upload_raw_name', '')} — "
                        f"{img_np.shape[1]}×{img_np.shape[0]} px",
                width="stretch",
            )
        with c_info:
            detected_res = st.session_state.get("_upload_detected_res")
            if detected_res:
                st.success(f"Wykryta rozdzielczość:\n**{detected_res:.4f} m/px**")
            else:
                st.warning("Nie wykryto rozdzielczości z metadanych.")
                res_label = st.selectbox(
                    "Podaj rozdzielczość źródłową:",
                    list(st.session_state["resolutions"].keys()),
                    key="manual_res",
                )
                st.session_state["source_res"] = st.session_state["resolutions"][res_label]

        if st.button("🗑️ Usuń wgrane zdjęcie", width="stretch"):
            _clear_upload()
            st.rerun()