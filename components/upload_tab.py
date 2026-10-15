import streamlit as st

from utils.display import load_image_from_upload
from utils.session import reset_image_state, set_native_resolution

PREVIEW_WIDTH = 400


def _clear_upload() -> None:
    """Clear the uploaded file, reset session keys, and refresh the file uploader widget."""
    if st.session_state.get("img_source") == "upload":
        reset_image_state()

    keys_to_clear = [
        "_upload_img_np",
        "_upload_raw_name",
        "_upload_detected_res",
        "_upload_processed_file_id",
        "_upload_manual_res",
        "_upload_native_label",
        "upload_manual_res_input",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    # Increment key to completely reset the st.file_uploader widget
    st.session_state["_upload_widget_key"] = st.session_state.get("_upload_widget_key", 0) + 1


def render_upload_tab() -> None:
    """
    Main function to render the upload tab (component for uploading user's own images (JPG/PNG/TIFF)
    Detects resolution from metadata or asks the user to provide this information when it is missing)
    """

    # Upload your own image + detect resolution from the metadata (if available)
    widget_key = f"upload_file_{st.session_state.get('_upload_widget_key', 0)}"
    uploaded = st.file_uploader(
        "Wgraj zdjęcie (JPG, PNG, TIF):",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key=widget_key,
    )

    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{uploaded.size}"

        # Process the file iff new one was uploaded
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
                    native_label = set_native_resolution(detected_res)
                    st.session_state["_upload_native_label"] = native_label
                else:

                    # If no metadata, specify source_res
                    st.session_state["source_res"] = None

    # Preview
    if st.session_state.get("_upload_img_np") is not None:
        img_np = st.session_state["_upload_img_np"]
        detected_res = st.session_state.get("_upload_detected_res")

        c_img, c_info = st.columns([3, 1])
        with c_img:
            st.image(
                img_np,
                caption=f"{st.session_state.get('_upload_raw_name', '')} — "
                        f"{img_np.shape[1]}×{img_np.shape[0]} px",
                width=PREVIEW_WIDTH,
            )
        with c_info:
            if detected_res:
                st.success(f"Wykryta rozdzielczość:\n**{detected_res:.2f} m/px**")
            else:
                st.warning("Nie wykryto rozdzielczości z metadanych.")
                st.markdown("**Podaj rozdzielczość źródłową (m/px):**")

                # Manual resolution
                manual_str = st.text_input(
                    "Rozdzielczość (m/px)",
                    value=str(st.session_state.get("_upload_manual_res", "")),
                    placeholder="np. 0.25",
                    label_visibility="collapsed",
                    key="upload_manual_res_input",
                )

                if manual_str.strip():
                    try:
                        manual_res = float(manual_str.replace(",", "."))
                        if manual_res <= 0:
                            raise ValueError("Wartość musi być dodatnia")

                        if st.session_state.get("_upload_manual_res") != manual_res:
                            st.session_state["_upload_manual_res"] = manual_res
                            st.session_state["source_res"] = manual_res
                            native_label = set_native_resolution(manual_res)
                            st.session_state["_upload_native_label"] = native_label
                            st.session_state["img_np"] = img_np
                        st.success(f"Ustawiono: **{manual_res} m/px**")
                    except ValueError:
                        st.error("Podaj liczbę dodatnią, np. `0.25` lub `10`")
                else:
                    st.caption("Brak rozdzielczości — podaj wartość żeby móc uruchomić segmentację.")

        if st.button("🗑️ Usuń wgrane zdjęcie", width='stretch'):
            _clear_upload()
            st.rerun()
