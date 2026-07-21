import streamlit as st

# keys describing the CURRENTLY selected image and its source — cleared on every source change
_RESET_KEYS = [
    "img_np",
    "img_source",
    "mask_gt",
    "chosen",
    "chosen_ds",
    "chosen_dataset_type",
    "_map_img_np",
    "_map_raw_bytes",
    "_map_caption",
    "_map_filename",
    "_map_actual_res",
    "_map_native_label",
    "_upload_img_np",
    "_upload_raw_name",
    "_upload_detected_res",
    "seg_running",   # interrupts/clears the "segmentation running" state on image change
]


def reset_image_state():
    """Clears all image state from the previous image."""
    for key in _RESET_KEYS:
        st.session_state.pop(key, None)
    st.session_state["mask_gt"] = None
    st.session_state["source_res"] = 0.25

    # new state for new image
    st.session_state["selection_gen"] = st.session_state.get("selection_gen", 0) + 1

    # remove the previous state (if it was native)
    resolutions = st.session_state.get("resolutions", {})
    for label in [l for l in resolutions if l.startswith("Natywna (")]:
        del resolutions[label]


def set_native_resolution(value: float) -> str:
    """
    Stores `value` as the ONLY "Native (...)" label in the resolutions dict
    (removing the previous one, if any) and returns its name.
    Used by upload_tab.py (resolution detected from metadata) and
    map_tab.py (resolution of the downloaded orthophoto fragment).
    """
    label = f"Natywna ({value:.2f} m/px)"
    resolutions = st.session_state.setdefault("resolutions", {})
    for old in [l for l in resolutions if l.startswith("Natywna (")]:
        del resolutions[old]
    resolutions[label] = round(value, 4)
    return label