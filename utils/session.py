import streamlit as st

# keys describing the CURRENTLY selected image and its source — cleared on every source change
_RESET_KEYS: list[str] = [
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
    "_upload_native_label",
    "seg_running",   # clears the "segmentation running" state on image change
]


def reset_image_state() -> None:
    """Clears all image state from the previous image."""
    for key in _RESET_KEYS:
        st.session_state.pop(key, None)
    st.session_state["mask_gt"] = None
    st.session_state["source_res"] = 0.25

    # new state for new image
    st.session_state["selection_gen"] = st.session_state.get("selection_gen", 0) + 1

    # remove the previous state (if it was native)
    resolutions = st.session_state.get("resolutions", {})
    for name in [n for n in resolutions if n.startswith("wykryta:")]:
        del resolutions[name]


def set_native_resolution(value: float) -> str:
    """
    Sets the native resolution of the session state.
    Used by upload_tab.py (resolution detected from metadata) and
    map_tab.py (resolution of the downloaded orthophoto fragment).
    """
    resolutions = st.session_state.setdefault("resolutions", {})

    # remove previous resolution in case no duplicates occur
    for old in [n for n in resolutions if n.startswith("wykryta:")]:
        del resolutions[old]

    # if a resolution with the same value already exists, reuse its label (no duplicates)
    rounded_value = round(value, 2)
    for existing_label, existing_value in resolutions.items():
        if round(existing_value, 2) == rounded_value:
            return existing_label

    label = f"wykryta: {value:.2f} m/px"
    resolutions[label] = round(value, 4)  # for further calculations
    return label
