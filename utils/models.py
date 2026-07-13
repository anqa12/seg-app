import torch
import streamlit as st
import segmentation_models_pytorch as smp
from utils.config import NUM_CLASSES, MODELS


def _build_model(encoder_key: str):
    """Build architecture based on key from config.MODELS."""
    if encoder_key == "unet_resnet34":
        return smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=NUM_CLASSES,
        )
    elif encoder_key == "deeplab_resnet34":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=NUM_CLASSES,
        )
    elif encoder_key == "segformer_mit_b2":
        return smp.Segformer(
            encoder_name="mit_b2", encoder_weights=None,
            in_channels=3, classes=NUM_CLASSES,
        )
    else:
        raise ValueError(f"Nieznany encoder: {encoder_key}")


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Builds and loads models weights. Done with caching - each model load only once per session.
    """
    cfg = MODELS[model_name]
    model = _build_model(cfg["encoder"])
    try:
        state = torch.load(cfg["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Nie znaleziono wag: `{cfg['path']}`")
        return None
    except Exception as e:
        st.error(f"Błąd ładowania modelu `{model_name}`: {e}")
        return None
