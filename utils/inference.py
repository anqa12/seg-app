import torch
import numpy as np
import cv2
import streamlit as st
from scipy.special import softmax as scipy_softmax
from utils.config import NUM_CLASSES


def load_mask_from_path(path: str) -> np.ndarray | None:
    """
    Wczytuje maskę GT z pliku PNG (wartości pikseli = indeksy klas 0–4).
    Obsługuje zarówno maski w skali szarości jak i RGB z CLASS_COLORS.
    """
    from PIL import Image
    try:
        m = np.array(Image.open(path))
        if m.ndim == 3:
            m = m[:, :, 0]   # tylko pierwszy kanał — indeks klasy
        return m.astype(np.uint8)
    except Exception as e:
        st.warning(f"Nie można wczytać maski GT: {e}")
        return None


def load_image_from_upload(uploaded_file) -> tuple[np.ndarray | None, float | None]:
    # (Zachowaj istniejącą logikę ładowania pliku bez zmian)
    import tempfile, os
    from PIL import Image
    import io

    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    res_m = None

    try:
        if name.endswith((".tif", ".tiff")):
            import rasterio
            from rasterio.plot import reshape_as_image
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp.write(raw);
            tmp.close()
            with rasterio.open(tmp.name) as src:
                img = reshape_as_image(src.read())
                try:
                    res_m = abs(src.transform[0])
                except Exception:
                    pass
            os.unlink(tmp.name)
        else:
            img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3].astype(np.uint8), res_m

    except Exception as e:
        st.error(f"Błąd wczytywania pliku: {e}")
        return None, None