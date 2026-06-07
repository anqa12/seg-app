import torch
import numpy as np
import cv2
import streamlit as st
from scipy.special import softmax as scipy_softmax
from utils.config import NUM_CLASSES
import tempfile, os
from PIL import Image
import io


# gaussian blending
def _gaussian_window(size, sigma_ratio=0.25):
    sigma = size * sigma_ratio
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size / 2.0, size / 2.0
    window = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return (window / window.max()).astype(np.float32)


_GAUSS_CACHE = {}


def _get_window(tile_size):
    if tile_size not in _GAUSS_CACHE:
        _GAUSS_CACHE[tile_size] = _gaussian_window(tile_size)
    return _GAUSS_CACHE[tile_size]


def rescale(img: np.ndarray, scale: float, tile_size: int) -> np.ndarray:
    """Rescales the image and crops to the tile_size which is the multiple of 32"""
    H, W = img.shape[:2]
    H_s = max(32, int(H * scale))
    W_s = max(32, int(W * scale))
    img_s = cv2.resize(img, (W_s, H_s), interpolation=cv2.INTER_AREA)

    # multiple of 32 (for unet)
    H_p = (H_s // 32) * 32
    W_p = (W_s // 32) * 32
    return img_s[:H_p, :W_p]


def predict(model, img_np: np.ndarray, tile_size: int,
            overlap: float = 0.5, progress_label: str = "") -> np.ndarray | None:

    if model is None:
        return None

    model.eval()
    H, W = img_np.shape[:2]

    tile_size = (min(tile_size, H, W) // 32) * 32
    stride = max(1, int(tile_size * (1 - overlap)))

    votes = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    window = _get_window(tile_size)

    def get_coords(length):
        coords = list(range(0, length - tile_size + 1, stride))
        if not coords or coords[-1] + tile_size < length:
            coords.append(length - tile_size)
        return coords

    ys = get_coords(H)
    xs = get_coords(W)
    total = len(ys) * len(xs)

    if total == 0:
        st.warning(f"Zdjęcie ({W}×{H} px) za małe dla kafelka {tile_size}×{tile_size} px.")
        return None

    bar = st.progress(0, text=progress_label or "Segmentuję…")
    done = 0

    for y in ys:
        for x in xs:
            tile = img_np[y:y + tile_size, x:x + tile_size]
            tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)

            # cpu or cuda
            device = next(model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor).squeeze(0).cpu().numpy()

            # softmax + gaussian window weight
            probs = scipy_softmax(logits, axis=0)
            votes[:, y:y + tile_size, x:x + tile_size] += probs * window[np.newaxis]
            counts[y:y + tile_size, x:x + tile_size] += window

            done += 1
            bar.progress(done / total, text=f"{progress_label} {done}/{total} kafelków…")

    bar.empty()
    counts = np.maximum(counts, 1e-8)
    return (votes / counts[np.newaxis]).argmax(axis=0).astype(np.uint8)


def load_image_from_upload(uploaded_file) -> tuple[np.ndarray | None, float | None]:

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