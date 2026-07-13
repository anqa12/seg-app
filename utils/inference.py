import torch
import numpy as np
import cv2
import streamlit as st
from scipy.special import softmax as scipy_softmax
from utils.config import NUM_CLASSES
import tempfile, os
from PIL import Image
import io
import rasterio
from rasterio.plot import reshape_as_image


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
    """Rescale the image to a multiple of 32 (Unet requirement)"""

    H, W = img.shape[:2]
    H_s = max(32, int(H * scale))
    W_s = max(32, int(W * scale))
    img_s = cv2.resize(img, (W_s, H_s), interpolation=cv2.INTER_AREA)

    H_p = (H_s // 32) * 32
    W_p = (W_s // 32) * 32
    return img_s[:H_p, :W_p]


# predic one model
def predict(
    model,
    img_np: np.ndarray,
    tile_size: int,
    overlap: float = 0.5,
    use_voting: bool = True,
    progress_label: str = "",
) -> np.ndarray | None:
    """
    Prediction on large image - partition into tiles

    overlap:    0.0 = none (sharp edges, faster)
                0.5 = 50% overlap (smoother edges)
    use_voting: True  = average probabilities with Gaussian window
                        (recommended when overlap > 0)
                False = the last tile overwrite (faster, overlap not used)
    """
    if model is None:
        return None

    model.eval()
    H, W = img_np.shape[:2]

    tile_size = (min(tile_size, H, W) // 32) * 32
    if tile_size <= 0:
        st.warning(f"Zdjęcie ({W}×{H} px) za małe dla segmentacji.")
        return None

    stride = max(1, int(tile_size * (1 - overlap)))

    def get_coords(length):
        coords = list(range(0, length - tile_size + 1, stride))
        if not coords or coords[-1] + tile_size < length:
            coords.append(max(0, length - tile_size))
        return coords

    ys, xs = get_coords(H), get_coords(W)
    total = len(ys) * len(xs)

    if total == 0:
        st.warning(f"Zdjęcie ({W}×{H} px) za małe dla kafelka {tile_size}×{tile_size} px.")
        return None

    device = next(model.parameters()).device

    if not use_voting:
        pred_full = np.zeros((H, W), dtype=np.uint8)
    else:
        votes = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
        window = _get_window(tile_size)

    bar = st.progress(0, text=progress_label or "Segmentuję…")
    done = 0

    for y in ys:
        for x in xs:
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            tile = img_np[y:y_end, x:x_end]

            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

            tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor).squeeze(0).cpu().numpy()

            h_real, w_real = y_end - y, x_end - x

            if not use_voting:
                pred = logits.argmax(0)
                pred_full[y:y_end, x:x_end] = pred[:h_real, :w_real]
            else:
                probs = scipy_softmax(logits, axis=0)
                w_real_window = window[:h_real, :w_real]
                votes[:, y:y_end, x:x_end] += probs[:, :h_real, :w_real] * w_real_window[np.newaxis]
                counts[y:y_end, x:x_end] += w_real_window

            done += 1
            bar.progress(done / total, text=f"{progress_label} {done}/{total} kafelków…")

    bar.empty()

    if not use_voting:
        return pred_full
    else:
        counts = np.maximum(counts, 1e-8)
        return (votes / counts[np.newaxis]).argmax(axis=0).astype(np.uint8)


# predict more models
def predict_all_models(
    models_cfg: list[dict],
    img_np: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Prediction for multiple models - same image

    models_cfg: list of dicts:
        name       — name of the model
        model      — loaded model Pytorch
        tile_size  —  size of tile (default 512)
        overlap    — (default 0.5)
        use_voting — if use Gaussian window (default True)

    Returns: {model_name: mask_np}
    """
    all_preds = {}
    for cfg in models_cfg:
        name = cfg["name"]
        model = cfg["model"]
        tile_size = cfg.get("tile_size", 512)
        overlap = cfg.get("overlap", 0.5)
        use_voting = cfg.get("use_voting", True)

        if model is None:
            st.warning(f"Pominięto `{name}` — model nie został wczytany.")
            continue

        pred = predict(
            model, img_np, tile_size,
            overlap=overlap, use_voting=use_voting,
            progress_label=f"[{name}]",
        )
        if pred is not None:
            all_preds[name] = pred

    return all_preds


# load image
def load_image_from_upload(uploaded_file) -> tuple[np.ndarray | None, float | None]:
    """Load img from UploadedFile. Returns (img_rgb uint8, resolution or None)"""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    res_m = None

    try:
        if name.endswith((".tif", ".tiff")):
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp.write(raw)
            tmp.close()
            with rasterio.open(tmp.name) as src:
                img = reshape_as_image(src.read())
                # UWAGA: plik BEZ georeferencji (np. TIFF z WMS GetMap, który
                # nie zapisuje ModelPixelScale/GeoKeyDirectory) nie daje w
                # rasterio błędu ani None — GDAL podstawia domyślny "identity"
                # transform (piksel = 1.0 jednostki mapy), co wygląda jak
                # prawdziwa rozdzielczość "1 m/px", a nią nie jest. Dlatego
                # ufamy transformowi TYLKO gdy plik ma faktyczny układ
                # współrzędnych (CRS) I transform nie jest identity — inaczej
                # zostawiamy res_m=None, żeby user sam podał rozdzielczość
                # zamiast dostać zgadywankę.
                has_real_georef = src.crs is not None and not src.transform.is_identity
                if has_real_georef:
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