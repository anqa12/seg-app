import torch
import numpy as np
import cv2
import streamlit as st
from utils.config import NUM_CLASSES


def _gaussian_window(size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """Generates a 2D Gaussian window to blend overlapping tile predictions."""
    sigma = size * sigma_ratio
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size / 2.0, size / 2.0
    window = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return (window / window.max()).astype(np.float32)


_GAUSS_CACHE = {}


def _get_window(tile_size: int) -> np.ndarray:
    """Retrieves or compute a cached Gaussian window for a given tile size."""
    if tile_size not in _GAUSS_CACHE:
        _GAUSS_CACHE[tile_size] = _gaussian_window(tile_size)
    return _GAUSS_CACHE[tile_size]


def _align_to_32(size: float) -> int:
    """Rounds a side length to the NEAREST multiple of 32 (never below 32)."""
    return max(32, int(size / 32 + 0.5) * 32)


def rescale(img: np.ndarray, scale: float) -> np.ndarray:
    """Rescales the image and aligns each side to a multiple of 32."""
    H, W = img.shape[:2]
    return cv2.resize(img, (_align_to_32(W * scale), _align_to_32(H * scale)), interpolation=cv2.INTER_AREA)


# smallest side (in px) the image may have after downscaling
MIN_SIDE_PX = 32

# largest change of a side (0.10 = 10 %) allowed when it is aligned to a multiple of 32
MAX_STRETCH_FRACTION = 0.10


def check_rescale(img_shape: tuple, scale: float) -> tuple[bool, str]:
    """Checks whether the image can be rescaled by `scale` without being deformed."""
    if scale > 1.05:
        return False, (
            "to zdjęcie ma gorszą rozdzielczość niż wybrana, a powiększanie "
            "nie doda do niego nowych szczegółów."
        )

    H, W = img_shape[:2]
    H_s, W_s = H * scale, W * scale

    if min(int(H_s), int(W_s)) < MIN_SIDE_PX:
        return False, (
            f"to zdjęcie jest za małe na taką rozdzielczość — po przeliczeniu miałoby tylko "
            f"{int(W_s)} × {int(H_s)} px, a model potrzebuje co najmniej {MIN_SIDE_PX} px na bok."
        )

    # how much the sides change when aligned to a multiple of 32 (worst of the two)
    stretch = max(abs(_align_to_32(H_s) / H_s - 1), abs(_align_to_32(W_s) / W_s - 1))
    if stretch > MAX_STRETCH_FRACTION:
        return False, (
            f"po przeliczeniu zdjęcie miałoby {int(W_s)} × {int(H_s)} px, a model wymaga wymiarów "
            f"podzielnych przez 32 — dopasowanie rozciągnęłoby obraz o {stretch:.0%} i pogorszyło wynik."
        )
    return True, ""


def tile_layout(H: int, W: int, tile_size: int, overlap: float) -> tuple[int, list[int], list[int]]:
    """
    Work out how predict() cuts an H x W image into tiles.
    """
    tile_size = (min(tile_size, H, W) // 32) * 32
    if tile_size <= 0:
        return 0, [], []

    stride = max(1, int(tile_size * (1 - overlap)))

    def get_coords(length):
        coords = list(range(0, length - tile_size + 1, stride))
        if not coords or coords[-1] + tile_size < length:
            coords.append(max(0, length - tile_size))
        return coords

    return tile_size, get_coords(H), get_coords(W)


def predict(
        model: torch.nn.Module | None,
        img_np: np.ndarray,
        tile_size: int,
        overlap: float = 0.5,
        use_voting: bool = True,
        progress_label: str = "",
        batch_size: int = 8,
) -> np.ndarray | None:
    """
    Performs semantic segmentation prediction on a large image by splitting it
    into smaller overlapping tiles and stitching the outputs back together.
    """
    if model is None:
        return None

    model.eval()
    H, W = img_np.shape[:2]

    tile_size, ys, xs = tile_layout(H, W, tile_size, overlap)
    if tile_size <= 0:
        st.warning(f"Zdjęcie ({W}×{H} px) za małe dla segmentacji.")
        return None

    total = len(ys) * len(xs)

    if total == 0:
        st.warning(f"Zdjęcie ({W}×{H} px) za małe dla wycinka {tile_size}×{tile_size} px.")
        return None

    device = next(model.parameters()).device

    if not use_voting:
        pred_full = np.zeros((H, W), dtype=np.uint8)
    else:
        votes = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
        window = _get_window(tile_size)

    bar = st.progress(0, text=progress_label or "Segmentacja w trakcie")
    done = 0

    # all tile positions in reading order; processed batch_size at a time
    coords = [(y, x) for y in ys for x in xs]

    for start in range(0, total, batch_size):
        batch = coords[start:start + batch_size]

        # cut the tiles of this batch (edge tiles are padded up to the full tile size)
        tiles = []
        for y, x in batch:
            tile = img_np[y:y + tile_size, x:x + tile_size]

            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            tiles.append(tile)

        # (B, H, W, 3) uint8  ->  (B, 3, H, W) float in 0..1, on the model's device
        tensor = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.to(device)

        # inference_mode = no_grad
        with torch.inference_mode():
            logits = model(tensor)
            out = torch.softmax(logits, dim=1) if use_voting else logits.argmax(dim=1)
        # (B, C, h, w) with voting, (B, h, w) without
        out = out.cpu().numpy()

        for (y, x), res in zip(batch, out):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            h_real, w_real = y_end - y, x_end - x

            if not use_voting:
                pred_full[y:y_end, x:x_end] = res[:h_real, :w_real]
            else:
                w_real_window = window[:h_real, :w_real]
                votes[:, y:y_end, x:x_end] += res[:, :h_real, :w_real] * w_real_window[np.newaxis]
                counts[y:y_end, x:x_end] += w_real_window

        done += len(batch)
        bar.progress(done / total, text=f"{progress_label} {done}/{total} wycinków")

    bar.empty()

    if not use_voting:
        return pred_full
    else:
        counts = np.maximum(counts, 1e-8)
        return (votes / counts[np.newaxis]).argmax(axis=0).astype(np.uint8)


def predict_all_models(
        models_cfg: list[dict],
        img_np: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run segmentation prediction across multiple models on the same input image."""
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
            model,
            img_np,
            tile_size,
            overlap=overlap,
            use_voting=use_voting,
            progress_label=f"[{name}]",
        )
        if pred is not None:
            all_preds[name] = pred

    return all_preds