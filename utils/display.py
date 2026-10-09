import os
import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image

from utils.config import CLASSES, CLASS_COLORS, CLASS_HEX, NUM_CLASSES, OVERLAY_ALPHA


def load_image_from_upload(uploaded_file) -> tuple[np.ndarray | None, float | None]:
    """Load an image from an uploaded Streamlit file object."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    res_m = None

    try:
        if name.endswith((".tif", ".tiff")):
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            try:
                tmp.write(raw)
                tmp.close()
                with rasterio.open(tmp.name) as src:
                    img = reshape_as_image(src.read())
                    if src.crs is not None and not src.transform.is_identity:
                        try:
                            res_m = abs(src.transform[0])
                        except Exception:
                            pass
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
        else:
            img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3].astype(np.uint8), res_m

    except Exception as e:
        st.error(f"Błąd wczytywania pliku: {e}")
        return None, None


def load_mask_from_path(path: str) -> np.ndarray | None:
    """
    Loads a mask from a given path.
    """
    try:
        m = np.array(Image.open(path))
        return (m[:, :, 0] if m.ndim == 3 else m).astype(np.uint8)
    except Exception as e:
        st.warning(f"Nie można wczytać maski GT: {e}")
        return None


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Converts a mask to RGB.
    """
    return CLASS_COLORS[mask]


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlays pred mask with an image.
    """
    h, w = mask.shape
    return ((1 - OVERLAY_ALPHA) * img[:h, :w] + OVERLAY_ALPHA * CLASS_COLORS[mask]).astype(np.uint8)


def class_percentages(mask: np.ndarray) -> dict:
    """
    Calculates class percentages.
    """
    total = mask.size
    if total == 0:
        return {cls: 0.0 for cls in CLASSES}
    return {cls: round((mask == i).sum() / total * 100, 1) for i, cls in enumerate(CLASSES)}


# --- METRICS & BASE RENDERING ---

def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Computes metrics (IOU, F1) when mask is given.
    """
    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    f1 = np.full(NUM_CLASSES, np.nan, dtype=np.float64)

    for c in range(NUM_CLASSES):
        p, g = (pred == c), (gt == c)
        if not p.any() and not g.any():
            continue
        tp = (p & g).sum()
        fp = (p & ~g).sum()
        fn = (~p & g).sum()
        denom = tp + fp + fn
        iou[c] = tp / denom if denom > 0 else 0.0
        f1[c] = (2 * tp) / (denom + tp) if (denom + tp) > 0 else 0.0

    return {"iou": iou, "f1": f1}


def render_legend() -> None:
    """
    Renders the legend with corresponding colors of each class.
    """
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;font-size:13px">'
        f'<span style="width:13px;height:13px;border-radius:3px;background:{CLASS_HEX[i]};border:1px solid rgba(0,0,0,.2)"></span>'
        f'{cls}</span>'
        for i, cls in enumerate(CLASSES)
    )
    st.markdown(f'<div style="margin:8px 0 14px 0">{items}</div>', unsafe_allow_html=True)


def render_coverage_tiles(mask: np.ndarray) -> None:
    """
    Renders coverage tiles with percentage result of each class.
    """
    pct = class_percentages(mask)
    items = list(pct.items())

    def _tile(col, cls, val):
        i = CLASSES.index(cls)
        hex_c = CLASS_HEX[i]
        active = val > 0.05
        opacity = "1" if active else "0.3"
        dot_border = "border:1px solid rgba(255,255,255,.5)" if i == 0 else "border:1px solid rgba(0,0,0,.15)"
        card_border = f"rgba(255,255,255,{'0.6' if active else '0.15'})" if i == 0 else f"{hex_c}{'bb' if active else '33'}"

        col.markdown(
            f"""<div style="background:{hex_c}18; border:2px solid {card_border}; border-radius:10px;
                padding:14px 6px 12px 6px; text-align:center; opacity:{opacity}; margin:4px;">
              <div style="width:14px;height:14px;border-radius:4px;background:{hex_c};margin:0 auto 8px auto;{dot_border}"></div>
              <div style="font-size:10px;color:gray;margin-bottom:4px">{cls}</div>
              <div style="font-size:20px;font-weight:700">{val:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    for col, (cls, val) in zip(st.columns(3), items[:3]):
        _tile(col, cls, val)
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    _, c1, c2, _ = st.columns([0.5, 1, 1, 0.5])
    for col, (cls, val) in zip([c1, c2], items[3:]):
        _tile(col, cls, val)


# Style of a highlighted (best) cell in the metrics tables
_BEST_STYLE = "background-color: rgba(46,160,67,0.22); color: #2EA043; font-weight: 700"


def render_metrics_table(metrics: list[dict], highlight_best: bool = False, best_class_by: str | None = None) -> None:
    """Show a metrics table. Highlights best result - in every metric column the best (highest) value is shown in green,
    all other cells stay plain. Used when rows are alternatives (models, resolutions)
    """
    df = pd.DataFrame(metrics)
    numeric_cols = [c for c in df.columns if c not in ("Klasa", "Rozdzielczość", "Model")]
    rounded = {c: df[c].astype(float).round(2) for c in numeric_cols}

    for c in numeric_cols:
        df[c] = df[c].apply(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}")

    # marks best metric
    def mark_best_per_column(col):
        vals = rounded[col.name]
        best = vals.max()
        return [_BEST_STYLE if (pd.notna(v) and v == best) else "" for v in vals]

    # marks highest score (excluding background)
    def mark_best_class(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        is_class_row = ~df["Klasa"].astype(str).str.startswith("—")
        is_eligible = is_class_row & (df["Klasa"] != CLASSES[0])
        vals = rounded[best_class_by].where(is_eligible)
        winners = vals == vals.max()  # NaN never equals the max, so absent classes are skipped
        styles.loc[winners, ["Klasa", best_class_by]] = _BEST_STYLE
        return styles

    if best_class_by is not None and best_class_by in rounded and "Klasa" in df.columns:
        st.dataframe(df.style.apply(mark_best_class, axis=None), width="stretch", hide_index=True)
    elif highlight_best and len(df) > 1:  # with a single row there is nothing to compare
        st.dataframe(df.style.apply(mark_best_per_column, subset=numeric_cols),
                     width="stretch", hide_index=True)
    else:
        st.dataframe(df, width="stretch", hide_index=True)


# How many predictions are placed side by side before the grid wraps to a new row.
MAX_PRED_COLS = 3

# One column of the comparison grid: (label, input image, predicted mask)
GridItem = tuple[str, np.ndarray, np.ndarray]


def _make_columns(n_used: int) -> list:
    """Create the grid columns and return only the ones that hold content.

    What: with 3 or more content columns they simply share the full width equally.
    With 2 (e.g. just the original and one mask) the two columns get the same width
    that 3 columns would have, and the leftover space is split evenly on both sides.
    Why: the images keep the same size for 2 and 3 columns (no tiny images, no empty
    gap on the right) and the group always looks centered on the page.
    """
    if n_used >= 3:
        return st.columns(n_used)
    cols = st.columns([0.5] + [1] * n_used + [0.5])  # half-width spacers on both sides
    return cols[1:-1]


# Height (px) of the small title above each group of tiles.
# Why: the same fixed height in every column keeps the tiles on the same line.
_COV_HEAD_H = 26


def _fit_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a class mask to (h, w) with nearest-neighbour.

    Why nearest: smoother filters would blend class ids into values that do not
    exist (e.g. between 1 and 3 you would get 2 = a class that is not there).
    """
    h, w = shape
    if mask.shape == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _coverage_tiles_html(title: str, mask: np.ndarray, ref_pct: dict | None = None) -> str:
    """
    Small square tiles with the share of each class, for one mask.
    """
    pct = class_percentages(mask)
    head = (
        f'<div style="height:{_COV_HEAD_H}px;font-size:12px;font-weight:600;white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis" title="{title}">{title}</div>'
    )
    tiles = []
    for i, cls in enumerate(CLASSES):
        val = pct[cls]
        hex_c = CLASS_HEX[i]
        active = val > 0.05
        opacity = "1" if active else "0.35"  # fade classes that are (almost) absent
        # Class 0 is black, so it gets a light border to stay visible on a dark theme
        dot_border = "border:1px solid rgba(255,255,255,.5)" if i == 0 else "border:1px solid rgba(0,0,0,.15)"
        card_border = (
            f"rgba(255,255,255,{'0.6' if active else '0.15'})" if i == 0
            else f"{hex_c}{'bb' if active else '33'}"
        )
        delta = ""
        if ref_pct is not None:
            delta = (
                f'<div style="font-size:9px;font-size:4.2cqw;color:gray;line-height:1.1">'
                f'{val - ref_pct[cls]:+.1f}</div>'
            )
        tiles.append(
            f'<div style="flex:0 0 calc((100% - 12px) / 3);aspect-ratio:1/1;box-sizing:border-box;min-width:0;background:{hex_c}18;'
            f'border:2px solid {card_border};border-radius:10px;opacity:{opacity};display:flex;'
            f'flex-direction:column;align-items:center;justify-content:center;gap:2px;text-align:center">'
            f'<div style="width:9px;height:9px;width:4cqw;height:4cqw;border-radius:3px;'
            f'background:{hex_c};{dot_border}"></div>'
            f'<div style="font-size:9px;font-size:4.4cqw;color:gray">{cls}</div>'
            f'<div style="font-size:15px;font-size:7cqw;font-weight:700;line-height:1.1">{val:.1f}%</div>'
            f'{delta}</div>'
        )

    grid = (
        '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px;width:100%;'
        'max-width:300px;margin:0 auto;container-type:inline-size">' + "".join(tiles) + "</div>"
    )
    return head + grid


def _render_image_row(
        chunk: list[GridItem],
        ref_img: np.ndarray,
        gt_fit: np.ndarray | None,
        n_cols: int,
        as_overlay: bool,
) -> None:
    """
    Renders one row of images: Original | (Ground Truth) | prediction 1 | prediction 2 ...
    """
    cols = _make_columns(n_cols)
    cols[0].image(ref_img, caption="Oryginał", width="stretch")
    first_pred = 1

    if gt_fit is not None:
        gt_view = overlay(ref_img, gt_fit) if as_overlay else mask_to_rgb(gt_fit)
        cols[1].image(gt_view, caption="Ground Truth", width="stretch")
        first_pred = 2

    for col, (label, img, mask) in zip(cols[first_pred:], chunk):
        view = overlay(img, mask) if as_overlay else mask_to_rgb(mask)
        col.image(view, caption=label, width="stretch")


def _render_coverage_row(chunk: list[GridItem], mask_gt: np.ndarray | None, n_cols: int) -> None:
    """
    Coverage tiles placed directly under the images, column by column.
    """
    cols = _make_columns(n_cols)
    cols[0].markdown("**Pokrycie terenu**")
    first_pred = 1
    ref_pct = None

    if mask_gt is not None:
        ref_pct = class_percentages(mask_gt)
        cols[0].caption("Mała liczba w kafelku predykcji = różnica względem Ground Truth (w p.p.).")
        cols[1].markdown(_coverage_tiles_html("Ground Truth", mask_gt), unsafe_allow_html=True)
        first_pred = 2

    for col, (label, _, mask) in zip(cols[first_pred:], chunk):
        col.markdown(_coverage_tiles_html(label, mask, ref_pct), unsafe_allow_html=True)


def _render_grid(items: list[GridItem], mask_gt: np.ndarray | None = None) -> None:
    """
    Shared renderer for all views: a comparison grid.

    items: list of (label, input image, predicted mask). One item = a single result,
    several items = a comparison of models or resolutions.
    """
    h, w = items[0][2].shape
    ref_img = items[0][1][:h, :w]  # shared "Oryginał" column
    gt_fit = _fit_mask(mask_gt, (h, w)) if mask_gt is not None else None

    n_ref = 2 if mask_gt is not None else 1  # original + gt
    slots = min(len(items), MAX_PRED_COLS)
    # Always the same number of columns, so images keep the same width in every group
    n_cols = n_ref + slots

    for start in range(0, len(items), slots):
        chunk = items[start:start + slots]

        if start > 0:
            st.markdown("---")
        if len(items) > slots:
            st.caption(f"Predykcje {start + 1}–{start + len(chunk)} z {len(items)}")

        tab_mask, tab_overlay = st.tabs(["Maska", "Nakładka"])
        with tab_mask:
            _render_image_row(chunk, ref_img, gt_fit, n_cols, as_overlay=False)
        with tab_overlay:
            _render_image_row(chunk, ref_img, gt_fit, n_cols, as_overlay=True)

        _render_coverage_row(chunk, mask_gt, n_cols)

    # extra space between the coverage tiles
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


def _render_summary_table(group_col_name: str, items: list[tuple[str, np.ndarray]], mask_gt: np.ndarray) -> None:
    """
    Creates table with MIoU and mF1 metrics comparing predictions with GT
    """
    with st.expander(f"Metryki per {group_col_name.lower()}", expanded=True):
        rows = []
        for label, mask in items:
            h, w = mask.shape
            gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST) if mask_gt.shape != (h, w) else mask_gt
            m = _compute_metrics(mask, gt_r)
            rows.append({
                group_col_name: label,
                "mIoU": round(float(np.nanmean(m["iou"])), 4),
                "mF1": round(float(np.nanmean(m["f1"])), 4),
            })
        render_metrics_table(rows, highlight_best=True)


def render_result_block(img_input: np.ndarray, mask: np.ndarray, label: str = "") -> None:
    """
    Renders whole result block.
    """
    if label: st.caption(label)
    render_legend()
    _render_grid([("Predykcja", img_input, mask)], mask_gt=None)


def render_gt_comparison(img_input: np.ndarray, mask_pred: np.ndarray, mask_gt: np.ndarray, label: str = "") -> None:
    """
    Single predictions with coverage tiles and metrics comparing GT and predictions per class.
    """
    h, w = mask_pred.shape
    mask_gt = _fit_mask(mask_gt, (h, w))

    st.markdown("##### Porównanie z Ground Truth")
    if label: st.caption(label)
    render_legend()
    _render_grid([("Predykcja", img_input, mask_pred)], mask_gt=mask_gt)

    with st.expander("Metryki per klasa (IoU / F1)", expanded=True):
        metrics = _compute_metrics(mask_pred, mask_gt)
        rows = [{"Klasa": cls, "IoU": metrics["iou"][i], "F1": metrics["f1"][i]} for i, cls in enumerate(CLASSES)]
        rows.append({
            "Klasa": "— średnia —",
            "IoU": round(float(np.nanmean(metrics["iou"])), 4),
            "F1": round(float(np.nanmean(metrics["f1"])), 4),
        })
        render_metrics_table(rows, best_class_by="IoU")


def render_resolution_comparison(results: list[dict], mask_gt: np.ndarray | None = None) -> None:
    """
    Compares predictions with different resolutions. Mean IoU and F1 metrics are available.
    """
    if not results: return
    render_legend()
    # Each column is labelled with the resolution and the size of the rescaled image
    items = [
        (f"{r['label']} · {r['img'].shape[1]}×{r['img'].shape[0]} px", r["img"], r["mask"])
        for r in results
    ]
    _render_grid(items, mask_gt=mask_gt)

    if mask_gt is not None:
        _render_summary_table("Rozdzielczość", [(r["label"], r["mask"]) for r in results], mask_gt)


def render_model_comparison(img_input: np.ndarray, model_results: dict, mask_gt: np.ndarray | None = None) -> None:
    """
    Compares predictions with different models. Mean IoU and F1 metrics are available.
    """
    render_legend()
    # All models ran on the same input, so every column shares img_input
    items = [(name, img_input, mask) for name, mask in model_results.items()]
    _render_grid(items, mask_gt=mask_gt)

    if mask_gt is not None:
        _render_summary_table("Model", list(model_results.items()), mask_gt)