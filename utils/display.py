import numpy as np
import pandas as pd
import math
import streamlit as st
from utils.config import CLASSES, CLASS_COLORS, CLASS_HEX, NUM_CLASSES
import cv2
from PIL import Image


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Converts the mask into 5-color RGB image."""
    return CLASS_COLORS[mask]


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays the mask on the image for visualization.
    You can manually arrange alpha value to manipulate the visibility of the mask on the image."""
    h, w = mask.shape
    img_c = img[:h, :w].astype(np.float32)
    mask_c = CLASS_COLORS[mask].astype(np.float32)
    return ((1 - alpha) * img_c + alpha * mask_c).astype(np.uint8)


def class_percentages(mask: np.ndarray) -> dict:
    """Calculates the percentage of each class in the mask."""
    total = mask.size
    return {cls: round((mask == i).sum() / total * 100, 1)
            for i, cls in enumerate(CLASSES)}


def load_mask_from_path(path: str) -> np.ndarray | None:
    """
    Loads GT mask from .png file (pixel values = indices of classes 0-4).
    """
    try:
        m = np.array(Image.open(path))
        if m.ndim == 3:
            m = m[:, :, 0]  # only first channel (class index)
        return m.astype(np.uint8)
    except Exception as e:
        st.warning(f"Nie można wczytać maski GT: {e}")
        return None


def render_legend():
    """Renders the legend on the screen."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;'
        f'gap:5px;margin-right:14px;font-size:13px">'
        f'<span style="width:13px;height:13px;border-radius:3px;'
        f'background:{CLASS_HEX[i]};border:1px solid rgba(0,0,0,.2)"></span>'
        f'{cls}</span>'
        for i, cls in enumerate(CLASSES)
    )
    st.markdown(f'<div style="margin:8px 0 14px 0">{items}</div>',
                unsafe_allow_html=True)


def render_coverage_tiles(mask: np.ndarray):
    """
    Renders segmentation coverage tiles with % scores.
    Layout: 3 tiles in the first row, 2 tiles (centered) in the second row.
    """
    pct = class_percentages(mask)
    items = list(pct.items())

    def _tile(col, cls, val):
        i = CLASSES.index(cls)
        hex_c = CLASS_HEX[i]
        active = val > 0.05
        opacity = "1" if active else "0.3"

        # Tło (i==0) — czarny kolor niewidoczny na ciemnym tle,
        # zastępujemy białą/szarą obwódką dla kropki i ramki karty
        if i == 0:
            dot_border = "border:1px solid rgba(255,255,255,.5)"
            card_border = f"rgba(255,255,255,{'0.6' if active else '0.15'})"
        else:
            dot_border = "border:1px solid rgba(0,0,0,.15)"
            card_border = f"{hex_c}{'bb' if active else '33'}"

        col.markdown(
            f"""<div style="
                background:{hex_c}18;
                border:2px solid {card_border};
                border-radius:10px;
                padding:14px 6px 12px 6px;
                text-align:center;
                opacity:{opacity};
                margin:4px;">
              <div style="width:14px;height:14px;border-radius:4px;
                   background:{hex_c};margin:0 auto 8px auto;
                   {dot_border}"></div>
              <div style="font-size:10px;color:gray;margin-bottom:4px">{cls}</div>
              <div style="font-size:20px;font-weight:700">{val:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # wiersz 1: pierwsze 3 klasy
    for col, (cls, val) in zip(st.columns(3), items[:3]):
        _tile(col, cls, val)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # wiersz 2: ostatnie 2 klasy wyśrodkowane pustymi kolumnami
    _, c1, c2, _ = st.columns([0.5, 1, 1, 0.5])
    for col, (cls, val) in zip([c1, c2], items[3:]):
        _tile(col, cls, val)


def render_metrics_table(metrics: list[dict]):
    df = pd.DataFrame(metrics)

    numeric_cols = [
        c for c in df.columns
        if c not in ("Klasa", "Rozdzielczość", "Model")
    ]

    def color_val(val):
        if val == "—":
            return "color: white"

        v = float(val)

        if v >= 0.7:
            return "color: #228B22; font-weight:600"
        elif v >= 0.4:
            return "color: #B8860B; font-weight:600"
        else:
            return "color: #C0392B"

    # tutaj zamieniamy NaN na tekst
    for c in numeric_cols:
        df[c] = df[c].apply(
            lambda x: "—" if pd.isna(x) else f"{float(x):.2f}"
        )

    styled = (
        df.style
        .map(color_val, subset=numeric_cols)
    )

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True
    )


def render_gt_comparison(img_input: np.ndarray, mask_pred: np.ndarray,
                         mask_gt: np.ndarray, alpha: float,
                         label: str = ""):
    """
    Five columns: original / GT mask / GT overlay / pred mask / pred overlay
    + two coverage tables (GT vs pred)
    + metrics table (IoU/F1) per class, if GT available
    """

    h, w = mask_pred.shape
    if mask_gt.shape != mask_pred.shape:
        mask_gt = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)

    st.markdown("##### Porównanie z Ground Truth")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.image(img_input[:h, :w], caption="Oryginał", width="stretch")
    c2.image(mask_to_rgb(mask_gt), caption="GT — maska", width="stretch")
    c3.image(overlay(img_input, mask_gt, alpha), caption="GT — nakładka", width="stretch")
    c4.image(mask_to_rgb(mask_pred), caption="Predykcja — maska", width="stretch")
    c5.image(overlay(img_input, mask_pred, alpha), caption="Predykcja — nakładka", width="stretch")

    if label:
        st.caption(label)
    render_legend()

    # Two segmentation tables (one for GT, one for pred)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    col_gt, col_pred = st.columns(2, gap="large")
    with col_gt:
        st.markdown("**Pokrycie terenu — Ground Truth**")
        render_coverage_tiles(mask_gt)
    with col_pred:
        st.markdown("**Pokrycie terenu — Predykcja**")
        render_coverage_tiles(mask_pred)
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    st.markdown("")

    # metrics per class
    with st.expander("Metryki per klasa (IoU / F1)", expanded=True):
        metrics = _compute_metrics(mask_pred, mask_gt)
        rows = []
        for i, cls in enumerate(CLASSES):
            iou_val = metrics["iou"][i]
            f1_val = metrics["f1"][i]
            rows.append({
                "Klasa": cls,
                "IoU": iou_val,
                "F1": f1_val,
            })

        # mean calculated only with not NaN classes
        mean_iou = float(np.nanmean(metrics["iou"]))
        mean_f1 = float(np.nanmean(metrics["f1"]))
        rows.append({
            "Klasa": "— średnia —",
            "IoU": round(mean_iou, 4),
            "F1": round(mean_f1, 4),
        })
        render_metrics_table(rows)


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute per-class IoU and F1.
    Classes absent from both GT and prediction are marked as NaN
    and excluded from the mean (macro average over present classes only).
    """
    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    f1 = np.full(NUM_CLASSES, np.nan, dtype=np.float64)

    for c in range(NUM_CLASSES):
        p = (pred == c)
        g = (gt == c)

        # class that does not exist neither in GT nor in pred
        if not p.any() and not g.any():
            continue

        tp = (p & g).sum()
        fp = (p & ~g).sum()
        fn = (~p & g).sum()

        denom_iou = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn

        iou[c] = tp / denom_iou if denom_iou > 0 else 0.0
        f1[c] = 2 * tp / denom_f1 if denom_f1 > 0 else 0.0

    return {"iou": iou, "f1": f1}


def render_result_block(img_input, mask, alpha, label: str = ""):
    """Renders results without GT mask.
    Three columns: original / mask / overlay + legend + coverage tiles."""
    h, w = mask.shape
    c1, c2, c3 = st.columns(3)
    c1.image(img_input[:h, :w], caption="Oryginał", width="stretch")
    c2.image(mask_to_rgb(mask), caption="Maska", width="stretch")
    c3.image(overlay(img_input, mask, alpha), caption="Nakładka", width="stretch")
    if label:
        st.caption(label)
    render_legend()
    render_coverage_tiles(mask)


# different resolutions comparison
def render_resolution_comparison(results: list[dict], alpha: float,
                                 mask_gt: np.ndarray | None = None):
    """
    Renders one full block per resolution (input image / GT / prediction /
    overlay side by side, plus land-cover coverage tiles), stacked vertically.

    results: [{"label": str, "img": ndarray, "mask": ndarray}]
    mask_gt: optional ground-truth mask. If provided, each block also gets a
        GT column, and a metrics table (mIoU/mF1) is computed at the end by
        comparing every prediction against it. If NOT provided, no metrics
        table is shown at all.
    """
    n = len(results)
    if n == 0:
        return

    render_legend()

    for r in results:
        h, w = r["mask"].shape
        st.markdown(f"#### {r['label']} · {r['img'].shape[1]}×{r['img'].shape[0]} px")

        if mask_gt is not None:
            gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
            c1, c2, c3, c4 = st.columns(4)
            c1.image(r["img"], caption="Wejście modelu", width="stretch")
            c2.image(mask_to_rgb(gt_r), caption="Ground Truth", width="stretch")
            c3.image(mask_to_rgb(r["mask"]), caption="Predykcja", width="stretch")
            c4.image(overlay(r["img"], r["mask"], alpha), caption="Nakładka", width="stretch")
        else:
            c1, c2, c3 = st.columns(3)
            c1.image(r["img"], caption="Wejście modelu", width="stretch")
            c2.image(mask_to_rgb(r["mask"]), caption="Predykcja", width="stretch")
            c3.image(overlay(r["img"], r["mask"], alpha), caption="Nakładka", width="stretch")

        if mask_gt is not None:
            col_gt, col_pred = st.columns(2)
            with col_gt:
                st.markdown("**Pokrycie terenu — Ground Truth**")
                render_coverage_tiles(gt_r)
            with col_pred:
                st.markdown("**Pokrycie terenu — Predykcja**")
                render_coverage_tiles(r["mask"])
        else:
            st.markdown("**Pokrycie terenu**")
            render_coverage_tiles(r["mask"])

        st.markdown("---")

    # metrics table
    if mask_gt is not None:
        with st.expander("Metryki per rozdzielczość", expanded=True):
            rows = []
            for r in results:
                h, w = r["mask"].shape
                gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                m = _compute_metrics(r["mask"], gt_r)
                rows.append({
                    "Rozdzielczość": r["label"],
                    "mIoU": round(float(m["iou"].mean()), 4),
                    "mF1": round(float(m["f1"].mean()), 4),
                })
            render_metrics_table(rows)


# different models comparison
def render_model_comparison(img_input, model_results: dict, alpha: float,
                            mask_gt: np.ndarray | None = None):
    """
    Renders one full block per model (input image / GT / prediction / overlay
    side by side, plus land-cover coverage tiles), stacked vertically.

    model_results: {"model name": mask_array, ...}
    mask_gt: optional ground-truth mask. If provided, each block also gets a
        GT column, and a metrics table (mIoU/mF1) is computed at the end by
        comparing every model's prediction against it. If NOT provided, no
        metrics table is shown at all — there is no fallback path here
        (unlike render_resolution_comparison).
    """
    render_legend()

    for name, mask in model_results.items():
        h, w = mask.shape
        st.markdown(f"#### {name}")

        if mask_gt is not None:
            gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
            c1, c2, c3, c4 = st.columns(4)
            c1.image(img_input[:h, :w], caption="Wejście modelu", width="stretch")
            c2.image(mask_to_rgb(gt_r), caption="Ground Truth", width="stretch")
            c3.image(mask_to_rgb(mask), caption="Predykcja", width="stretch")
            c4.image(overlay(img_input, mask, alpha), caption="Nakładka", width="stretch")
        else:
            c1, c2, c3 = st.columns(3)
            c1.image(img_input[:h, :w], caption="Wejście modelu", width="stretch")
            c2.image(mask_to_rgb(mask), caption="Predykcja", width="stretch")
            c3.image(overlay(img_input, mask, alpha), caption="Nakładka", width="stretch")

        if mask_gt is not None:
            col_gt, col_pred = st.columns(2)
            with col_gt:
                st.markdown("**Pokrycie terenu — Ground Truth**")
                render_coverage_tiles(gt_r)
            with col_pred:
                st.markdown("**Pokrycie terenu — Predykcja**")
                render_coverage_tiles(mask)
        else:
            st.markdown("**Pokrycie terenu**")
            render_coverage_tiles(mask)

        st.markdown("---")

    if mask_gt is not None:
        with st.expander("Metryki per model", expanded=True):
            rows = []
            for name, mask in model_results.items():
                h, w = mask.shape
                gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                m = _compute_metrics(mask, gt_r)
                rows.append({
                    "Model": name,
                    "mIoU": round(float(m["iou"].mean()), 4),
                    "mF1": round(float(m["f1"].mean()), 4),
                })
            render_metrics_table(rows)