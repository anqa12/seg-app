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


# --- I/O & TRANSFORMS ---

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
    try:
        m = np.array(Image.open(path))
        return (m[:, :, 0] if m.ndim == 3 else m).astype(np.uint8)
    except Exception as e:
        st.warning(f"Nie można wczytać maski GT: {e}")
        return None


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    return ((1 - OVERLAY_ALPHA) * img[:h, :w] + OVERLAY_ALPHA * CLASS_COLORS[mask]).astype(np.uint8)


def class_percentages(mask: np.ndarray) -> dict:
    total = mask.size
    if total == 0:
        return {cls: 0.0 for cls in CLASSES}
    return {cls: round((mask == i).sum() / total * 100, 1) for i, cls in enumerate(CLASSES)}


# --- METRICS & BASE RENDERING ---

def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
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


def render_legend():
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;font-size:13px">'
        f'<span style="width:13px;height:13px;border-radius:3px;background:{CLASS_HEX[i]};border:1px solid rgba(0,0,0,.2)"></span>'
        f'{cls}</span>'
        for i, cls in enumerate(CLASSES)
    )
    st.markdown(f'<div style="margin:8px 0 14px 0">{items}</div>', unsafe_allow_html=True)


def render_coverage_tiles(mask: np.ndarray):
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


def render_metrics_table(metrics: list[dict]):
    df = pd.DataFrame(metrics)
    numeric_cols = [c for c in df.columns if c not in ("Klasa", "Rozdzielczość", "Model")]

    def color_val(val):
        if val == "—": return "color: gray"
        v = float(val)
        return "color: #228B22; font-weight:600" if v >= 0.7 else ("color: #B8860B; font-weight:600" if v >= 0.4 else "color: #C0392B")

    for c in numeric_cols:
        df[c] = df[c].apply(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}")

    st.dataframe(df.style.map(color_val, subset=numeric_cols), use_container_width=True, hide_index=True)


# --- WSPÓLNY RENDERER WIERSZA I PODSUMOWANIA (ELIMINACJA DUPLIKATÓW) ---

def _render_row(img: np.ndarray, mask_pred: np.ndarray, mask_gt: np.ndarray | None = None):
    """Pomocniczy renderer wiersza: 3 kolumny (bez GT) lub 4 kolumny (z GT)."""
    h, w = mask_pred.shape
    if mask_gt is not None:
        if mask_gt.shape != (h, w):
            mask_gt = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
        cols = st.columns(4)
        cols[0].image(img[:h, :w], caption="Wejście modelu", width="stretch")
        cols[1].image(mask_to_rgb(mask_gt), caption="Ground Truth", width="stretch")
        cols[2].image(mask_to_rgb(mask_pred), caption="Predykcja", width="stretch")
        cols[3].image(overlay(img, mask_pred), caption="Nakładka", width="stretch")

        col_gt, col_pred = st.columns(2)
        with col_gt:
            st.markdown("**Pokrycie terenu — Ground Truth**")
            render_coverage_tiles(mask_gt)
        with col_pred:
            st.markdown("**Pokrycie terenu — Predykcja**")
            render_coverage_tiles(mask_pred)
    else:
        cols = st.columns(3)
        cols[0].image(img[:h, :w], caption="Wejście modelu", width="stretch")
        cols[1].image(mask_to_rgb(mask_pred), caption="Predykcja", width="stretch")
        cols[2].image(overlay(img, mask_pred), caption="Nakładka", width="stretch")
        st.markdown("**Pokrycie terenu**")
        render_coverage_tiles(mask_pred)


def _render_summary_table(group_col_name: str, items: list[tuple[str, np.ndarray]], mask_gt: np.ndarray):
    """Tworzy tabelę mIoU / mF1 porównującą predykcje z GT."""
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
        render_metrics_table(rows)


# --- GŁÓWNE WIDOKI DLA APLIKACJI ---

def render_result_block(img_input: np.ndarray, mask: np.ndarray, label: str = ""):
    if label: st.caption(label)
    render_legend()
    _render_row(img_input, mask, mask_gt=None)


def render_gt_comparison(img_input: np.ndarray, mask_pred: np.ndarray, mask_gt: np.ndarray, label: str = ""):
    """Pięć kolumn ze szczegółowym rozbiciem na klasy."""
    h, w = mask_pred.shape
    if mask_gt.shape != (h, w):
        mask_gt = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)

    st.markdown("##### Porównanie z Ground Truth")
    cols = st.columns(5)
    cols[0].image(img_input[:h, :w], caption="Oryginał", width="stretch")
    cols[1].image(mask_to_rgb(mask_gt), caption="GT — maska", width="stretch")
    cols[2].image(overlay(img_input, mask_gt), caption="GT — nakładka", width="stretch")
    cols[3].image(mask_to_rgb(mask_pred), caption="Predykcja — maska", width="stretch")
    cols[4].image(overlay(img_input, mask_pred), caption="Predykcja — nakładka", width="stretch")

    if label: st.caption(label)
    render_legend()

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    c_gt, c_pred = st.columns(2, gap="large")
    with c_gt:
        st.markdown("**Pokrycie terenu — Ground Truth**")
        render_coverage_tiles(mask_gt)
    with c_pred:
        st.markdown("**Pokrycie terenu — Predykcja**")
        render_coverage_tiles(mask_pred)

    with st.expander("Metryki per klasa (IoU / F1)", expanded=True):
        metrics = _compute_metrics(mask_pred, mask_gt)
        rows = [{"Klasa": cls, "IoU": metrics["iou"][i], "F1": metrics["f1"][i]} for i, cls in enumerate(CLASSES)]
        rows.append({
            "Klasa": "— średnia —",
            "IoU": round(float(np.nanmean(metrics["iou"])), 4),
            "F1": round(float(np.nanmean(metrics["f1"])), 4),
        })
        render_metrics_table(rows)


def render_resolution_comparison(results: list[dict], mask_gt: np.ndarray | None = None):
    if not results: return
    render_legend()
    for r in results:
        st.markdown(f"#### {r['label']} · {r['img'].shape[1]}×{r['img'].shape[0]} px")
        _render_row(r["img"], r["mask"], mask_gt=mask_gt)
        st.markdown("---")

    if mask_gt is not None:
        _render_summary_table("Rozdzielczość", [(r["label"], r["mask"]) for r in results], mask_gt)


def render_model_comparison(img_input: np.ndarray, model_results: dict, mask_gt: np.ndarray | None = None):
    render_legend()
    for name, mask in model_results.items():
        st.markdown(f"#### {name}")
        _render_row(img_input, mask, mask_gt=mask_gt)
        st.markdown("---")

    if mask_gt is not None:
        _render_summary_table("Model", list(model_results.items()), mask_gt)