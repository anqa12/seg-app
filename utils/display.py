import numpy as np
import pandas as pd
import streamlit as st
from utils.config import CLASSES, CLASS_COLORS, CLASS_HEX, NUM_CLASSES
import cv2


# masks preparation
def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = mask.shape
    img_c = img[:h, :w].astype(np.float32)
    mask_c = CLASS_COLORS[mask].astype(np.float32)
    return ((1 - alpha) * img_c + alpha * mask_c).astype(np.uint8)


def class_percentages(mask: np.ndarray) -> dict:
    total = mask.size
    return {cls: round((mask == i).sum() / total * 100, 1)
            for i, cls in enumerate(CLASSES)}


def load_mask_from_path(path: str) -> np.ndarray | None:
    """
    Wczytuje maskę GT z pliku PNG (wartości pikseli = indeksy klas 0–4).
    Obsługuje zarówno maski w skali szarości jak i RGB z CLASS_COLORS.
    """
    from PIL import Image
    try:
        m = np.array(Image.open(path))
        if m.ndim == 3:
            m = m[:, :, 0]  # only first channel (class index)
        return m.astype(np.uint8)
    except Exception as e:
        st.warning(f"Nie można wczytać maski GT: {e}")
        return None


# legend
def render_legend():
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


# segmentation tiles
def render_coverage_tiles(mask: np.ndarray):
    """Kolorowe kafelki z % pokrycia — wyszarzone jeśli 0%."""
    pct = class_percentages(mask)
    cols = st.columns(NUM_CLASSES)
    for col, (cls, val) in zip(cols, pct.items()):
        i = CLASSES.index(cls)
        hex_c = CLASS_HEX[i]
        active = val > 0.05
        opacity = "1" if active else "0.3"
        col.markdown(
            f"""<div style="
                background:{hex_c}18;
                border:2px solid {hex_c}{'bb' if active else '33'};
                border-radius:10px;
                padding:10px 6px 8px 6px;
                text-align:center;
                opacity:{opacity}">
              <div style="width:14px;height:14px;border-radius:4px;
                   background:{hex_c};margin:0 auto 6px auto;
                   border:1px solid rgba(0,0,0,.15)"></div>
              <div style="font-size:10px;color:gray;margin-bottom:2px">{cls}</div>
              <div style="font-size:20px;font-weight:700">{val:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )


# metrics table
def render_metrics_table(metrics: dict):
    df = pd.DataFrame(metrics)

    def color_val(val):
        try:
            v = float(val)
            if v >= 0.7: return "color: #228B22; font-weight:600"
            elif v >= 0.4: return "color: #B8860B; font-weight:600"
            else: return "color: #C0392B"
        except Exception:
            return ""

    numeric_cols = [c for c in df.columns if c != "Klasa" and c != "Rozdzielczość"
                    and c != "Model"]
    styled = (df.style
                .map(color_val, subset=numeric_cols)
                .format({c: "{:.4f}" for c in numeric_cols}))
    st.dataframe(styled, use_container_width=True, hide_index=True)


# GT comparison
def render_gt_comparison(img_input: np.ndarray, mask_pred: np.ndarray,
                         mask_gt: np.ndarray, alpha: float,
                         label: str = ""):
    """
    Pięć kolumn: oryginał / GT maska / GT nakładka / predykcja maska / predykcja nakładka
    + dwie tabelki pokrycia (GT vs predykcja) obok siebie
    + tabela metryk IoU/F1 per klasa jeśli GT dostępne.
    """
    # Wyrównaj rozmiary — GT może mieć inny rozmiar niż predykcja
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
    col_gt, col_pred = st.columns(2)
    with col_gt:
        st.markdown("**Pokrycie terenu — Ground Truth**")
        render_coverage_tiles(mask_gt)
    with col_pred:
        st.markdown("**Pokrycie terenu — Predykcja**")
        render_coverage_tiles(mask_pred)

    st.markdown("")

    # metrics per class
    with st.expander("Metryki per klasa (IoU / F1)", expanded=True):
        metrics = _compute_metrics(mask_pred, mask_gt)
        rows = []
        for i, cls in enumerate(CLASSES):
            rows.append({
                "Klasa": cls,
                "IoU": round(float(metrics["iou"][i]), 4),
                "F1": round(float(metrics["f1"][i]), 4),
            })
        rows.append({
            "Klasa": "— średnia —",
            "IoU": round(float(metrics["iou"].mean()), 4),
            "F1": round(float(metrics["f1"].mean()), 4),
        })
        rows = [
            {"Klasa": r["Klasa"], "IoU": r["IoU"], "F1": r["F1"]}
            for r in rows
        ]

        render_metrics_table(rows)


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:

    iou = np.zeros(NUM_CLASSES, dtype=np.float32)
    f1 = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        p = (pred == c)
        g = (gt == c)
        tp = (p & g).sum()
        fp = (p & ~g).sum()
        fn = (~p & g).sum()
        denom_iou = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        iou[c] = tp / denom_iou if denom_iou > 0 else 0.0
        f1[c] = 2 * tp / denom_f1 if denom_f1 > 0 else 0.0
    return {"iou": iou, "f1": f1}


# results - without GT
def render_result_block(img_input, mask, alpha, label: str = ""):
    """Trzy kolumny: oryginał / maska / nakładka + legenda + kafelki pokrycia."""
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
    results: [{"label": str, "img": ndarray, "mask": ndarray, "metrics": dict|None}]
    mask_gt: opcjonalna maska GT — jeśli podana, dodaje wiersz GT i metryki.
    """
    n = len(results)
    if n == 0:
        return

    # row - original img
    st.markdown("##### Wejście modelu")
    cols = st.columns(n)
    for col, r in zip(cols, results):
        col.image(r["img"], width="stretch")
        col.caption(f"{r['label']} · {r['img'].shape[1]}×{r['img'].shape[0]} px")

    # row - GT (if available)
    if mask_gt is not None:
        st.markdown("##### Ground Truth")
        cols = st.columns(n)
        for col, r in zip(cols, results):
            h, w = r["mask"].shape
            gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
            col.image(mask_to_rgb(gt_r), width="stretch")
            col.caption(r["label"])

    # row - pred mask
    st.markdown("##### Maska predykcji")
    cols = st.columns(n)
    for col, r in zip(cols, results):
        col.image(mask_to_rgb(r["mask"]), width="stretch")
        col.caption(r["label"])

    # row - overlay
    st.markdown("##### Nakładka")
    cols = st.columns(n)
    for col, r in zip(cols, results):
        col.image(overlay(r["img"], r["mask"], alpha), width="stretch")
        col.caption(r["label"])

    render_legend()

    # covering per class
    for r in results:
        if mask_gt is not None:
            col_gt, col_pred = st.columns(2)
            h, w = r["mask"].shape
            gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
            with col_gt:
                st.markdown(f"**GT — {r['label']}**")
                render_coverage_tiles(gt_r)
            with col_pred:
                st.markdown(f"**Predykcja — {r['label']}**")
                render_coverage_tiles(r["mask"])
        else:
            st.markdown(f"**Pokrycie terenu — {r['label']}**")
            render_coverage_tiles(r["mask"])
        st.markdown("")

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
    elif results[0].get("metrics"):
        with st.expander("Metryki per rozdzielczość"):
            rows = []
            for r in results:
                m = r["metrics"]
                rows.append({
                    "Rozdzielczość": r["label"],
                    "mIoU": round(m["mIoU"], 4),
                    "mF1": round(m["mF1"], 4),
                    "Acc": round(m["pixel_accuracy"], 4),
                })
            render_metrics_table(rows)


# different models comparison
def render_model_comparison(img_input, model_results: dict, alpha: float,
                            mask_gt: np.ndarray | None = None):
    """
    model_results: {"Nazwa modelu": mask_array, ...}
    mask_gt: opcjonalna maska GT.
    """
    n = len(model_results)
    h, w = list(model_results.values())[0].shape

    st.markdown("##### Wejście modelu")
    cols = st.columns(n)
    for col in cols:
        col.image(img_input[:h, :w], width="stretch")

    if mask_gt is not None:
        st.markdown("##### Ground Truth")
        gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
        cols = st.columns(n)
        for col in cols:
            col.image(mask_to_rgb(gt_r), width="stretch")

    st.markdown("##### Maska predykcji")
    cols = st.columns(n)
    for col, (name, mask) in zip(cols, model_results.items()):
        col.image(mask_to_rgb(mask), width="stretch")
        col.caption(name)

    st.markdown("##### Nakładka")
    cols = st.columns(n)
    for col, (name, mask) in zip(cols, model_results.items()):
        col.image(overlay(img_input, mask, alpha), width="stretch")
        col.caption(name)

    render_legend()

    for name, mask in model_results.items():
        if mask_gt is not None:
            gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
            col_gt, col_pred = st.columns(2)
            with col_gt:
                st.markdown(f"**GT — {name}**")
                render_coverage_tiles(gt_r)
            with col_pred:
                st.markdown(f"**Predykcja — {name}**")
                render_coverage_tiles(mask)
        else:
            st.markdown(f"**Pokrycie terenu — {name}**")
            render_coverage_tiles(mask)
        st.markdown("")

    if mask_gt is not None:
        with st.expander("Metryki per model", expanded=True):
            rows = []
            for name, mask in model_results.items():
                gt_r = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                m = _compute_metrics(mask, gt_r)
                rows.append({
                    "Model": name,
                    "mIoU": round(float(m["iou"].mean()), 4),
                    "mF1": round(float(m["f1"].mean()), 4),
                })
            render_metrics_table(rows)
