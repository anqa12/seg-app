import numpy as np
import os

# segmentation classes
NUM_CLASSES = 5
CLASSES = ["Tło", "Zabudowa", "Lasy", "Woda", "Drogi"]
CLASS_COLORS = np.array([
    [0,   0,   0],  # background - black
    [139,  90,  43],  # buildings — brown
    [34, 139,  34],  # forest - green
    [65, 105, 225],  # water - blue
    [192, 192, 192],  # roads - gray
], dtype=np.uint8)
CLASS_HEX = ["#000000", "#8B5A2B", "#228B22", "#4169E1", "#C0C0C0"]

MODELS = {
    "UNet — LandCover": {"path": "models/unet_best.pth", "encoder": "unet_resnet34", "tile": 512},
    "DeepLabV3+ — LandCover": {"path": "models/deeplab_best.pth", "encoder": "deeplab_resnet50", "tile": 512},
    "SegFormer — LandCover": {"path": "models/segformer_best.pth", "encoder": "segformer", "tile": 512},
    "UNet — BigEarthNet": {"path": "models/unet_bigearthnet_best.pth", "encoder": "unet_resnet34", "tile": 120},
}

RESOLUTIONS = {
    "25 cm/px": 0.25,
    "50 cm/px": 0.50,
    "1 m/px": 1.00,
    "2.5 m/px": 2.50,
    "5 m/px": 5.00,
    "10 m/px": 10.00,
}


def _load_samples(dataset: str, res_m: float = 0.25) -> dict:
    """
    Automatycznie wczytuje wszystkie zdjęcia z assets/samples/<dataset>/images/
    i dopasowuje maski z assets/samples/<dataset>/masks/ po nazwie pliku.
    """
    img_dir = f"assets/samples/{dataset}/images"
    mask_dir = f"assets/samples/{dataset}/masks"
    samples = {}

    if not os.path.exists(img_dir):
        return samples

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            continue
        stem = os.path.splitext(fname)[0]  # only filename
        img_path = os.path.join(img_dir, fname)

        # mask with the same filename
        mask_path = None
        for ext in (".png", ".tif", ".jpg"):
            candidate = os.path.join(mask_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        samples[stem] = {
            "img":  img_path,
            "mask": mask_path,
            "desc": f"{stem} — {dataset}",
        }

    return samples


SAMPLE_IMAGES = {
    "LandCover":   _load_samples("landcover"),
    "BigEarthNet": _load_samples("bigearthnet"),
}