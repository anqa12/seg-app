import numpy as np
import os

# segmentation classes
NUM_CLASSES: int = 5
CLASSES: list[str] = ["Tło", "Zabudowa", "Lasy", "Woda", "Drogi"]
CLASS_COLORS: np.ndarray = np.array([
    [0,   0,   0],  # background - black
    [139,  90,  43],  # buildings — brown
    [34, 139,  34],  # forest - green
    [65, 105, 225],  # water - blue
    [192, 192, 192],  # roads - gray
], dtype=np.uint8)
CLASS_HEX: list[str] = ["#000000", "#8B5A2B", "#228B22", "#4169E1", "#C0C0C0"]

MODELS: dict[str, dict[str, str | int]] = {
    "UNet — LandCover": {"path": "models/landcover/unet_best.pth", "encoder": "unet_resnet34", "tile": 512},
    "DeepLabV3+ — LandCover": {"path": "models/landcover/deeplab_best.pth", "encoder": "deeplab_resnet34", "tile": 512},
    "SegFormer — LandCover":   {"path": "models/landcover/segformer_best.pth", "encoder": "segformer_mit_b2", "tile": 512},

    "UNet — BigEarthNet": {"path": "models/bigearthnet/unet_ben_best.pth", "encoder": "unet_resnet34", "tile": 128},
    "DeepLabV3+ — BigEarthNet": {"path": "models/bigearthnet/deeplab_ben_best.pth", "encoder": "deeplab_resnet34", "tile": 128},
    "SegFormer — BigEarthNet": {"path": "models/bigearthnet/segformer_ben_best.pth", "encoder": "segformer_mit_b2", "tile": 128},
}

LANDCOVER_MODELS: list[str] = [k for k in MODELS if "LandCover" in k]
BIGEARTHNET_MODELS: list[str] = [k for k in MODELS if "BigEarthNet" in k]

RESOLUTIONS: dict[str, float] = {
    "0.25 m/px": 0.25,
    "0.5 m/px": 0.50,
    "1 m/px": 1.00,
    "2.5 m/px": 2.50,
    "5 m/px": 5.00,
    "10 m/px": 10.00,
}

OVERLAY_ALPHA: float = 0.45
TILE_OVERLAP: float = 0.5
USE_GAUSSIAN_VOTING: bool = True

# one sample image: paths, display name and source resolution.
# "mask" is None when no ground-truth mask was found for the image.
Sample = dict[str, str | float | None]


def _load_samples(dataset: str, source_res: float = 0.25) -> dict[str, Sample]:
    """
    Automatically loads samples from assets/samples/<dataset>/images/ and adjust masks
    from assets/samples/<dataset>/masks/ by filename
    """
    img_dir = f"assets/samples/{dataset}/images"
    mask_dir = f"assets/samples/{dataset}/masks"
    samples: dict[str, Sample] = {}

    if not os.path.exists(img_dir):
        return samples

    suffix = "LC" if "landcover" in dataset.lower() else "BEN"

    for i, fname in enumerate(
            sorted(f for f in os.listdir(img_dir)
                   if not f.startswith(".")
                      and f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))),
            start=1
    ):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)

        mask_path = None
        for ext in (".png", ".tif", ".jpg"):
            candidate = os.path.join(mask_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        # display name
        label = f"test{i}_{suffix}"

        samples[label] = {
            "img": img_path,
            "mask": mask_path,
            "desc": label,
            "source_res": source_res,
        }

    return samples


SAMPLE_IMAGES: dict[str, dict[str, Sample]] = {
    "LandCover": _load_samples("landcover", source_res=0.25),
    "BigEarthNet": _load_samples("bigearthnet", source_res=10.0),
}