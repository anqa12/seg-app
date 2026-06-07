# 🛰️ Semantic Segmentation of Remote Sensing Images

An interactive web application for testing semantic segmentation models trained on high-resolution aerial and satellite imagery.

## Overview

This app allows you to:
- Perform semantic segmentation on aerial images (LandCover.ai, 25 cm/px) and satellite images (BigEarthNet, 10 m/px)
- Test the effect of spatial resolution on segmentation quality (from 25 cm/px down to 10 m/px)
- Compare multiple models on the same image side by side
- Compare model predictions against Ground Truth masks (for test images) with per-class IoU and F1 metrics
- Upload your own image (JPG, PNG, TIF) with automatic resolution detection from GeoTIFF metadata

## Segmentation Classes

| Class | Color |
|-------|-------|
| Background | ⬛ black |
| Buildings | 🟫 brown |
| Forest | 🟩 green |
| Water | 🟦 blue |
| Roads | ⬜ gray |

## Models

| Model | Encoder | Training Dataset | Tile Size |
|-------|---------|-----------------|-----------|
| UNet | ResNet34 | LandCover.ai | 512 px |
| DeepLabV3+ | ResNet50 | LandCover.ai | 512 px |
| SegFormer | MiT-B2 | LandCover.ai | 512 px |
| UNet | ResNet34 | BigEarthNet | 120 px |

## Project Structure

```
.
├── app.py                  # Main page (Streamlit entry point)
├── pages/
│   └── segmentacja.py      # Segmentation page
├── utils/
│   ├── config.py           # Configuration: classes, models, resolutions, sample images
│   ├── display.py          # Streamlit rendering functions
│   ├── inference.py        # Preprocessing, tiled prediction with Gaussian blending
│   └── models.py           # PyTorch model loading
├── models/
│   ├── unet_best.pth
│   ├── deeplab_best.pth
│   ├── segformer_best.pth
│   └── unet_bigearthnet_best.pth
├── assets/
│   └── samples/
│       ├── landcover/
│       │   ├── images/     # LandCover.ai test images (.jpg)
│       │   └── masks/      # LandCover.ai GT masks (.png, values 0–4)
│       └── bigearthnet/
│           ├── images/     # BigEarthNet test images (.jpg)
│           └── masks/      # BigEarthNet GT masks (.png, values 0–4)
└── requirements.txt
```

## Installation

### Requirements

- Python 3.10+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add model weights

Copy the `.pth` files into the `models/` folder:

```
models/unet_best.pth
models/deeplab_best.pth
models/segformer_best.pth
models/unet_bigearthnet_best.pth
```

> Model weights are not stored in the repository due to file size.
> Download them from the shared link provided separately.

### Step 4 — Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Technical Details

### Tiled Prediction with Gaussian Blending

For large images the model operates tile by tile with configurable overlap. Each tile is weighted by a 2D Gaussian window — pixels near tile edges receive lower weight than central pixels. Softmax is applied before accumulating votes so that averaging happens in probability space rather than over raw logits. This eliminates visible seam artefacts at tile boundaries.

### Resolution Scaling

The scale factor is computed dynamically from the native resolution of the input image read from GeoTIFF metadata. For images without metadata, 25 cm/px is assumed. The image is cropped to a multiple of 32 px, which is required by the U-Net architecture with ResNet34 encoder (5 pooling levels: 2⁵ = 32).

## Author
Anna Grygar <br>
Engineering thesis — Informatics, Faculty of Applied Mathematics, Silesian University of Technology