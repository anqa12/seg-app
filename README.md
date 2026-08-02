# 🛰️ Semantic Segmentation of Remote Sensing Images

An interactive web application for testing semantic segmentation models trained on high-resolution aerial and satellite imagery.

## Overview

This app allows you to:
- Perform semantic segmentation on aerial images (LandCover.ai, 25 cm/px) and satellite images (BigEarthNet, 10 m/px)
- Test the effect of spatial resolution on segmentation quality (from 25 cm/px down to 10 m/px)
- Compare multiple models on the same image side by side
- Compare model predictions against Ground Truth masks (for test images) with per-class IoU and F1 metrics
- Upload your own image (JPG, PNG, TIF) with automatic resolution detection from GeoTIFF metadata
- Select an area directly on an interactive map of Poland and fetch a real-time ortophoto from the national Geoportal (GUGiK WMS)

## Segmentation Classes

| Class | Color |
|-------|-------|
| Background | ⬛ black |
| Buildings | 🟫 brown |
| Forest | 🟩 green |
| Water | 🟦 blue |
| Roads | ⬜ gray |

## Models

Six models are available — three architectures trained on each dataset:

### LandCover.ai (aerial, 25 cm/px)

| Model | Encoder | Tile Size | Weights file |
|-------|---------|-----------|--------------|
| UNet | ResNet34 | 512 px | `landcover/unet_best.pth` |
| DeepLabV3+ | ResNet34 | 512 px | `landcover/deeplab_best.pth` |
| SegFormer | MiT-B2 | 512 px | `landcover/segformer_best.pth` |

### BigEarthNet (Sentinel-2, 10 m/px)

| Model | Encoder | Tile Size | Weights file |
|-------|---------|-----------|--------------|
| UNet | ResNet34 | 120 px | `bigearthnet/unet_best.pth` |
| DeepLabV3+ | ResNet34 | 120 px | `bigearthnet/deeplab_best.pth` |
| SegFormer | MiT-B2 | 120 px | `bigearthnet/segformer_best.pth` |

All CNN models (UNet, DeepLabV3+) use a ResNet34 encoder pretrained on ImageNet.
SegFormer uses its native Mix Transformer encoder (MiT-B2).

## Application Structure

The app consists of a home page (`app.py`) and a single segmentation page
(`pages/segmentation.py`) with three tabs:

| Tab | Description |
|-----|-------------|
| **Sample images** | Choose from pre-loaded test tiles from LandCover.ai or BigEarthNet; Ground Truth masks are available for per-class IoU and F1 evaluation |
| **Upload** | Upload your own image (JPG, PNG, TIF); resolution is detected automatically from GeoTIFF metadata or set manually |
| **Map** | Draw a rectangle on an interactive map of Poland; fetch an ortophoto from GUGiK Geoportal WMS; run segmentation on the downloaded image |

On all three tabs you can select one or more models and one or more target
resolutions — the app automatically switches between:
- **resolution comparison mode** (one model, multiple resolutions)
- **model comparison mode** (multiple models, one resolution)
- **single result mode** (one model, one resolution)

## Project Structure

```
seg-app/
├── app.py                      # Home page (Streamlit entry point)
├── pages/
│   └── segmentation.py         # Main segmentation page (hosts all three tabs)
├── components/
│   ├── sample_tab.py           # Tab: choose from test sample images
│   ├── upload_tab.py           # Tab: upload your own image
│   └── map_tab.py              # Tab: interactive map + Geoportal WMS
├── utils/
│   ├── config.py               # Classes, model paths, resolutions, sample image index
│   ├── display.py              # Streamlit rendering: masks, overlays, coverage tiles
│   ├── inference.py            # Tiled prediction with Gaussian blending
│   ├── models.py               # PyTorch model loading with st.cache_resource
│   └── session.py              # Shared session state helpers
├── models/
│   ├── landcover/
│   │   ├── unet_best.pth
│   │   ├── deeplab_best.pth
│   │   └── segformer_best.pth
│   └── bigearthnet/
│       ├── unet_best.pth
│       ├── deeplab_best.pth
│       └── segformer_best.pth
├── assets/
│   └── samples/
│       ├── landcover/
│       │   ├── images/         # LandCover.ai test tiles (.jpg)
│       │   └── masks/          # Ground Truth masks (.png, pixel values 0–4)
│       └── bigearthnet/
│           ├── images/         # BigEarthNet test tiles (.png)
│           └── masks/          # Ground Truth masks (.png, pixel values 0–4)
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python 3.10+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd seg-app
```

### Step 2 — Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> The map tab requires `streamlit-folium`. It is included in `requirements.txt`.

### Step 3 — Add model weights

Copy the `.pth` files into the `models/` folder preserving the subdirectory
structure:

```
models/landcover/unet_best.pth
models/landcover/deeplab_best.pth
models/landcover/segformer_best.pth
models/bigearthnet/unet_best.pth
models/bigearthnet/deeplab_best.pth
models/bigearthnet/segformer_best.pth
```

> Model weights are not stored in the repository due to file size.  
> Download them from the shared link provided separately.

### Step 4 — Add sample images

```
assets/samples/landcover/images/    ← JPG tiles from LandCover.ai test split
assets/samples/landcover/masks/     ← PNG masks (pixel values 0–4)
assets/samples/bigearthnet/images/  ← PNG tiles exported from BigEarthNet
assets/samples/bigearthnet/masks/   ← PNG masks (pixel values 0–4)
```

> BigEarthNet images must be exported as **PNG** (not JPG) to preserve the
> exact pixel values after Sentinel-2 reflectance normalisation
> (`clip(reflectance / 3000, 0, 1) × 255`).
> JPEG compression introduces artefacts that degrade model predictions.

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Technical Details

### Tiled Prediction with Gaussian Blending

For large images the model operates tile by tile with configurable overlap.
Each tile is weighted by a 2D Gaussian window — pixels near tile edges receive
lower weight than central pixels. Softmax probabilities (not raw logits) are
accumulated across overlapping tiles and the final class is determined by
`argmax` over the averaged probability maps. This eliminates visible seam
artefacts at tile boundaries.

Padding at image borders uses **reflect padding** (`REFLECT_101`), consistent
with the `PadIfNeeded` augmentation applied during training, ensuring that the
model receives pixel value distributions at inference time that match those
seen during training.

### Resolution Scaling

The scale factor is computed from the native resolution of the source image.
For GeoTIFF files, resolution is read from metadata (only when a valid CRS
and non-identity geotransform are present). For other formats or files without
georeferencing, the user sets the source resolution manually (default: 25 cm/px).

The image is cropped to a multiple of 32 px after scaling, required by UNet
and DeepLabV3+ with ResNet34 encoder (5 max-pooling levels: 2⁵ = 32).

### Geoportal WMS Integration

The map tab fetches ortophotos from the Polish national Geoportal via OGC WMS:

```
https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardOGC
  ?SERVICE=WMS&REQUEST=GetMap&LAYERS=Raster
  &BBOX={lon_min},{lat_min},{lon_max},{lat_max}
  &WIDTH=2048&HEIGHT=2048&FORMAT=image/jpeg&SRS=EPSG:4326
```

Returned images have a resolution of approximately 25 cm/px (urban areas) or
50 cm/px (rural areas), matching the LandCover.ai training distribution. 
LandCover.ai models can therefore be applied directly without rescaling. But we can also
manipulate the resolution of the downloaded images for the purposes of experiments.

### Normalization

| Dataset | Normalization at inference                                                    |
|---------|-------------------------------------------------------------------------------|
| LandCover.ai (aerial, RGB uint8) | `pixel / 255.0` → [0, 1]                                                      |
| BigEarthNet (exported as PNG uint8) | `pixel / 255.0` → [0, 1] — values were already mapped to 0–255 at export time |

### Model Caching

All models are loaded once per session via `@st.cache_resource` — switching
between tabs or models does not trigger a reload.

## Author

Anna Grygar  
Engineering thesis — Informatics, Faculty of Applied Mathematics,  
Silesian University of Technology