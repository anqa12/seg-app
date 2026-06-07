# 🛰️ Segmentacja semantyczna obrazów teledetekcyjnych

Interaktywna aplikacja webowa do testowania modeli segmentacji semantycznej wytrenowanych na wysokorozdzielczych zdjęciach lotniczych i satelitarnych.

## Opis

Aplikacja pozwala na:
- segmentację semantyczną zdjęć lotniczych (LandCover.ai, 25 cm/px) i satelitarnych (BigEarthNet, 10 m/px)
- testowanie wpływu rozdzielczości na jakość segmentacji (od 25 cm/px do 10 m/px)
- porównanie wyników wielu modeli na tym samym zdjęciu
- porównanie predykcji z maską Ground Truth (dla zdjęć testowych) wraz z metrykami IoU i F1 per klasa
- wgranie własnego zdjęcia (JPG, PNG, TIF) z automatycznym wykryciem rozdzielczości z metadanych GeoTIFF

## Klasy segmentacji

| Klasa | Kolor |
|-------|-------|
| Tło | ⬛ czarny |
| Zabudowa | 🟫 brązowy |
| Lasy | 🟩 zielony |
| Woda | 🟦 niebieski |
| Drogi | ⬜ szary |

## Modele

| Model | Enkoder | Datasset treningowy | Tile |
|-------|---------|---------------------|------|
| UNet | ResNet34 | LandCover.ai | 512 px |
| DeepLabV3+ | ResNet50 | LandCover.ai | 512 px |
| SegFormer | MiT-B2 | LandCover.ai | 512 px |
| UNet | ResNet34 | BigEarthNet | 120 px |

## Struktura projektu

```
.
├── app.py                  # Strona główna (Streamlit entry point)
├── pages/
│   └── segmentacja.py      # Strona segmentacji
├── utils/
│   ├── config.py           # Konfiguracja: klasy, modele, rozdzielczości, zdjęcia testowe
│   ├── display.py          # Funkcje renderujące wyniki w Streamlit
│   ├── inference.py        # Preprocessing, predykcja kafelkowa z Gaussian blending
│   └── models.py           # Ładowanie modeli PyTorch
├── models/
│   ├── unet_best.pth
│   ├── deeplab_best.pth
│   ├── segformer_best.pth
│   └── unet_bigearthnet_best.pth
├── assets/
│   └── samples/
│       ├── landcover/
│       │   ├── images/     # Zdjęcia testowe LandCover.ai (.jpg)
│       │   └── masks/      # Maski GT LandCover.ai (.png, wartości 0–4)
│       └── bigearthnet/
│           ├── images/     # Zdjęcia testowe BigEarthNet (.jpg)
│           └── masks/      # Maski GT BigEarthNet (.png, wartości 0–4)
└── requirements.txt
```

## Instalacja i uruchomienie

### Wymagania

- Python 3.10+
- pip

### Krok 1 — sklonuj repozytorium

```bash
git clone https://github.com/<twoj-login>/<nazwa-repo>.git
cd <nazwa-repo>
```

### Krok 2 — zainstaluj zależności

```bash
pip install -r requirements.txt
```

### Krok 3 — dodaj wagi modeli

Skopiuj pliki `.pth` do folderu `models/`:

```
models/unet_best.pth
models/deeplab_best.pth
models/segformer_best.pth
models/unet_bigearthnet_best.pth
```

> Wagi modeli nie są przechowywane w repozytorium ze względu na rozmiar.
> Skontaktuj się z autorką lub pobierz z udostępnionego linku.

### Krok 4 — uruchom aplikację

```bash
streamlit run app.py
```

Aplikacja otworzy się w przeglądarce pod adresem `http://localhost:8501`.

## Szczegóły techniczne

### Predykcja kafelkowa (Gaussian blending)

Dla dużych zdjęć model działa kafelkowo z nakładaniem (overlap). Każdy kafelek jest ważony dwuwymiarowym oknem Gaussa — piksele przy krawędziach kafelka mają mniejszą wagę niż piksele centralne. Przed sumowaniem głosów stosowany jest softmax, dzięki czemu uśrednianie odbywa się w przestrzeni prawdopodobieństw a nie surowych logitów. Eliminuje to artefakty na granicach kafelków.

### Skalowanie rozdzielczości

Scale obliczany jest dynamicznie na podstawie natywnej rozdzielczości wejściowego zdjęcia odczytanej z metadanych GeoTIFF. Dla zdjęć bez metadanych zakładana jest rozdzielczość 25 cm/px. Obraz jest przycinany do wielokrotności 32 px (wymóg architektury U-Net z ResNet34: 2⁵ = 32).

## Autorka
Anna Grygar
Praca inżynierska — Informatyka, Wydział Matematyki Stosowanej, Politechnika Śląska