import pytest
import torch
import torch.nn as nn
import numpy as np
from utils.inference import tile_layout, _snap32, predict, check_rescale
from utils.config import NUM_CLASSES

# Udawany model do testów - przyjmuje obraz i zwraca losowe wartości klas
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dodajemy sztuczny parametr, żeby funkcja predict() mogła odczytać .device
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x ma wymiary: (Batch, Channels, Height, Width)
        B, _, H, W = x.shape
        # Zwracamy "predykcje" dla każdej z klas, zrzucone na to samo urządzenie co model
        return torch.randn(B, NUM_CLASSES, H, W).to(self.dummy_param.device)


def test_snap32():
    """Test zaokrąglania wymiarów do wielokrotności 32 (wymóg sieci UNet)."""
    assert _snap32(10) == 32  # Minimum to zawsze 32
    assert _snap32(31) == 32
    assert _snap32(33) == 32
    assert _snap32(48) == 64  # 48/32 = 1.5 -> int(1.5 + 0.5) * 32 = 64
    assert _snap32(60) == 64


def test_tile_layout_no_overlap():
    """Test cięcia na kafelki bez nachodzenia na siebie (overlap 0%)."""
    # Obraz 128x128, kafelek 64x64, overlap 0.0
    tile_size, ys, xs = tile_layout(H=128, W=128, tile_size=64, overlap=0.0)

    assert tile_size == 64
    # Kafelki powinny zaczynać się w pikselach 0 i 64 (idealnie pokryją 128px)
    assert ys == [0, 64]
    assert xs == [0, 64]


def test_tile_layout_with_overlap():
    """Test cięcia na kafelki z nachodzeniem i dorabianiem marginesu na końcu."""
    # Obraz 100x100, kafelek 64, overlap 50%
    # Krok (stride) wynosi 32.
    # Pierwszy: 0..64. Drugi: 32..96. Brakuje 4px do 100, więc skrypt dorzuca kafelek od 36..100
    tile_size, ys, xs = tile_layout(H=100, W=100, tile_size=64, overlap=0.5)

    assert tile_size == 64
    assert ys == [0, 32, 36]
    assert xs == [0, 32, 36]


def test_tile_layout_image_smaller_than_tile():
    """Test zachowania, gdy podane zdjęcie jest mniejsze od żądanego kafelka."""
    # Obraz 50x50, chcemy ciąć na 128x128
    tile_size, ys, xs = tile_layout(H=50, W=50, tile_size=128, overlap=0.0)

    # Skrypt sam pomniejszy kafel do wielokrotności 32 mieszczącej się w 50px (czyli 32px)
    assert tile_size == 32
    # Skoro kafelek ma 32px, to krok to 32. Pierwszy to 0..32, drugi to 18..50
    assert ys == [0, 18]
    assert xs == [0, 18]


def test_tile_layout_too_small_image():
    """Test odrzucania obrazów mniejszych niż 32x32."""
    tile_size, ys, xs = tile_layout(H=20, W=20, tile_size=64, overlap=0.0)

    assert tile_size == 0
    assert ys == []
    assert xs == []


def test_check_rescale_blocks_upsampling():
    """Test czy program blokuje sztuczne powiększanie rozdzielczości (upsampling)."""
    # Jeśli mamy zdjęcie 1 m/px i chcemy 0.25 m/px, skala wyniesie 4.0.
    # Program powinien to zablokować, bo nie dodamy magicznie detali do pikseli.
    is_valid, reason = check_rescale((500, 500), 4.0)

    assert is_valid is False
    assert "powiększanie" in reason


def test_predict_without_voting():
    """Test składania kafelków w trybie twardym (argmax bez Gaussa)."""
    model = DummyModel()
    # Pusty obraz 100x100 (3 kanały RGB)
    img_np = np.zeros((100, 100, 3), dtype=np.uint8)

    # Wywołujemy predykcję tnąc na kafelki 64x64
    result = predict(model, img_np, tile_size=64, overlap=0.0, use_voting=False)

    # Sprawdzamy, czy aplikacja poprawnie skleiła wynik z powrotem do 100x100
    assert result is not None
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
    # Wartości w masce nie mogą przekroczyć liczby klas (0 do NUM_CLASSES-1)
    assert result.max() < NUM_CLASSES


def test_predict_with_gaussian_voting():
    """Test zaawansowanego uśredniania z nakładaniem się kafelków (overlap)."""
    model = DummyModel()
    img_np = np.zeros((150, 150, 3), dtype=np.uint8)

    # Tym razem z oknem Gaussa i overlapem
    result = predict(model, img_np, tile_size=64, overlap=0.5, use_voting=True)

    assert result is not None
    assert result.shape == (150, 150)
    assert result.dtype == np.uint8