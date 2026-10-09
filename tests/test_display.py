import pytest
import numpy as np
from utils.display import _compute_metrics, class_percentages
# Importujemy config, bo funkcja opiera się na definicji klas (NUM_CLASSES = 5)
from utils.config import NUM_CLASSES


def test_compute_metrics_perfect_match():
    """Test idealnego dopasowania maski predykcji do Ground Truth."""
    # Tworzymy obraz 4x4, gdzie wszystkie piksele to klasa 1 (np. Zabudowa)
    pred = np.ones((4, 4), dtype=np.uint8)
    gt = np.ones((4, 4), dtype=np.uint8)

    metrics = _compute_metrics(pred, gt)

    # Dla klasy 1 (Zabudowa) IoU i F1 musi być 1.0 (100%)
    assert metrics["iou"][1] == 1.0
    assert metrics["f1"][1] == 1.0

    # Dla pozostałych klas (0, 2, 3, 4), które nie występują na zdjęciu,
    # skrypt powinien zwrócić NaN, żeby nie zaniżać średniej wynikiem 0.0
    assert np.isnan(metrics["iou"][0])
    assert np.isnan(metrics["f1"][2])


def test_compute_metrics_partial_match():
    """Test matematycznego wyliczania IoU i F1 przy częściowym pokryciu."""
    # Ground Truth: górny rząd to 1, dolny to 0
    gt = np.array([[1, 1],
                   [0, 0]], dtype=np.uint8)

    # Predykcja: zgadła tylko jeden lewy górny piksel
    pred = np.array([[1, 0],
                     [0, 0]], dtype=np.uint8)

    metrics = _compute_metrics(pred, gt)

    # Analiza klasy 1:
    # True Positives = 1, False Positives = 0, False Negatives = 1
    # IoU = TP / (TP + FP + FN) = 1 / (1 + 0 + 1) = 0.5
    assert np.isclose(metrics["iou"][1], 0.5)

    # Analiza klasy 0:
    # TP = 2, FP = 1 (tam gdzie predykcja dała 0, a było 1 w GT), FN = 0
    # IoU = 2 / (2 + 1 + 0) = 2/3
    assert np.isclose(metrics["iou"][0], 2 / 3)


def test_compute_metrics_mismatch():
    """Test całkowitego pudła predykcji."""
    gt = np.ones((2, 2), dtype=np.uint8) * 2  # Wszystko to klasa 2
    pred = np.ones((2, 2), dtype=np.uint8) * 3  # Model wymyślił klasę 3

    metrics = _compute_metrics(pred, gt)

    # Klasa 2 występuje w GT, ale predykcja jej nie wykryła -> 0.0
    assert metrics["iou"][2] == 0.0
    # Klasa 3 została sztucznie wygenerowana (tzw. False Positives) -> 0.0
    assert metrics["iou"][3] == 0.0


def test_class_percentages():
    """Test poprawnego zliczania pokrycia terenu (procentów w kafelkach)."""
    # Maska 10 pikseli: pięć 0, trzy 1, dwa 2
    mask = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.uint8)

    pct = class_percentages(mask)

    # Sprawdzamy klucze słownika zgodnie z configiem
    assert pct["Tło"] == 50.0
    assert pct["Zabudowa"] == 30.0
    assert pct["Lasy"] == 20.0
    assert pct["Woda"] == 0.0