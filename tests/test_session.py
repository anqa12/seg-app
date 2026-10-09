import pytest
import streamlit as st
from utils.session import set_native_resolution, reset_image_state


def test_set_native_resolution(monkeypatch):
    """Test dodawania wykrytej rozdzielczości do stanu (bez duplikatów)."""
    # 1. Przygotowujemy sztuczny stan Streamlita (zwykły dict)
    mock_state = {"resolutions": {"0.25 m/px": 0.25, "1 m/px": 1.0}}
    monkeypatch.setattr(st, "session_state", mock_state)

    # 2. Wywołujemy funkcję tak, jakby wgrywano zdjęcie 0.5m/px
    label = set_native_resolution(0.50)

    # 3. Sprawdzamy, czy funkcja zwróciła odpowiedni napis
    assert label == "wykryta: 0.50 m/px"
    # I czy ten napis faktycznie zapisał się w słowniku
    assert mock_state["resolutions"]["wykryta: 0.50 m/px"] == 0.50


def test_set_native_resolution_overwrites_old_native(monkeypatch):
    """Test sprawdza, czy po wgraniu drugiego zdjęcia poprzednia 'wykryta' znika."""
    mock_state = {"resolutions": {"0.25 m/px": 0.25, "wykryta: 0.10 m/px": 0.10}}
    monkeypatch.setattr(st, "session_state", mock_state)

    set_native_resolution(2.0)

    # Stara rozdzielczość 0.10 musiała zostać usunięta
    assert "wykryta: 0.10 m/px" not in mock_state["resolutions"]
    # Zastąpiła ją nowa
    assert mock_state["resolutions"]["wykryta: 2.00 m/px"] == 2.0


def test_reset_image_state(monkeypatch):
    """Test twardego resetu przed załadowaniem nowego zdjęcia."""
    mock_state = {
        "img_np": [1, 2, 3],
        "img_source": "upload",
        "mask_gt": [0, 0],
        "seg_running": True,
        "resolutions": {"wykryta: 0.50 m/px": 0.50}
    }
    monkeypatch.setattr(st, "session_state", mock_state)

    reset_image_state()

    # Sprawdzamy, czy aplikacja wyczyściła główne zmienne z pamięci RAM
    assert "img_np" not in mock_state
    assert "img_source" not in mock_state
    assert "seg_running" not in mock_state

    # Pamięć po masce GT powinna być wyzerowana
    assert mock_state["mask_gt"] is None
    # Dynamiczne rozdzielczości powinny zniknąć
    assert "wykryta: 0.50 m/px" not in mock_state["resolutions"]