import pytest
import streamlit as st
from utils.session import set_native_resolution, reset_image_state


# add detected res to the session state (no duplicates)
def test_set_native_resolution(monkeypatch):
    # fake state
    mock_state = {"resolutions": {"0.25 m/px": 0.25, "1 m/px": 1.0}}
    monkeypatch.setattr(st, "session_state", mock_state)

    # fake res = 0.5m/px
    label = set_native_resolution(0.50)

    assert label == "wykryta: 0.50 m/px"
    assert mock_state["resolutions"]["wykryta: 0.50 m/px"] == 0.50


# replace old detected res with the new one
def test_set_native_resolution_overwrite(monkeypatch):
    mock_state = {"resolutions": {"0.25 m/px": 0.25, "wykryta: 0.10 m/px": 0.10}}
    monkeypatch.setattr(st, "session_state", mock_state)

    set_native_resolution(2.0)

    # old res deleted
    assert "wykryta: 0.10 m/px" not in mock_state["resolutions"]
    # new ores added
    assert mock_state["resolutions"]["wykryta: 2.00 m/px"] == 2.0


# reset all params before uploading new img
def test_reset_image_state(monkeypatch):
    mock_state = {
        "img_np": [1, 2, 3],
        "img_source": "upload",
        "mask_gt": [0, 0],
        "seg_running": True,
        "resolutions": {"wykryta: 0.50 m/px": 0.50}
    }
    monkeypatch.setattr(st, "session_state", mock_state)

    reset_image_state()

    # check if old params were deleted from RAM
    assert "img_np" not in mock_state
    assert "img_source" not in mock_state
    assert "seg_running" not in mock_state

    assert mock_state["mask_gt"] is None
    assert "wykryta: 0.50 m/px" not in mock_state["resolutions"]