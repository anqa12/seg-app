import pytest
import torch
import torch.nn as nn
import numpy as np
from utils.inference import tile_layout, _align_to_32, predict, check_rescale
from utils.config import NUM_CLASSES


# test model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # for obtaining the device (cuda or cpu)
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x.shape = (Batch, Channels, Height, Width)
        B, _, H, W = x.shape
        return torch.randn(B, NUM_CLASSES, H, W).to(self.dummy_param.device)


# round to multiple of 32
def test_align_to_32():
    assert _align_to_32(10) == 32
    assert _align_to_32(31) == 32
    assert _align_to_32(33) == 32
    assert _align_to_32(48) == 64  # 48/32 = 1.5 -> int(1.5 + 0.5) * 32 = 64
    assert _align_to_32(60) == 64


# cut the image with overlap = 0
def test_tile_layout_no_overlap():
    tile_size, ys, xs = tile_layout(H=128, W=128, tile_size=64, overlap=0.0)

    assert tile_size == 64
    assert ys == [0, 64]
    assert xs == [0, 64]


# cut the image with overlap = 0.5
def test_tile_layout_with_overlap():
    # img 100x100, tile 64, overlap 50%, stride 32
    # first: 0..64, second: 32..96, 4px are missing to 100, so it will be one more tile 36..100
    tile_size, ys, xs = tile_layout(H=100, W=100, tile_size=64, overlap=0.5)

    assert tile_size == 64
    assert ys == [0, 32, 36]
    assert xs == [0, 32, 36]


# img smaller than tile size (automatically rescaled to the smalest posiible multiple of 32
def test_tile_layout_image_smaller_than_tile():
    # img 50x50, tile 128x128
    tile_size, ys, xs = tile_layout(H=50, W=50, tile_size=128, overlap=0.0)

    assert tile_size == 32
    # tile = 32px, so stride is 32, first 0..32, second 18..50
    assert ys == [0, 18]
    assert xs == [0, 18]


# do not allow images smaller than 32x32
def test_tile_layout_too_small_image():
    tile_size, ys, xs = tile_layout(H=20, W=20, tile_size=64, overlap=0.0)

    assert tile_size == 0
    assert ys == []
    assert xs == []


# do not allow upsample from lower res img to higher res img
def test_check_rescale():
    is_valid, reason = check_rescale((500, 500), 4.0)

    assert is_valid is False
    assert "powiększanie" in reason


# test combining tiles back together without gaussian func
def test_predict_without_voting():
    model = DummyModel()
    # img
    img_np = np.zeros((100, 100, 3), dtype=np.uint8)

    result = predict(model, img_np, tile_size=64, overlap=0.0, use_voting=False)

    # check if the size is back 100x100
    assert result is not None
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
    assert result.max() < NUM_CLASSES


# test combining tiles back together with tile overlap = 0.5 and gaussian voting
def test_predict_with_gaussian_voting():
    model = DummyModel()
    img_np = np.zeros((150, 150, 3), dtype=np.uint8)

    result = predict(model, img_np, tile_size=64, overlap=0.5, use_voting=True)

    assert result is not None
    assert result.shape == (150, 150)
    assert result.dtype == np.uint8