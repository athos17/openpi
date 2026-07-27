import logging

import pytest
import safetensors.torch
import torch

from scripts import train_pytorch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.match = torch.nn.Linear(2, 2)
        self.action_in_proj = torch.nn.Linear(3, 4)
        self.action_out_proj = torch.nn.Linear(4, 3)


def test_shape_compatible_loader_loads_matching_tensors_and_skips_mismatches(tmp_path, caplog):
    model = TinyModel()
    original_mismatch = model.action_in_proj.weight.detach().clone()
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(
        {
            "match.weight": torch.full_like(model.match.weight, 3.0),
            "match.bias": torch.full_like(model.match.bias, 4.0),
            "action_in_proj.weight": torch.ones((4, 2)),
            "extra.weight": torch.ones((1,)),
        },
        checkpoint_path,
    )

    with caplog.at_level(logging.INFO):
        result = train_pytorch.load_shape_compatible_safetensors(model, checkpoint_path)

    assert torch.all(model.match.weight == 3.0)
    assert torch.all(model.match.bias == 4.0)
    assert torch.equal(model.action_in_proj.weight, original_mismatch)
    assert result.loaded == 2
    assert "action_in_proj.weight" in result.shape_mismatched
    assert "action_in_proj.weight" in caplog.text


def test_shape_compatible_loader_fails_when_no_tensors_load(tmp_path):
    model = TinyModel()
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file({"action_in_proj.weight": torch.ones((4, 2))}, checkpoint_path)

    with pytest.raises(ValueError, match="No tensors"):
        train_pytorch.load_shape_compatible_safetensors(model, checkpoint_path)


def test_shape_compatible_loader_rejects_unexpected_mismatch(tmp_path):
    model = TinyModel()
    original_match = model.match.weight.detach().clone()
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(
        {
            "match.weight": torch.ones((2, 3)),
            "match.bias": torch.ones_like(model.match.bias),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="match.weight"):
        train_pytorch.load_shape_compatible_safetensors(model, checkpoint_path)

    assert torch.equal(model.match.weight, original_match)
