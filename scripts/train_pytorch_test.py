import logging

import pytest
import safetensors.torch
import torch

import scripts.train_pytorch as train_pytorch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.match = torch.nn.Linear(2, 2)
        self.action_in_proj = torch.nn.Linear(3, 4)
        self.action_out_proj = torch.nn.Linear(4, 3)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = torch.nn.Linear(2, 2)
        self.action_in_proj = torch.nn.Linear(2, 2)
        self.action_out_proj = torch.nn.Linear(2, 2)
        self.time_mlp_in = torch.nn.Linear(2, 2)
        self.time_mlp_out = torch.nn.Linear(2, 2)


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


def test_apply_pytorch_freeze_filter_keeps_action_expert_trainable():
    model = _TinyModel()

    train_pytorch.apply_pytorch_freeze_filter(model, "vlm_except_action_expert")

    named = dict(model.named_parameters())
    assert named["paligemma_with_expert.paligemma.weight"].requires_grad is False
    assert named["paligemma_with_expert.gemma_expert.weight"].requires_grad is True
    assert named["action_in_proj.weight"].requires_grad is True
    assert named["action_out_proj.weight"].requires_grad is True


def test_create_adamw_optimizer_disables_foreach_temporaries():
    model = TinyModel()

    optim = train_pytorch.create_adamw_optimizer(
        train_pytorch.trainable_parameters(model),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )

    assert isinstance(optim, torch.optim.AdamW)
    assert all(group["foreach"] is False for group in optim.param_groups)
