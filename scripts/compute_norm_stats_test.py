import pathlib

import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms
from scripts import compute_norm_stats


class _AddKey:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, data: dict) -> dict:
        return {**data, self.key: True}


def test_norm_stats_output_path_uses_asset_id_for_absolute_repo_id(tmp_path: pathlib.Path):
    data_config = _config.DataConfig(repo_id="/absolute/local/dataset", asset_id="wuji_spray_water_rot6d")

    output_path = compute_norm_stats.get_output_path(tmp_path, data_config)

    assert output_path == tmp_path / "wuji_spray_water_rot6d"


def test_norm_stats_output_path_requires_asset_id(tmp_path: pathlib.Path):
    data_config = _config.DataConfig(repo_id="/absolute/local/dataset", asset_id=None)

    with pytest.raises(ValueError, match="asset_id"):
        compute_norm_stats.get_output_path(tmp_path, data_config)


def test_norm_stats_transforms_override_data_transforms():
    data_config = _config.DataConfig(
        repo_id="fake",
        repack_transforms=_transforms.Group(inputs=[_AddKey("training_repack")]),
        norm_stats_repack_transforms=_transforms.Group(inputs=[_AddKey("norm_repack")]),
        data_transforms=_transforms.Group(inputs=[_AddKey("data")]),
        norm_stats_transforms=_transforms.Group(inputs=[_AddKey("norm")]),
    )

    transforms = compute_norm_stats.get_input_transforms(data_config)
    result = _transforms.compose(transforms)({})

    assert result["norm"] is True
    assert "data" not in result
    assert result["norm_repack"] is True
    assert "training_repack" not in result


def test_torch_dataloader_uses_raw_lerobot_rows_for_norm_stats_transforms(monkeypatch, tmp_path: pathlib.Path):
    class FakeRawDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "state": np.zeros((58,), dtype=np.float32),
                "actions": np.ones((58,), dtype=np.float32),
            }

    def fail_create_torch_dataset(*args, **kwargs):
        raise AssertionError("LeRobotDataset should not be constructed for raw norm stats")

    monkeypatch.setattr(compute_norm_stats, "load_dataset", lambda *args, **kwargs: FakeRawDataset(), raising=False)
    monkeypatch.setattr(_data_loader, "create_torch_dataset", fail_create_torch_dataset)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_config = _config.DataConfig(
        repo_id=str(tmp_path),
        norm_stats_transforms=_transforms.Group(),
    )

    data_loader, num_batches = compute_norm_stats.create_torch_dataloader(
        data_config,
        action_horizon=16,
        batch_size=1,
        model_config=pi0_config.Pi0Config(),
        num_workers=0,
    )
    batch = next(iter(data_loader))

    assert num_batches == 1
    assert batch["state"].shape == (1, 58)
    assert batch["actions"].shape == (1, 58)
