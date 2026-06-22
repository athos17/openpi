import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import wuji_policy


def _example() -> dict:
    return {
        "images": {
            "head_view": np.zeros((3, 4, 5), dtype=np.float32),
            "left_wrist_view": np.ones((4, 5, 3), dtype=np.uint8),
            "right_wrist_view": np.full((3, 4, 5), 0.5, dtype=np.float32),
        },
        "state": np.arange(58, dtype=np.float32),
        "actions": np.ones((16, 58), dtype=np.float32),
        "prompt": b"spray water",
    }


def test_wuji_inputs_map_three_cameras_and_preserve_58d_arrays():
    transform = wuji_policy.WujiInputs(model_type=_model.ModelType.PI05)

    result = transform(_example())

    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["right_wrist_0_rgb"].shape == (4, 5, 3)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    }
    assert result["state"].shape == (58,)
    assert result["actions"].shape == (16, 58)
    assert result["prompt"] == "spray water"


def test_wuji_inputs_reject_missing_image_key():
    data = _example()
    del data["images"]["left_wrist_view"]

    with pytest.raises(ValueError, match="left_wrist_view"):
        wuji_policy.WujiInputs(model_type=_model.ModelType.PI05)(data)


def test_wuji_inputs_reject_non_58d_state_and_actions():
    transform = wuji_policy.WujiInputs(model_type=_model.ModelType.PI05)

    data = _example()
    data["state"] = np.zeros((57,), dtype=np.float32)
    with pytest.raises(ValueError, match="state.*58"):
        transform(data)

    data = _example()
    data["actions"] = np.zeros((16, 57), dtype=np.float32)
    with pytest.raises(ValueError, match="actions.*58"):
        transform(data)


def test_wuji_outputs_return_58d_actions():
    actions = np.ones((16, 60), dtype=np.float32)

    result = wuji_policy.WujiOutputs()({"actions": actions})

    assert result["actions"].shape == (16, 58)


def test_wuji_norm_stats_inputs_validate_58d_without_images():
    result = wuji_policy.WujiNormStatsInputs()(
        {
            "state": np.zeros((58,), dtype=np.float32),
            "actions": np.zeros((16, 58), dtype=np.float32),
            "prompt": "ignored",
        }
    )

    assert result["state"].shape == (58,)
    assert result["actions"].shape == (16, 58)
