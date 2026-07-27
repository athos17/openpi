import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

WUJI_JOINT_ACTION_DIM = 54
_IMAGE_KEYS = ("head_view", "left_wrist_view", "right_wrist_view")
_MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def make_wuji_example() -> dict:
    """Creates a random input example for the WUJI policy."""
    return {
        "images": {
            "head_view": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "left_wrist_view": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "right_wrist_view": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "state": np.random.rand(WUJI_JOINT_ACTION_DIM).astype(np.float32),
        "prompt": "Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _require_action_dim(name: str, value, action_dim: int) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[-1] != action_dim:
        raise ValueError(f"WUJI {name} must have trailing dimension {action_dim}; got shape {value.shape}")
    return value


@dataclasses.dataclass(frozen=True)
class WujiInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    action_dim: int = WUJI_JOINT_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type: {self.model_type}")

        images_in = data.get("images")
        if not isinstance(images_in, dict):
            raise ValueError('WujiInputs requires an "images" dictionary')

        missing_images = [key for key in _IMAGE_KEYS if key not in images_in]
        if missing_images:
            raise ValueError(f"Missing WUJI image key(s): {', '.join(missing_images)}")

        state = _require_action_dim("state", data["state"], self.action_dim)
        images = tuple(_parse_image(images_in[key]) for key in _IMAGE_KEYS)

        inputs = {
            "state": state,
            "image": dict(zip(_MODEL_IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(_MODEL_IMAGE_KEYS, (np.True_, np.True_, np.True_), strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = _require_action_dim("actions", data["actions"], self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class WujiNormStatsInputs(transforms.DataTransformFn):
    action_dim: int = WUJI_JOINT_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        result = {
            "state": _require_action_dim("state", data["state"], self.action_dim),
            "actions": _require_action_dim("actions", data["actions"], self.action_dim),
        }
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class WujiOutputs(transforms.DataTransformFn):
    action_dim: int = WUJI_JOINT_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
