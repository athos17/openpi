import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

_WUJI_ACTION_DIM = 58
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
        "state": np.random.rand(_WUJI_ACTION_DIM).astype(np.float32),
        "prompt": "Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _require_58d(name: str, value) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[-1] != _WUJI_ACTION_DIM:
        raise ValueError(f"WUJI {name} must have trailing dimension {_WUJI_ACTION_DIM}; got shape {value.shape}")
    return value


def _to_scalar_index(name: str, value) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
    elif array.size == 1:
        value = array.reshape(()).item()
    else:
        raise ValueError(f"{name} must be scalar-like; got shape {array.shape}")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return int(value)


def _decode_prompt(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        value = np.asarray(value).item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return str(value)


@dataclasses.dataclass(frozen=True)
class WujiSubtaskPromptsFromIndices(transforms.DataTransformFn):
    tasks: dict[int, str]
    subtasks: dict[int, str]

    def __call__(self, data: dict) -> dict:
        if "task_index" not in data:
            raise ValueError('Cannot extract high_prompt without "task_index"')
        if "subtask_index" not in data:
            raise ValueError('Cannot extract low_prompt without "subtask_index"')

        task_index = _to_scalar_index("task_index", data["task_index"])
        subtask_index = _to_scalar_index("subtask_index", data["subtask_index"])
        if subtask_index < 0:
            raise ValueError(f"Cannot map unlabeled subtask_index={subtask_index}")
        if task_index not in self.tasks:
            raise ValueError(f"task_index={task_index} not found in task mapping: {self.tasks}")
        if subtask_index not in self.subtasks:
            raise ValueError(f"subtask_index={subtask_index} not found in subtask mapping: {self.subtasks}")
        return {**data, "high_prompt": self.tasks[task_index], "low_prompt": self.subtasks[subtask_index]}


@dataclasses.dataclass(frozen=True)
class WujiInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type: {self.model_type}")

        images_in = data.get("images")
        if not isinstance(images_in, dict):
            raise ValueError('WujiInputs requires an "images" dictionary')

        missing_images = [key for key in _IMAGE_KEYS if key not in images_in]
        if missing_images:
            raise ValueError(f"Missing WUJI image key(s): {', '.join(missing_images)}")

        state = _require_58d("state", data["state"])
        images = tuple(_parse_image(images_in[key]) for key in _IMAGE_KEYS)

        inputs = {
            "state": state,
            "image": dict(zip(_MODEL_IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(_MODEL_IMAGE_KEYS, (np.True_, np.True_, np.True_), strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = _require_58d("actions", data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class WujiSubtaskInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "high_prompt" not in data or "low_prompt" not in data:
            raise ValueError("WujiSubtaskInputs requires high_prompt and low_prompt")

        base = WujiInputs(model_type=self.model_type)({**data, "prompt": data.get("prompt", "")})
        base.pop("prompt", None)
        base["high_prompt"] = _decode_prompt(data["high_prompt"])
        base["low_prompt"] = _decode_prompt(data["low_prompt"])
        return base


@dataclasses.dataclass(frozen=True)
class WujiNormStatsInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        result = {
            "state": _require_58d("state", data["state"]),
            "actions": _require_58d("actions", data["actions"]),
        }
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class WujiOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :_WUJI_ACTION_DIM])}
