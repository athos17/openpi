# WUJI Subtask Open-Loop Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-checkpoint-only WUJI subtask open-loop evaluation script with `fast` and `full` modes: `fast` writes a main-view video annotated with the model-generated subtask, and `full` writes that video plus the original action trajectory plot.

**Architecture:** Create a focused script `scripts/open_loop_eval_subtask.py` that reuses stable helpers from `scripts/open_loop_eval.py` for dataset loading, trajectory bounds, ground-truth action extraction, metrics, and plotting. Add small, testable helper functions for output path resolution, subtask text scheduling, main-camera frame extraction, video annotation, and local PyTorch subtask policy inference. Keep the existing `scripts/open_loop_eval.py` unchanged.

**Tech Stack:** Python, NumPy, PyTorch-backed OpenPI policy internals, LeRobot dataset access through existing OpenPI config/data-loader code, OpenCV (`cv2`) if available for MP4 writing and text overlay, pytest for unit tests.

---

## File Structure

- Create `scripts/open_loop_eval_subtask.py`
  - CLI dataclass and `main()` entrypoint.
  - Local checkpoint policy creation only.
  - Subtask-aware evaluation loop.
  - Video rendering helpers.
  - Reuse `scripts.open_loop_eval` helpers where possible.
- Create `scripts/open_loop_eval_subtask_test.py`
  - Unit tests for path resolution, subtask schedule expansion, frame extraction, video overlay behavior, fast/full mode behavior, and fake-policy evaluation behavior.
- Reuse `scripts/open_loop_eval.py`
  - `create_data_config()`
  - `create_lerobot_dataset()`
  - `get_episode_bounds()`
  - `prepare_repacked_item()` where useful for ground-truth chunks
  - `_extract_action_chunk()`
  - `_to_numpy()`
  - `collect_state_trajectory()`
  - `plot_trajectory_results()`

## Task 1: CLI paths and mode semantics

**Files:**
- Create: `scripts/open_loop_eval_subtask.py`
- Create: `scripts/open_loop_eval_subtask_test.py`

- [ ] **Step 1: Write failing tests for video/plot path resolution and mode behavior**

Add this to `scripts/open_loop_eval_subtask_test.py`:

```python
import dataclasses
import pathlib

from . import open_loop_eval_subtask


def test_resolve_video_path_uses_directory_for_multiple_trajectories(tmp_path: pathlib.Path):
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        traj_ids=[0, 1],
        save_video_path=str(tmp_path / "videos"),
    )

    assert open_loop_eval_subtask.resolve_video_path(args, traj_id=1) == tmp_path / "videos" / "traj_1.mp4"


def test_resolve_video_path_preserves_file_path_for_single_trajectory(tmp_path: pathlib.Path):
    path = tmp_path / "eval.mp4"
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        traj_ids=[0],
        save_video_path=str(path),
    )

    assert open_loop_eval_subtask.resolve_video_path(args, traj_id=0) == path


def test_resolve_plot_path_returns_none_in_fast_mode(tmp_path: pathlib.Path):
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        mode="fast",
        save_plot_path=str(tmp_path / "plots"),
    )

    assert open_loop_eval_subtask.resolve_plot_path(args, traj_id=0) is None


def test_resolve_plot_path_delegates_to_original_open_loop_in_full_mode(tmp_path: pathlib.Path):
    path = tmp_path / "plot.jpeg"
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        mode="full",
        traj_ids=[0],
        save_plot_path=str(path),
    )

    assert open_loop_eval_subtask.resolve_plot_path(args, traj_id=0) == path


def test_checkpoint_dir_is_required_for_local_subtask_eval():
    args = open_loop_eval_subtask.Args(config_name="debug", checkpoint_dir=None)

    try:
        open_loop_eval_subtask.validate_args(args)
    except ValueError as exc:
        assert "checkpoint_dir" in str(exc)
    else:
        raise AssertionError("validate_args should reject missing checkpoint_dir")
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: FAIL during import because `scripts.open_loop_eval_subtask` does not exist.

- [ ] **Step 3: Implement minimal CLI dataclass and path helpers**

Create `scripts/open_loop_eval_subtask.py` with:

```python
"""Subtask-aware open-loop evaluation for local WUJI OpenPI checkpoints."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import tyro

from . import open_loop_eval


@dataclasses.dataclass
class Args:
    """Arguments for local subtask open-loop evaluation."""

    config_name: str = tyro.MISSING
    """Training config name used to build dataset transforms and checkpoint policy."""

    checkpoint_dir: str | None = None
    """Local checkpoint directory. Required; websocket policies are intentionally unsupported."""

    default_prompt: str | None = None
    """Default prompt injected by locally loaded policies when the dataset does not provide one."""

    dataset_repo_id: str | None = None
    """Optional LeRobot repo ID or local dataset path overriding the selected config's dataset."""

    traj_ids: list[int] = dataclasses.field(default_factory=lambda: [0])
    """Episode IDs to evaluate."""

    steps: int = 1000
    """Maximum number of steps per episode. This is capped by the episode length."""

    action_horizon: int | None = None
    """Number of actions to consume from each predicted chunk. Defaults to the model action horizon."""

    num_steps: int | None = None
    """Optional denoising steps passed to local policy sampling."""

    mode: Literal["fast", "full"] = "fast"
    """fast writes only annotated video; full writes annotated video and action plots."""

    save_video_path: str | None = None
    """File or directory path for annotated main-view videos."""

    save_plot_path: str | None = None
    """File or directory path for trajectory plots in full mode."""

    plot_state: bool = True
    """Whether full-mode plots include state traces when state/action dimensions match."""

    video_backend: str | None = None
    """Optional LeRobot video backend."""

    video_fps: int | None = None
    """Output video FPS. Defaults to dataset metadata FPS when available, otherwise 30."""

    main_camera_key: str = "observation.images.head_view"
    """Raw dataset key for the video main camera."""

    max_subtask_decoding_steps: int = 25
    """Maximum generated low-level subtask token count per inference point."""

    subtask_temperature: float = 0.0
    """Temperature for subtask token sampling."""


def validate_args(args: Args) -> None:
    if args.checkpoint_dir is None:
        raise ValueError("Subtask open-loop evaluation requires --checkpoint-dir for a local checkpoint.")


def resolve_video_path(args: Args, traj_id: int) -> Path | None:
    if args.save_video_path is None:
        return None
    path = Path(args.save_video_path)
    if len(args.traj_ids) > 1 or path.suffix == "" or path.is_dir():
        return path / f"traj_{traj_id}.mp4"
    return path


def resolve_plot_path(args: Args, traj_id: int) -> Path | None:
    if args.mode != "full" or args.save_plot_path is None:
        return None
    original_args = open_loop_eval.Args(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        dataset_repo_id=args.dataset_repo_id,
        traj_ids=args.traj_ids,
        steps=args.steps,
        action_horizon=args.action_horizon,
        num_steps=args.num_steps,
        save_plot_path=args.save_plot_path,
        plot=True,
        plot_state=args.plot_state,
        video_backend=args.video_backend,
    )
    return open_loop_eval.resolve_plot_path(original_args, traj_id)


def main(args: Args) -> None:
    validate_args(args)
    raise NotImplementedError("Subtask open-loop evaluation is implemented in later tasks.")


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS for Task 1 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
git commit -m "test: add subtask open loop eval cli helpers"
```

## Task 2: Frame extraction, subtask schedule, and video overlay helpers

**Files:**
- Modify: `scripts/open_loop_eval_subtask.py`
- Modify: `scripts/open_loop_eval_subtask_test.py`

- [ ] **Step 1: Write failing tests for pure video helpers**

Append to `scripts/open_loop_eval_subtask_test.py`:

```python
import numpy as np


class FrameDataset:
    def __init__(self):
        self.items = [
            {"observation.images.head_view": np.zeros((2, 3, 3), dtype=np.uint8)},
            {"observation.images.head_view": np.ones((2, 3, 3), dtype=np.float32)},
        ]

    def __getitem__(self, index):
        return self.items[index]


def test_expand_subtask_schedule_repeats_chunk_text_until_next_inference_point():
    schedule = open_loop_eval_subtask.expand_subtask_schedule(
        actual_steps=5,
        action_horizon=2,
        chunk_subtasks=["pick up bottle", "pump", "spray"],
    )

    assert schedule == ["pick up bottle", "pick up bottle", "pump", "pump", "spray"]


def test_extract_main_frame_supports_raw_dataset_key_and_float_images():
    dataset = FrameDataset()

    frame0 = open_loop_eval_subtask.extract_main_frame(dataset[0], "observation.images.head_view")
    frame1 = open_loop_eval_subtask.extract_main_frame(dataset[1], "observation.images.head_view")

    assert frame0.dtype == np.uint8
    assert frame0.shape == (2, 3, 3)
    assert frame1.dtype == np.uint8
    assert frame1.max() == 255


def test_overlay_subtask_text_changes_pixels_in_top_right_region():
    frame = np.zeros((80, 160, 3), dtype=np.uint8)

    annotated = open_loop_eval_subtask.overlay_subtask_text(frame, "pump")

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated[:40, 80:] != frame[:40, 80:])
    assert np.all(frame == 0), "overlay_subtask_text must not mutate the input frame"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: FAIL with missing functions `expand_subtask_schedule`, `extract_main_frame`, or `overlay_subtask_text`.

- [ ] **Step 3: Implement helper functions**

Add imports and functions to `scripts/open_loop_eval_subtask.py`:

```python
from typing import Any

import numpy as np


def expand_subtask_schedule(*, actual_steps: int, action_horizon: int, chunk_subtasks: list[str]) -> list[str]:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive; got {action_horizon}")
    schedule: list[str] = []
    for chunk_index, subtask in enumerate(chunk_subtasks):
        start = chunk_index * action_horizon
        end = min(start + action_horizon, actual_steps)
        if start >= actual_steps:
            break
        schedule.extend([subtask] * (end - start))
    if len(schedule) != actual_steps:
        raise ValueError(
            f"Subtask schedule has {len(schedule)} frames, expected {actual_steps}; "
            f"got {len(chunk_subtasks)} chunk subtasks with horizon {action_horizon}."
        )
    return schedule


def _lookup_nested(data: dict[str, Any], dotted_key: str) -> Any:
    if dotted_key in data:
        return data[dotted_key]
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Could not find camera key '{dotted_key}' in dataset item.")
        current = current[part]
    return current


def extract_main_frame(item: dict[str, Any], main_camera_key: str) -> np.ndarray:
    frame = np.asarray(_lookup_nested(item, main_camera_key))
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8, copy=False)
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Main camera frame must have shape HxWxC or CxHxW; got {frame.shape}")
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.ascontiguousarray(frame)


def overlay_subtask_text(frame: np.ndarray, subtask: str) -> np.ndarray:
    import cv2

    annotated = np.asarray(frame).copy()
    label = f"Subtask: {subtask}" if subtask else "Subtask: <empty>"
    if len(label) > 48:
        label = label[:45] + "..."

    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(width, height) / 450.0)
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 8
    x0 = max(0, width - text_w - 2 * pad)
    y0 = 0
    x1 = width
    y1 = min(height, text_h + baseline + 2 * pad)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        label,
        (x0 + pad, y0 + pad + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated
```

- [ ] **Step 4: Run tests and verify Task 2 passes**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
git commit -m "feat: add subtask video annotation helpers"
```

## Task 3: Local subtask policy adapter with fake-policy tests

**Files:**
- Modify: `scripts/open_loop_eval_subtask.py`
- Modify: `scripts/open_loop_eval_subtask_test.py`

- [ ] **Step 1: Write failing tests for subtask inference adapter**

Append to `scripts/open_loop_eval_subtask_test.py`:

```python
import torch


class FakeLocalSubtaskPolicy:
    def __init__(self):
        self.calls = []

    def infer_subtask_then_actions(
        self,
        repacked,
        *,
        action_horizon,
        num_steps,
        max_subtask_decoding_steps,
        subtask_temperature,
    ):
        start = int(repacked["state"][0])
        self.calls.append((start, action_horizon, num_steps, max_subtask_decoding_steps, subtask_temperature))
        return open_loop_eval_subtask.SubtaskInferenceResult(
            actions=np.asarray([[start + offset, -(start + offset)] for offset in range(action_horizon)], dtype=np.float32),
            subtask=f"subtask-{start}",
            subtask_tokens=np.asarray([start, start + 1], dtype=np.int64),
        )


def test_fake_subtask_policy_adapter_returns_action_chunk_and_subtask():
    policy = FakeLocalSubtaskPolicy()

    result = policy.infer_subtask_then_actions(
        {"state": np.asarray([3.0, 0.0], dtype=np.float32)},
        action_horizon=2,
        num_steps=4,
        max_subtask_decoding_steps=5,
        subtask_temperature=0.0,
    )

    assert result.subtask == "subtask-3"
    assert result.subtask_tokens.tolist() == [3, 4]
    np.testing.assert_allclose(result.actions, np.asarray([[3, -3], [4, -4]], dtype=np.float32))
    assert policy.calls == [(3, 2, 4, 5, 0.0)]


def test_decode_subtask_tokens_strips_eos_when_tokenizer_has_decode():
    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=True):
            assert skip_special_tokens is True
            return " pump\n"

    text = open_loop_eval_subtask.decode_subtask_tokens(Tokenizer(), torch.tensor([[7, 1, 0]]))

    assert text == "pump"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: FAIL with missing `SubtaskInferenceResult` and `decode_subtask_tokens`.

- [ ] **Step 3: Implement result dataclass and token decoding helper**

Add to `scripts/open_loop_eval_subtask.py`:

```python
@dataclasses.dataclass(frozen=True)
class SubtaskInferenceResult:
    actions: np.ndarray
    subtask: str
    subtask_tokens: np.ndarray


def decode_subtask_tokens(tokenizer: Any, tokens: Any) -> str:
    token_array = open_loop_eval._to_numpy(tokens)
    if token_array.ndim == 2:
        token_array = token_array[0]
    token_list = [int(token) for token in token_array.tolist()]
    if 1 in token_list:
        token_list = token_list[: token_list.index(1)]
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_list, skip_special_tokens=True)).strip()
    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "decode"):
        return str(tokenizer._tokenizer.decode(token_list, skip_special_tokens=True)).strip()
    return " ".join(str(token) for token in token_list).strip()
```

- [ ] **Step 4: Run tests and verify Task 3 passes**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS.

- [ ] **Step 5: Add local PyTorch policy adapter implementation**

Add imports and class to `scripts/open_loop_eval_subtask.py`:

```python
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class LocalSubtaskPolicyAdapter:
    def __init__(self, policy):
        self._policy = policy
        self._model = getattr(policy, "_model", None)
        self._device = getattr(policy, "_pytorch_device", None)
        self._input_transform = getattr(policy, "_input_transform", None)
        self._output_transform = getattr(policy, "_output_transform", None)
        self._tokenizer = getattr(getattr(self._model, "paligemma_with_expert", None), "tokenizer", None)
        if self._model is None or self._device is None or self._input_transform is None or self._output_transform is None:
            raise ValueError("Local subtask eval requires a PyTorch policy with model, device, and transforms.")
        if not hasattr(self._model, "sample_low_level_task") or not hasattr(self._model, "denoise_step"):
            raise ValueError("Checkpoint model does not expose subtask generation APIs.")

    def infer_subtask_then_actions(
        self,
        repacked: dict[str, Any],
        *,
        action_horizon: int,
        num_steps: int,
        max_subtask_decoding_steps: int,
        subtask_temperature: float,
    ) -> SubtaskInferenceResult:
        import torch

        transformed = self._input_transform(dict(repacked))
        batched = {key: _batch_value(value) for key, value in transformed.items()}
        observation = _model.Observation.from_dict(batched)
        observation = _model.preprocess_observation(observation, train=False)
        observation = _move_observation_to_device(observation, self._device)

        output_tokens, past_key_values, prefix_pad_masks, _ = self._model.sample_low_level_task(
            self._device,
            observation,
            max_decoding_steps=max_subtask_decoding_steps,
            paligemma_eos_token=1,
            temperature=subtask_temperature,
        )
        subtask = decode_subtask_tokens(self._tokenizer, output_tokens)

        bsize = observation.state.shape[0]
        actions_shape = (bsize, self._model.config.action_horizon, self._model.config.action_dim)
        noise = self._model.sample_noise(actions_shape, self._device)
        _, _, _, _, state = self._model._preprocess_observation(observation, train=False)  # noqa: SLF001

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=self._device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=self._device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt

        raw_actions = open_loop_eval._to_numpy(x_t)[0]
        unnormalized = self._output_transform({"actions": raw_actions})
        actions = open_loop_eval._extract_action_chunk(unnormalized, action_horizon, source="subtask policy")
        return SubtaskInferenceResult(
            actions=actions,
            subtask=subtask,
            subtask_tokens=open_loop_eval._to_numpy(output_tokens[0]),
        )


def _batch_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _batch_value(child) for key, child in value.items()}
    return np.expand_dims(value, axis=0)


def _move_observation_to_device(observation: _model.Observation, device) -> _model.Observation:
    import torch

    def move(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {key: move(child) for key, child in value.items()}
        tensor = torch.as_tensor(value, device=device)
        if tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)
        return tensor

    return _model.Observation.from_dict({key: move(value) for key, value in observation.to_dict().items()})


def create_local_subtask_policy(args: Args, train_config: _config.TrainConfig) -> LocalSubtaskPolicyAdapter:
    validate_args(args)
    sample_kwargs = {"num_steps": args.num_steps} if args.num_steps is not None else None
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs=sample_kwargs,
        default_prompt=args.default_prompt,
    )
    return LocalSubtaskPolicyAdapter(policy)
```

- [ ] **Step 6: Run tests after adapter addition**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
git commit -m "feat: add local subtask policy adapter"
```

## Task 4: Subtask-aware trajectory evaluation

**Files:**
- Modify: `scripts/open_loop_eval_subtask.py`
- Modify: `scripts/open_loop_eval_subtask_test.py`

- [ ] **Step 1: Write failing trajectory evaluation test using fake dataset/policy**

Append to `scripts/open_loop_eval_subtask_test.py`:

```python
from openpi.training import config as _config
import openpi.transforms as _transforms


class FakeSubtaskDataset:
    def __init__(self, *, length: int, action_horizon: int):
        self.episode_data_index = {"from": np.asarray([0]), "to": np.asarray([length])}
        self._length = length
        self._action_horizon = action_horizon

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        action_steps = [min(index + offset, self._length - 1) for offset in range(self._action_horizon)]
        return {
            "state": np.asarray([index, index + 0.5], dtype=np.float32),
            "actions": np.asarray([[step, -step] for step in action_steps], dtype=np.float32),
            "observation.images.head_view": np.full((4, 5, 3), index, dtype=np.uint8),
        }


def test_evaluate_single_trajectory_records_subtasks_and_metrics_without_plot_or_video():
    action_horizon = 2
    dataset = FakeSubtaskDataset(length=5, action_horizon=action_horizon)
    policy = FakeLocalSubtaskPolicy()
    data_config = _config.DataConfig(
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform({"state": "state", "actions": "actions"})]
        )
    )

    result = open_loop_eval_subtask.evaluate_single_trajectory(
        policy=policy,
        dataset=dataset,
        data_config=data_config,
        traj_id=0,
        steps=5,
        action_horizon=action_horizon,
        num_steps=4,
        max_subtask_decoding_steps=5,
        subtask_temperature=0.0,
        save_video_path=None,
        save_plot_path=None,
        main_camera_key="observation.images.head_view",
        video_fps=30,
        plot_state=True,
    )

    assert result.actual_steps == 5
    assert result.chunk_subtasks == ["subtask-0", "subtask-2", "subtask-4"]
    assert result.frame_subtasks == ["subtask-0", "subtask-0", "subtask-2", "subtask-2", "subtask-4"]
    np.testing.assert_allclose(result.mse, 0.0)
    np.testing.assert_allclose(result.mae, 0.0)
    assert policy.calls == [(0, 2, 4, 5, 0.0), (2, 2, 4, 5, 0.0), (4, 2, 4, 5, 0.0)]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: FAIL with missing `evaluate_single_trajectory` and/or trajectory result fields.

- [ ] **Step 3: Implement trajectory result and evaluation loop**

Add to `scripts/open_loop_eval_subtask.py`:

```python
@dataclasses.dataclass(frozen=True)
class SubtaskTrajectoryResult:
    traj_id: int
    mse: float
    mae: float
    actual_steps: int
    gt_actions: np.ndarray
    pred_actions: np.ndarray
    chunk_subtasks: list[str]
    frame_subtasks: list[str]


def evaluate_single_trajectory(
    *,
    policy,
    dataset,
    data_config: _config.DataConfig,
    traj_id: int,
    steps: int,
    action_horizon: int,
    num_steps: int,
    max_subtask_decoding_steps: int,
    subtask_temperature: float,
    save_video_path: str | Path | None,
    save_plot_path: str | Path | None,
    main_camera_key: str,
    video_fps: int,
    plot_state: bool = True,
) -> SubtaskTrajectoryResult:
    start, end = open_loop_eval.get_episode_bounds(dataset, traj_id)
    actual_steps = min(steps, end - start)

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    chunk_subtasks: list[str] = []
    for offset in range(0, actual_steps, action_horizon):
        global_index = start + offset
        repacked = open_loop_eval.prepare_repacked_item(dataset[global_index], data_config, dataset)
        gt_chunks.append(open_loop_eval._extract_action_chunk(repacked, action_horizon, source="ground truth"))
        inference = policy.infer_subtask_then_actions(
            repacked,
            action_horizon=action_horizon,
            num_steps=num_steps,
            max_subtask_decoding_steps=max_subtask_decoding_steps,
            subtask_temperature=subtask_temperature,
        )
        pred_chunks.append(inference.actions)
        chunk_subtasks.append(inference.subtask)

    gt_actions = np.concatenate(gt_chunks, axis=0)[:actual_steps]
    pred_actions = np.concatenate(pred_chunks, axis=0)[:actual_steps]
    if gt_actions.shape != pred_actions.shape:
        raise ValueError(f"gt_actions shape {gt_actions.shape} does not match pred_actions shape {pred_actions.shape}")

    mse = float(np.mean((gt_actions - pred_actions) ** 2))
    mae = float(np.mean(np.abs(gt_actions - pred_actions)))
    frame_subtasks = expand_subtask_schedule(
        actual_steps=actual_steps,
        action_horizon=action_horizon,
        chunk_subtasks=chunk_subtasks,
    )

    if save_video_path is not None:
        write_annotated_video(
            dataset=dataset,
            start=start,
            actual_steps=actual_steps,
            frame_subtasks=frame_subtasks,
            save_video_path=Path(save_video_path),
            main_camera_key=main_camera_key,
            fps=video_fps,
        )

    if save_plot_path is not None:
        state = open_loop_eval.collect_state_trajectory(dataset, data_config, traj_id, actual_steps)
        open_loop_eval.plot_trajectory_results(
            state_across_time=state,
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            traj_id=traj_id,
            action_horizon=action_horizon,
            save_plot_path=Path(save_plot_path),
            plot_state=plot_state,
        )

    return SubtaskTrajectoryResult(
        traj_id=traj_id,
        mse=mse,
        mae=mae,
        actual_steps=actual_steps,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        chunk_subtasks=chunk_subtasks,
        frame_subtasks=frame_subtasks,
    )
```

- [ ] **Step 4: Add temporary video writer stub to unblock non-video test**

Add this below `evaluate_single_trajectory()` in `scripts/open_loop_eval_subtask.py`; Task 5 will replace it with the full implementation:

```python
def write_annotated_video(
    *,
    dataset,
    start: int,
    actual_steps: int,
    frame_subtasks: list[str],
    save_video_path: Path,
    main_camera_key: str,
    fps: int,
) -> None:
    raise NotImplementedError("Video writing is implemented in Task 5.")
```

- [ ] **Step 5: Run tests and verify Task 4 passes**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS because this test uses `save_video_path=None`.

- [ ] **Step 6: Commit**

```bash
git add scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
git commit -m "feat: evaluate subtask open loop trajectories"
```

## Task 5: Annotated MP4 writing

**Files:**
- Modify: `scripts/open_loop_eval_subtask.py`
- Modify: `scripts/open_loop_eval_subtask_test.py`

- [ ] **Step 1: Write failing test for video file creation**

Append to `scripts/open_loop_eval_subtask_test.py`:

```python
def test_write_annotated_video_creates_mp4(tmp_path: pathlib.Path):
    dataset = FakeSubtaskDataset(length=3, action_horizon=1)
    path = tmp_path / "traj_0.mp4"

    open_loop_eval_subtask.write_annotated_video(
        dataset=dataset,
        start=0,
        actual_steps=3,
        frame_subtasks=["pick", "pump", "spray"],
        save_video_path=path,
        main_camera_key="observation.images.head_view",
        fps=10,
    )

    assert path.exists()
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py::test_write_annotated_video_creates_mp4 -q
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `write_annotated_video()` using OpenCV**

Replace the stub in `scripts/open_loop_eval_subtask.py` with:

```python
def write_annotated_video(
    *,
    dataset,
    start: int,
    actual_steps: int,
    frame_subtasks: list[str],
    save_video_path: Path,
    main_camera_key: str,
    fps: int,
) -> None:
    import cv2

    if len(frame_subtasks) != actual_steps:
        raise ValueError(f"Expected {actual_steps} frame subtasks, got {len(frame_subtasks)}")
    if actual_steps <= 0:
        raise ValueError("actual_steps must be positive to write a video")

    save_video_path.parent.mkdir(parents=True, exist_ok=True)
    first = overlay_subtask_text(extract_main_frame(dataset[start], main_camera_key), frame_subtasks[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(save_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {save_video_path}")
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for offset in range(1, actual_steps):
            frame = extract_main_frame(dataset[start + offset], main_camera_key)
            annotated = overlay_subtask_text(frame, frame_subtasks[offset])
            if annotated.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {offset} has shape {annotated.shape[:2]}, expected {(height, width)}"
                )
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
```

- [ ] **Step 4: Run video test and full script test file**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py::test_write_annotated_video_creates_mp4 -q
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
git commit -m "feat: write subtask annotated open loop videos"
```

## Task 6: Main entrypoint wiring and full-mode plot behavior

**Files:**
- Modify: `scripts/open_loop_eval_subtask.py`
- Modify: `scripts/open_loop_eval_subtask_test.py`

- [ ] **Step 1: Write failing test for default FPS resolution**

Append to `scripts/open_loop_eval_subtask_test.py`:

```python
class DatasetWithMetaFps:
    class Meta:
        fps = 17

    meta = Meta()


def test_resolve_video_fps_prefers_args_then_dataset_meta():
    assert open_loop_eval_subtask.resolve_video_fps(24, DatasetWithMetaFps()) == 24
    assert open_loop_eval_subtask.resolve_video_fps(None, DatasetWithMetaFps()) == 17
    assert open_loop_eval_subtask.resolve_video_fps(None, object()) == 30
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py::test_resolve_video_fps_prefers_args_then_dataset_meta -q
```

Expected: FAIL with missing `resolve_video_fps`.

- [ ] **Step 3: Implement FPS helper and replace `main()`**

Add to `scripts/open_loop_eval_subtask.py`:

```python
import logging


def resolve_video_fps(arg_fps: int | None, dataset) -> int:
    if arg_fps is not None:
        return int(arg_fps)
    fps = getattr(getattr(dataset, "meta", None), "fps", None)
    if fps is not None:
        return int(fps)
    return 30
```

Replace `main()` with:

```python
def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    validate_args(args)
    train_config = _config.get_config(args.config_name)
    action_horizon = args.action_horizon or train_config.model.action_horizon
    num_steps = args.num_steps or 10
    data_config = open_loop_eval.create_data_config(
        open_loop_eval.Args(
            config_name=args.config_name,
            checkpoint_dir=args.checkpoint_dir,
            default_prompt=args.default_prompt,
            dataset_repo_id=args.dataset_repo_id,
            action_horizon=args.action_horizon,
            num_steps=args.num_steps,
            video_backend=args.video_backend,
        ),
        train_config,
    )
    policy = create_local_subtask_policy(args, train_config)
    dataset = open_loop_eval.create_lerobot_dataset(data_config, action_horizon, video_backend=args.video_backend)
    video_fps = resolve_video_fps(args.video_fps, dataset)

    logging.info("Dataset length: %d", len(dataset))
    logging.info("Running subtask evaluation on trajectories: %s", args.traj_ids)
    logging.info("Mode: %s", args.mode)

    results: list[SubtaskTrajectoryResult] = []
    for traj_id in args.traj_ids:
        try:
            result = evaluate_single_trajectory(
                policy=policy,
                dataset=dataset,
                data_config=data_config,
                traj_id=traj_id,
                steps=args.steps,
                action_horizon=action_horizon,
                num_steps=num_steps,
                max_subtask_decoding_steps=args.max_subtask_decoding_steps,
                subtask_temperature=args.subtask_temperature,
                save_video_path=resolve_video_path(args, traj_id),
                save_plot_path=resolve_plot_path(args, traj_id),
                main_camera_key=args.main_camera_key,
                video_fps=video_fps,
                plot_state=args.plot_state,
            )
        except ValueError as exc:
            logging.warning("Skipping trajectory %d: %s", traj_id, exc)
            continue
        logging.info(
            "Trajectory %d: MSE=%s MAE=%s subtasks=%s",
            traj_id,
            result.mse,
            result.mae,
            result.chunk_subtasks,
        )
        results.append(result)

    if not results:
        logging.info("No valid trajectories were evaluated.")
        return

    logging.info("Average MSE across all trajectories: %s", np.mean([result.mse for result in results]))
    logging.info("Average MAE across all trajectories: %s", np.mean([result.mae for result in results]))
    logging.info("Done")
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing open-loop tests to ensure no regression**

Run:

```bash
pytest scripts/open_loop_eval_test.py scripts/open_loop_eval_subtask_test.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
git commit -m "feat: wire subtask open loop eval entrypoint"
```

## Task 7: Final verification and manual command documentation

**Files:**
- Modify: `scripts/open_loop_eval_subtask.py` only if verification reveals issues.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest scripts/open_loop_eval_test.py scripts/open_loop_eval_subtask_test.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run lint/import smoke check**

Run:

```bash
python -m compileall scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
```

Expected: compile succeeds without syntax errors.

- [ ] **Step 3: Print CLI help smoke test**

Run:

```bash
python -m scripts.open_loop_eval_subtask --help | sed -n '1,120p'
```

Expected: help includes `--mode`, `--save-video-path`, `--save-plot-path`, `--max-subtask-decoding-steps`, and `--subtask-temperature`.

- [ ] **Step 4: Document example commands in final response**

Use these examples:

```bash
# Fast mode: video only
python -m scripts.open_loop_eval_subtask \
  --config-name pi05_wuji_spray_water_rot6d_subtask_fast_pytorch \
  --checkpoint-dir checkpoints/pi05_wuji_spray_water_rot6d_subtask_fast_pytorch/subtask_fast_h32_8gpu/<checkpoint> \
  --traj-ids 0 \
  --steps 200 \
  --mode fast \
  --save-video-path outputs/subtask_eval_fast

# Full mode: video + original action plot
python -m scripts.open_loop_eval_subtask \
  --config-name pi05_wuji_spray_water_rot6d_subtask_fast_pytorch \
  --checkpoint-dir checkpoints/pi05_wuji_spray_water_rot6d_subtask_fast_pytorch/subtask_fast_h32_8gpu/<checkpoint> \
  --traj-ids 0 \
  --steps 200 \
  --mode full \
  --save-video-path outputs/subtask_eval_full/videos \
  --save-plot-path outputs/subtask_eval_full/plots
```

- [ ] **Step 5: Commit final fixes if any**

If files changed during verification:

```bash
git add scripts/open_loop_eval_subtask.py scripts/open_loop_eval_subtask_test.py
git commit -m "fix: polish subtask open loop eval"
```

If no files changed, skip this commit.

## Self-Review

- Spec coverage: The plan covers local-checkpoint-only loading, `fast` and `full` modes, annotated main-view video with generated subtask in the top-right, full-mode action plots via the original plotting helper, metrics, and tests.
- Placeholder scan: No TBD/TODO placeholders are present. The temporary Task 4 video stub is explicitly replaced in Task 5.
- Type consistency: `Args`, `SubtaskInferenceResult`, `SubtaskTrajectoryResult`, `resolve_video_path`, `resolve_plot_path`, `expand_subtask_schedule`, `extract_main_frame`, `overlay_subtask_text`, `write_annotated_video`, `resolve_video_fps`, and `evaluate_single_trajectory` use consistent names across tasks.
