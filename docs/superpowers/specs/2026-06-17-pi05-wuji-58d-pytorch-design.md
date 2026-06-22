# pi0.5 WUJI 58D PyTorch Fine-Tuning Design

Date: 2026-06-17

## Scope

Design the repository changes needed to fine-tune pi0.5 with the PyTorch trainer on the local WUJI/GR00T LeRobot dataset:

`/data_all/liyunhao/Isaac-GR00T/examples/wuji_rot6d/spray_water_rot6d_rosbag_ts_split`

This document is a design spec only. It does not implement the changes.

## Current Context

The target dataset is already in LeRobot v2 format:

- `codebase_version`: `v2.0`
- `fps`: about `30.033`
- `total_episodes`: `156`
- `total_frames`: `114814`
- `observation.state`: `float32[58]`
- `action`: `float32[58]`
- Video keys:
  - `observation.images.head_view`
  - `observation.images.left_wrist_view`
  - `observation.images.right_wrist_view`
- Task prompt from `meta/tasks.jsonl`:
  - `Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers`

The 58 action/state dimensions are:

- dims `0:9`: left end-effector xyz + rot6d
- dims `9:18`: right end-effector xyz + rot6d
- dims `18:38`: left hand 20 joints
- dims `38:58`: right hand 20 joints

The existing repository has examples for ALOHA, LIBERO, and DROID. None of those action spaces match this dataset:

- ALOHA assumes 14 dimensions and ALOHA camera names.
- DROID assumes 7 arm joints + 1 gripper and DROID camera names.
- pi0.5 defaults to `action_dim=32`; this dataset needs `action_dim=58`.

## Goals

- Add a clean WUJI-specific data path for PyTorch pi0.5 fine-tuning.
- Preserve the full 58D action/state semantics.
- Reuse pi0.5 base model weights where tensor shapes are compatible.
- Randomly initialize only the shape-incompatible 58D action projection layers.
- Compute and use WUJI-specific normalization statistics.
- Keep the design compatible with training and later policy serving.

## Non-Goals

- Do not compress 58D actions into 32D.
- Do not train pi0.5 from scratch.
- Do not reuse DROID, ALOHA, or other robot normalization statistics.
- Do not introduce relative rot6d actions in the first implementation.
- Do not change the dataset files.
- Do not implement ROS/runtime execution in this design.

## Chosen Approach

Use a native 58D pi0.5 PyTorch fine-tuning path.

The model config sets `action_dim=58`. The WUJI policy transform maps dataset fields into the standard pi0.5 observation interface while preserving full state/action vectors. The PyTorch trainer loads a converted pi0.5 base checkpoint with shape-compatible loading: compatible tensors are loaded, and incompatible tensors such as the action input/output projection layers are skipped and kept randomly initialized.

This approach keeps the action semantics intact and minimizes custom model changes.

## Components

### WUJI Policy Transform

Add `src/openpi/policies/wuji_policy.py`.

This module should define:

- `WujiInputs`
- `WujiOutputs`
- optional helper `make_wuji_example`

`WujiInputs` receives repacked data with this shape:

```python
{
    "images": {
        "head_view": ...,
        "left_wrist_view": ...,
        "right_wrist_view": ...,
    },
    "state": np.ndarray[58],
    "actions": np.ndarray[action_horizon, 58],  # training only
    "prompt": str,  # when present
}
```

It produces the model format:

```python
{
    "image": {
        "base_0_rgb": head_view,
        "left_wrist_0_rgb": left_wrist_view,
        "right_wrist_0_rgb": right_wrist_view,
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    },
    "state": state_58d,
    "actions": actions_58d,
    "prompt": prompt,
}
```

Images should be converted to `uint8` HWC if the LeRobot loader returns float CHW tensors, following the local pattern in `droid_policy.py`.

`WujiOutputs` should return the full unnormalized action array:

```python
{"actions": np.asarray(data["actions"][..., :58])}
```

The `:58` slice is defensive and should not truncate valid dimensions because the model action dimension is 58.

### WUJI Data Config

Add `LeRobotWujiDataConfig` in `src/openpi/training/config.py`.

The repack transform should map LeRobot fields into WUJI transform inputs:

```python
_transforms.RepackTransform(
    {
        "images": {
            "head_view": "observation.images.head_view",
            "left_wrist_view": "observation.images.left_wrist_view",
            "right_wrist_view": "observation.images.right_wrist_view",
        },
        "state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    }
)
```

Because the raw dataset contains `task_index` and LeRobot metadata tasks rather than a direct `prompt` column, the training config should use:

```python
base_config=DataConfig(prompt_from_task=True)
```

The data transform should be:

```python
_transforms.Group(
    inputs=[wuji_policy.WujiInputs(model_type=model_config.model_type)],
    outputs=[wuji_policy.WujiOutputs()],
)
```

The model transform should use `ModelTransformFactory()`.

The config should leave `action_sequence_keys=("action",)`, which is the default and matches the dataset.

### PyTorch Training Config

Add a training config in `_CONFIGS`:

```python
TrainConfig(
    name="pi05_wuji_spray_water_rot6d_pytorch",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=58,
        action_horizon=16,
        max_token_len=200,
    ),
    data=LeRobotWujiDataConfig(
        repo_id="/data_all/liyunhao/Isaac-GR00T/examples/wuji_rot6d/spray_water_rot6d_rosbag_ts_split",
        assets=AssetsConfig(asset_id="wuji_spray_water_rot6d"),
        base_config=DataConfig(prompt_from_task=True),
    ),
    pytorch_weight_path="./checkpoints/pi05_base_pytorch_converted",
    batch_size=16,
    num_train_steps=20_000,
    save_interval=1000,
)
```

`batch_size=16` is the conservative default for pi0.5 PyTorch fine-tuning with 58D actions and three camera streams. If memory allows, increase to `32`.

`action_horizon=16` is the first-pass default. At about 30Hz this covers roughly 0.53 seconds, and it matches existing pi0.5 DROID fine-tuning defaults.

`max_token_len=200` is the pi0.5 default and should be sufficient for a 58D discretized state plus the single task prompt. Training should watch for tokenizer truncation warnings. If warnings appear frequently, raise it to `256`.

### PyTorch Shape-Compatible Weight Loading

The PyTorch trainer currently loads `config.pytorch_weight_path` through `safetensors.torch.load_model`. With `action_dim=58`, direct loading from a 32D pi0.5 checkpoint will encounter shape mismatches in action projection layers.

Add a shape-compatible loader path for PyTorch checkpoints:

- Load tensors from `model.safetensors`.
- Iterate over current model `state_dict`.
- Copy only tensors whose names exist in the checkpoint and whose shapes match exactly.
- Keep all missing or shape-mismatched tensors at their initialized values.
- Log:
  - loaded tensor count
  - skipped missing tensor count
  - skipped shape-mismatch tensor count
  - each shape-mismatched key and source/target shape
- Treat shape mismatches as expected for:
  - `action_in_proj.weight`
  - `action_out_proj.weight`
  - `action_out_proj.bias`
- Fail only if the checkpoint path is missing or if no tensors are loaded.

The intended result is:

- vision tower, language model, action expert, and pi0.5 time-conditioning weights load from base.
- 58D action input/output heads are randomly initialized and trained on WUJI data.

If parameter names differ after conversion, the loader should report the unmatched names clearly rather than silently succeeding.

### Base Checkpoint Conversion

The PyTorch trainer expects a PyTorch checkpoint directory containing `model.safetensors`.

If only the JAX pi0.5 base checkpoint is available, convert it with the existing conversion flow before training. Use `./checkpoints/pi05_base_pytorch_converted` as the local converted checkpoint directory for this config.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --config_name pi05_wuji_spray_water_rot6d_pytorch \
    --output_path ./checkpoints/pi05_base_pytorch_converted
```

The converted checkpoint may still contain 32D action projection tensors if produced from a 32D base model. Shape-compatible loading must still be used during fine-tuning.

### Normalization Statistics

Compute WUJI-specific norm stats before training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/compute_norm_stats.py \
    --config-name pi05_wuji_spray_water_rot6d_pytorch
```

The current `compute_norm_stats.py` writes to:

```python
output_path = config.assets_dirs / data_config.repo_id
```

That is unsafe when `repo_id` is an absolute local path. It can write outside the intended assets directory and later training will not find stats under `asset_id`.

Change the output path to:

```python
output_path = config.assets_dirs / data_config.asset_id
```

When `asset_id` is missing, fail with a clear error. The WUJI config must set `asset_id="wuji_spray_water_rot6d"`.

Expected stats location:

```text
assets/pi05_wuji_spray_water_rot6d_pytorch/wuji_spray_water_rot6d/norm_stats.json
```

The model type is pi0.5, so `use_quantile_norm=True` through the existing `DataConfigFactory.create_base_config` behavior. The computed stats must include quantile fields `q01` and `q99`.

### Action Semantics

First implementation should train absolute 58D actions:

- eef xyz target values remain absolute.
- eef rot6d target values remain absolute.
- hand joint target values remain absolute.

Do not apply `DeltaActions` in the initial WUJI data config.

Reasoning:

- Existing `DeltaActions` subtracts state dimensions directly.
- Direct subtraction is acceptable for some joint-position spaces but not for rot6d orientation.
- Proper relative rot6d action would require rotation-matrix composition/inversion and conversion back to rot6d.

A future design may add relative action conversion for eef positions and orientations, but that is out of scope for this spec.

## Data Flow

Training sample flow:

1. `LeRobotDataset` reads local dataset from `repo_id`.
2. `PromptFromLeRobotTask` converts `task_index` to `prompt`.
3. `RepackTransform` maps LeRobot keys to `images`, `state`, `actions`, `prompt`.
4. `WujiInputs` maps images/state/actions to pi0.5 model input names and preserves 58D arrays.
5. `Normalize` applies WUJI-specific quantile norm stats to `state` and `actions`.
6. `ModelTransformFactory` resizes images, tokenizes prompt and state, and pads state/actions to `action_dim=58`.
7. `scripts/train_pytorch.py` receives PyTorch tensors and trains pi0.5.

Inference flow:

1. Runtime observation is repacked to WUJI policy input format.
2. `WujiInputs` maps images and 58D state into model format.
3. `Normalize` applies checkpoint norm stats.
4. Model samples actions with shape `[action_horizon, 58]`.
5. `Unnormalize` converts actions back to WUJI scale.
6. `WujiOutputs` returns the full 58D action chunk.

## Error Handling

Add explicit validation in WUJI transforms:

- If a required image key is missing, raise `ValueError` naming the missing key.
- If `state.shape[-1] != 58`, raise `ValueError`.
- If training actions exist and `actions.shape[-1] != 58`, raise `ValueError`.
- If prompt is bytes, decode as UTF-8.
- If image dtype is floating point, convert to `uint8`.
- If image is CHW, convert to HWC.

Add explicit validation in norm stats computation:

- If `data_config.asset_id` is missing, fail before writing stats.
- Print the final stats path.

Add explicit validation in PyTorch weight loading:

- If `model.safetensors` does not exist, fail with the expected path.
- If zero tensors load, fail.
- Log skipped shape mismatches.

## Testing Strategy

Unit tests:

- `WujiInputs` maps three images to `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`.
- `WujiInputs` preserves `state.shape == (58,)`.
- `WujiInputs` preserves training `actions.shape[-1] == 58`.
- `WujiOutputs` returns `actions.shape[-1] == 58`.
- Missing image/state/action shape errors are explicit.
- Shape-compatible PyTorch loader loads matching tensors and skips mismatched tensors.
- `compute_norm_stats.py` writes under `asset_id`, not absolute `repo_id`.

Integration checks:

- Create a data loader for `pi05_wuji_spray_water_rot6d_pytorch` with `num_batches=1`.
- Confirm a batch has:
  - three image tensors
  - `state.shape[-1] == 58`
  - `actions.shape[-1] == 58`
  - tokenized prompt present
- Run norm stats computation on a bounded frame count for a smoke test.
- Run one PyTorch training step or a tiny training run with dummy/small settings after stats exist.

Operational checks:

- Confirm training logs show expected shape-compatible weight skips for action projection layers.
- Confirm no tokenizer truncation warnings, or increase `max_token_len`.
- Confirm checkpoint save contains copied assets and can be loaded by `serve_policy.py`.

## Commands

Compute norm stats:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/compute_norm_stats.py \
    --config-name pi05_wuji_spray_water_rot6d_pytorch
```

Run PyTorch training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/train_pytorch.py \
    pi05_wuji_spray_water_rot6d_pytorch \
    --exp_name spray_water_rot6d_58d \
    --save_interval 1000
```

If using two GPUs:

```bash
UV_CACHE_DIR=/tmp/uv-cache torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py \
    pi05_wuji_spray_water_rot6d_pytorch \
    --exp_name spray_water_rot6d_58d \
    --save_interval 1000
```

Serve the step-1000 checkpoint:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_wuji_spray_water_rot6d_pytorch \
    --policy.dir=checkpoints/pi05_wuji_spray_water_rot6d_pytorch/spray_water_rot6d_58d/1000
```

## Acceptance Criteria

- The WUJI config can instantiate without errors.
- Norm stats are written to `assets/pi05_wuji_spray_water_rot6d_pytorch/wuji_spray_water_rot6d/norm_stats.json`.
- A transformed training batch has `state.shape[-1] == 58` and `actions.shape[-1] == 58`.
- PyTorch weight loading succeeds while explicitly skipping only expected shape-incompatible tensors.
- A one-step PyTorch training smoke test completes after norm stats are available.
- Inference policy returns action chunks with shape `[action_horizon, 58]`.

## Risks

- 58D action head starts from random initialization, so early training loss may be noisy.
- The dataset has one task and about 115k frames, so overfitting is possible.
- Absolute rot6d actions may be harder to learn than carefully designed relative actions.
- PyTorch checkpoint conversion may expose parameter naming differences that need loader logging to diagnose.
- `uv run` may need `UV_CACHE_DIR=/tmp/uv-cache` in restricted environments because the default cache path may be read-only.

## Future Work

- Add a mathematically correct relative eef action transform for xyz and rot6d.
- Add robot-runtime repack transforms for live WUJI inference.
- Tune `action_horizon`, batch size, and `max_token_len` based on training logs.
- Evaluate whether freezing parts of the backbone improves stability for this small single-task dataset.
