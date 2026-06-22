# pi0.5 WUJI Subtask PyTorch Training Design

Date: 2026-06-22

## Scope

Design the repository changes needed to reproduce the subtask training modes from
`/data_all/liyunhao/openpi_subtask` for the WUJI 58D PyTorch training path in this
repository.

The target local dataset is:

`/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask`

This document is a design spec only. It does not implement the changes.

## Current Context

The current repository already has a WUJI 58D PyTorch training path:

- `src/openpi/policies/wuji_policy.py`
- `LeRobotWujiDataConfig` in `src/openpi/training/config.py`
- `pi05_wuji_spray_water_rot6d_pytorch`
- `scripts/train_pytorch.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`

That path supports three camera views, 58D state, 58D action chunks, and a high-level
task prompt. It only trains the continuous action flow matching objective.

The reference subtask repository implements hierarchical pi0.5 training for JAX/NNX
models. It adds:

- high-level and low-level prompts
- token masks for subtask text and FAST action tokens
- separate loss weights for subtask token loss, FAST action token loss, and flow matching loss
- multiple training modes:
  - subtask + flow
  - subtask + FAST
  - action expert only
  - subtask + FAST + flow hybrid
- subtask generation at inference time through `sample_low_level_task`

The WUJI target needs the same training modes, but in the PyTorch model path.

## Dataset Format

The target dataset is LeRobot v2 format:

- `total_episodes`: `156`
- `total_frames`: `110053`
- `observation.state`: `float32[58]`
- `action`: `float32[58]`
- video keys:
  - `observation.images.head_view`
  - `observation.images.left_wrist_view`
  - `observation.images.right_wrist_view`
- `task_index`: high-level task index
- `subtask_index`: low-level subtask index
- `task_index_high_level`: present but currently `-1` in samples

Subtask metadata:

- `meta/tasks.jsonl` maps `task_index` to the high-level task text.
- `meta/subtasks.jsonl` and `meta/subtasks.parquet` map:
  - `0` -> `pick up bottle`
  - `1` -> `pump`
  - `2` -> `spray`
- `meta/lerobot_annotations.json` stores episode-level subtask intervals.
- The parquet episode files already contain per-frame `subtask_index`.

Current annotation coverage:

- episode `0` has valid subtask labels for most frames.
- episode `1` has valid subtask labels for most frames.
- episodes `2+` currently have `subtask_index == -1`.

The implementation must treat `subtask_index == -1` as unlabeled. Unlabeled frames
must not be used for subtask or FAST token supervision. The first implementation should
filter them out for all subtask training modes. After the user labels all trajectories,
the same training configs should use all newly valid frames without model changes.

## Goals

- Add WUJI subtask training configs that match all training strategies implemented in
  `openpi_subtask`.
- Keep the existing WUJI 58D action/state semantics unchanged.
- Keep the existing non-subtask WUJI PyTorch config working.
- Implement the subtask machinery in the PyTorch path, not by switching WUJI training
  to the JAX/NNX reference path.
- Support training from the converted pi0.5 PyTorch base checkpoint.
- Make current partially labeled data usable for smoke tests.
- Make fully labeled future data usable without code changes.

## Non-Goals

- Do not modify the dataset files.
- Do not infer subtask labels for unlabeled frames.
- Do not compress 58D actions to a smaller action space.
- Do not implement ROS or real-robot execution in this design.
- Do not port the full async websocket stack in the first training implementation.
- Do not make FAST token loss mandatory for all modes.

## Training Modes

Add four WUJI PyTorch configs matching the reference repository.

### Subtask + Flow

Config name:

`pi05_wuji_spray_water_rot6d_subtask_flow_pytorch`

Loss weights:

- `subtask_loss_weight > 0`
- `fast_token_loss_weight = 0`
- `flow_matching_loss_weight > 0`

This mode trains the model to generate the current subtask text and to predict
continuous 58D action chunks through flow matching.

### Subtask + FAST

Config name:

`pi05_wuji_spray_water_rot6d_subtask_fast_pytorch`

Loss weights:

- `subtask_loss_weight > 0`
- `fast_token_loss_weight > 0`
- `flow_matching_loss_weight = 0`

This mode corresponds to the first stage of the knowledge-insulation strategy in
`openpi_subtask`. It trains token generation only: subtask text plus FAST action tokens.

### Action Expert

Config name:

`pi05_wuji_spray_water_rot6d_action_expert_pytorch`

Loss weights:

- `subtask_loss_weight = 0`
- `fast_token_loss_weight = 0`
- `flow_matching_loss_weight > 0`

This mode corresponds to the second stage of the knowledge-insulation strategy. It
initializes from a subtask+FAST checkpoint and trains the continuous action expert.

The PyTorch trainer should support a freeze policy for this mode:

- freeze the PaliGemma VLM branch by default
- keep the action expert and flow projection layers trainable

The first implementation may expose this as a train config option interpreted by the
PyTorch trainer, because the existing JAX `freeze_filter` does not directly apply to
PyTorch module parameters.

### Hybrid Joint Training

Config name:

`pi05_wuji_spray_water_rot6d_subtask_hybrid_pytorch`

Loss weights:

- `subtask_loss_weight > 0`
- `fast_token_loss_weight > 0`
- `flow_matching_loss_weight > 0`

This mode is the joint or centralized training strategy from `openpi_subtask`. It trains
subtask generation, FAST action token prediction, and continuous action flow matching in
one run.

When FAST token loss and flow loss are both enabled, the flow branch must not condition
on ground-truth FAST action tokens. Before computing the flow prefix, action-token
positions should be masked out of `tokenized_prompt` and `tokenized_prompt_mask`. This
matches the reference implementation and avoids train/inference mismatch.

## Model Config Changes

Extend `Pi0Config` with fields used by PyTorch subtask training:

- `subtask_loss_weight: float = 0.0`
- `fast_token_loss_weight: float = 0.0`
- `flow_matching_loss_weight: float = 1.0`
- `fast_tokenizer_path: str = "physical-intelligence/fast"`
- `stop_gradient_flow_to_prefix: bool = False`
- `pytorch_freeze_filter: str | None = None`

Defaults must preserve existing behavior:

- no subtask token loss
- no FAST token loss
- flow matching enabled
- no PyTorch freezing unless requested

The existing `pi05_wuji_spray_water_rot6d_pytorch` config should behave the same after
these fields are added.

## Data Pipeline Design

Add a WUJI subtask data config, likely `LeRobotWujiSubtaskDataConfig`.

It should reuse the existing WUJI image/state/action mapping:

```python
{
    "images": {
        "head_view": "observation.images.head_view",
        "left_wrist_view": "observation.images.left_wrist_view",
        "right_wrist_view": "observation.images.right_wrist_view",
    },
    "state": "observation.state",
    "actions": "action",
    "task_index": "task_index",
    "subtask_index": "subtask_index",
}
```

Then a dedicated transform should map indices to text:

```python
{
    "high_prompt": tasks[task_index],
    "low_prompt": subtasks[subtask_index],
}
```

The transform must:

- accept scalar tensors, NumPy arrays, or Python ints for both indices
- decode bytes if the LeRobot loader returns string-like values
- raise a clear error if `subtask_index` is unknown
- reject `subtask_index == -1` unless the sample has already been filtered

### Filtering Unlabeled Frames

Use a dataset wrapper after `LeRobotDataset` construction and before model transforms.
The wrapper should keep only samples whose `subtask_index >= 0`.

This avoids:

- failing tokenization on unlabeled samples
- applying subtask loss to invalid labels
- silently training on guessed labels

For the current partially labeled dataset, this leaves approximately 1,894 frames from
episodes 0 and 1. When future annotation is complete, the wrapper will retain all
validly labeled frames.

If loader performance becomes a problem, the wrapper can precompute valid indices once
from the HuggingFace dataset column rather than scanning parquet files on every access.

## Policy Transform Design

Extend or subclass the WUJI policy transform for subtask training.

The transform should output:

- `image`
- `image_mask`
- `state`
- `actions`
- `high_prompt`
- `low_prompt`

The existing `WujiInputs` can remain unchanged for normal WUJI training. A new
`WujiSubtaskInputs` is preferable because it keeps the non-subtask behavior simple and
lets tests target subtask-specific fields directly.

`WujiSubtaskInputs` should still validate:

- exactly three WUJI camera inputs are present
- `state.shape[-1] == 58`
- `actions.shape[-1] == 58`

## Tokenization Design

Port the reference high/low prompt tokenizer behavior to this repository.

Add a transform:

`TokenizeHighLowPrompt`

Inputs:

- `high_prompt`
- `low_prompt`
- normalized `state`
- optional normalized `actions`

Outputs:

- `tokenized_prompt`
- `tokenized_prompt_mask`
- `token_ar_mask`
- `token_loss_mask`
- `subtask_region_mask`
- `action_region_mask`

Prompt structure:

```text
Task: <high task>; State: <discretized state>; Subtask: <low subtask>;
Action: <FAST action tokens>|<eos>
```

When `fast_token_loss_weight == 0`, the tokenizer should not require the FAST tokenizer
and should omit the FAST action-token segment. It should still terminate the subtask
segment with the action delimiter and EOS in the same style as the reference flow mode.

When `fast_token_loss_weight > 0`, load the FAST tokenizer from `fast_tokenizer_path`,
convert the normalized continuous action chunk to FAST tokens, and map them into the
tail of the PaliGemma vocabulary as in `openpi_subtask`.

The model config should use a larger `max_token_len` for WUJI subtask configs than the
existing WUJI config because the prompt includes:

- high-level task text
- 58 discretized state values
- low-level subtask text
- optional FAST action tokens

Initial values:

- flow mode: `max_token_len=320`
- subtask+FAST mode: `max_token_len=384`
- action expert mode: `max_token_len=320`
- hybrid mode: `max_token_len=384`

These values should be adjusted if token truncation warnings appear.

## Observation Schema

Extend `openpi.models.model.Observation` with:

- `subtask_region_mask`
- `action_region_mask`

Update:

- `Observation.from_dict`
- `Observation.to_dict`
- `preprocess_observation`
- `preprocess_observation_pytorch`

Both masks are optional. Existing models and configs should continue to work when they
are absent.

## PyTorch Model Design

Extend `PI0Pytorch.forward` to compute up to three loss components.

### Token Loss

When either `subtask_loss_weight` or `fast_token_loss_weight` is positive:

1. Build prefix embeddings from images and the full tokenized prompt.
2. Run the PaliGemma language branch.
3. Project hidden states through the PaliGemma language head.
4. Compute next-token cross entropy against `tokenized_prompt[:, 1:]`.
5. Apply `subtask_region_mask[:, 1:]` for subtask loss.
6. Apply `action_region_mask[:, 1:]` for FAST action token loss.

Each loss should be normalized by the number of active mask tokens per sample, with a
minimum denominator of one to avoid division by zero.

### Flow Loss

When `flow_matching_loss_weight` is positive:

1. Use the current continuous action flow implementation.
2. If FAST token loss is also active and `action_region_mask` exists, build a flow-only
   observation with action-token positions masked out.
3. Compute MSE against the denoising target as today.
4. Apply `flow_matching_loss_weight`.

### Return Value and Logging

The model can return either:

- a tensor loss with attached auxiliary metrics through a side channel, or
- a dictionary containing `loss`, `flow_loss`, `subtask_loss`, and `fast_token_loss`.

The training loop should support the dictionary form because it makes wandb logging
and debugging clearer.

Required metrics:

- `train/loss`
- `train/flow_loss`
- `train/subtask_loss`
- `train/fast_token_loss`
- `train/grad_norm`
- current learning rate

## PyTorch Freezing for Action Expert Mode

Add a small PyTorch freeze helper used by `scripts/train_pytorch.py`.

For `pytorch_freeze_filter == "vlm_except_action_expert"`:

- set `requires_grad=False` for PaliGemma language and vision parameters
- keep `gemma_expert`, `action_in_proj`, `action_out_proj`, and pi0.5 time MLP parameters trainable

Log trainable and frozen parameter counts before optimizer construction. The optimizer
must receive only trainable parameters.

## Inference Design

The first implementation should focus on training. However, the code shape should not
block later inference.

Add PyTorch helpers after training support is stable:

- `sample_low_level_task`
- `sample_actions_with_subtask`

These should mirror the reference behavior:

1. Tokenize high-level prompt and state up to `Subtask:`.
2. Autoregressively decode the low-level subtask.
3. Use the generated subtask prefix for continuous action denoising.

The websocket server integration is outside this first design scope. The saved
checkpoints and transforms should contain enough metadata to add serving later.

## Config Values

All WUJI subtask configs should use:

- `action_dim=58`
- `action_horizon=16`
- `repo_id="/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask"`
- `assets=AssetsConfig(asset_id="wuji_spray_water_rot6d_subtask")`
- `prompt_from_task=False`, because high prompt extraction is handled by the subtask
  metadata transform
- `lerobot_tolerance_s=0.08`
- `pytorch_weight_path="./checkpoints/pi05_base_pytorch_converted"`

Initial hyperparameters:

- subtask+flow: batch size `16`, `20_000` steps
- subtask+FAST: batch size `16`, `20_000` steps
- action expert: batch size `16`, `8_000` steps
- hybrid: batch size `16`, `40_000` steps

The small currently labeled subset is only suitable for smoke tests and overfitting
checks. Meaningful training should wait until the remaining trajectories are annotated.

## Tests

Add focused unit tests:

- `WujiSubtaskInputs` maps three cameras, 58D arrays, `high_prompt`, and `low_prompt`.
- subtask metadata loading maps `subtask_index` values to expected text.
- unlabeled `subtask_index == -1` samples are filtered or rejected with a clear error.
- `TokenizeHighLowPrompt` emits all expected masks.
- `subtask_region_mask` covers low-level subtask tokens and not the high-level prefix.
- `action_region_mask` is all false when FAST loss is disabled.
- `action_region_mask` is nonempty when FAST loss is enabled.
- `Observation.from_dict` preserves subtask and action region masks.
- `preprocess_observation_pytorch` preserves both masks.
- each new train config instantiates and points to the subtask dataset.
- a one-batch PyTorch forward smoke test returns finite losses for:
  - subtask+flow
  - subtask+FAST
  - action expert
  - hybrid

## Verification Commands

Expected verification after implementation:

```bash
uv run pytest src/openpi/policies/wuji_policy_test.py src/openpi/training/wuji_config_test.py src/openpi/transforms_test.py
```

```bash
uv run python - <<'PY'
from openpi.training import config as _config
from openpi.training import data_loader

for name in [
    "pi05_wuji_spray_water_rot6d_subtask_flow_pytorch",
    "pi05_wuji_spray_water_rot6d_subtask_fast_pytorch",
    "pi05_wuji_spray_water_rot6d_action_expert_pytorch",
    "pi05_wuji_spray_water_rot6d_subtask_hybrid_pytorch",
]:
    cfg = _config.get_config(name)
    loader = data_loader.create_data_loader(cfg, framework="pytorch", shuffle=False, num_batches=1, skip_norm_stats=True)
    obs, actions = next(iter(loader))
    print(name, obs.state.shape, actions.shape)
PY
```

```bash
uv run scripts/train_pytorch.py pi05_wuji_spray_water_rot6d_subtask_flow_pytorch \
  --exp-name=smoke_subtask_flow \
  --num-train-steps=2 \
  --save-interval=1 \
  --overwrite \
  --wandb-enabled=false
```

Repeat the two-step smoke run for the other three configs after the first mode passes.

## Risks

- The current labeled subset is very small. It can validate mechanics but cannot produce
  a reliable policy.
- FAST tokenizer availability may require a local Hugging Face snapshot. The code should
  fail with a clear message when `fast_token_loss_weight > 0` and the tokenizer is
  missing.
- Token length may exceed the initial `max_token_len` because WUJI has 58 state values
  and optional FAST action tokens.
- PyTorch language-head loss may increase memory usage. If needed, disable gradient
  checkpointing only for debugging, not for default training.
- Action expert freezing in PyTorch must be checked by parameter names, which are less
  structured than the JAX filter system.

## Acceptance Criteria

- All four WUJI subtask PyTorch configs exist and instantiate.
- Existing non-subtask WUJI PyTorch config still instantiates and trains as before.
- Current partially labeled dataset can produce a one-batch dataloader for each subtask
  config.
- PyTorch forward computes finite loss values for enabled objectives in each mode.
- Hybrid mode masks GT FAST action tokens out of the flow prefix.
- Action expert mode freezes VLM parameters and optimizes only trainable action-expert
  parameters.
- The design remains compatible with future fully annotated WUJI trajectories without
  requiring model or config changes.
