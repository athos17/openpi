# WUJI Subtask Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the WUJI 58D PyTorch subtask training modes described in `docs/superpowers/specs/2026-06-22-wuji-subtask-training-design.md`.

**Architecture:** Keep the existing WUJI PyTorch path intact and add subtask-specific extensions at the existing boundaries: WUJI policy transforms, model tokenization transforms, optional observation masks, dataloader filtering, PyTorch model losses, and PyTorch trainer freezing/logging. The normal `pi05_wuji_spray_water_rot6d_pytorch` config must continue using the existing `WujiInputs` and standard `TokenizePrompt` path.

**Tech Stack:** Python dataclasses, NumPy, PyTorch, JAX tree utilities, LeRobot dataset APIs, SentencePiece PaliGemma tokenizer, optional Hugging Face FAST tokenizer, `uv run pytest`.

---

## File Map

- Modify `src/openpi/models/pi0_config.py`: add subtask loss weights, FAST tokenizer path, optional prefix stop-gradient flag, and PyTorch freeze filter config fields.
- Modify `src/openpi/models/model.py`: carry optional `subtask_region_mask` and `action_region_mask` through `Observation` and JAX preprocessing.
- Modify `src/openpi/models_pytorch/preprocessing_pytorch.py`: preserve optional region masks in processed observations.
- Modify `src/openpi/models/tokenizer.py`: add optional high/low prompt tokenization to `PaligemmaTokenizer`.
- Modify `src/openpi/transforms.py`: add `TokenizeHighLowPrompt`.
- Modify `src/openpi/policies/wuji_policy.py`: add `WujiSubtaskInputs` and subtask metadata/index-to-text transform.
- Modify `src/openpi/training/config.py`: add a WUJI subtask data config plus four new train configs.
- Modify `src/openpi/training/data_loader.py`: add a filtered dataset wrapper for `subtask_index >= 0`.
- Modify `src/openpi/models_pytorch/pi0_pytorch.py`: add token losses, weighted flow loss, hybrid action-token masking, and dictionary metrics.
- Modify `scripts/train_pytorch.py`: support dictionary losses, log component metrics, apply PyTorch freezing before optimizer construction, and optimize trainable params only.
- Modify tests:
  - `src/openpi/policies/wuji_policy_test.py`
  - `src/openpi/training/wuji_config_test.py`
  - `src/openpi/transforms_test.py`
  - `src/openpi/models/model_test.py`
  - add or extend a PyTorch model test if a lightweight dummy path is available.

---

### Task 1: Extend Pi0Config Without Changing Defaults

**Files:**
- Modify: `src/openpi/models/pi0_config.py`
- Test: `src/openpi/training/wuji_config_test.py`

- [x] **Step 1: Write the failing config default test**

Add to `src/openpi/training/wuji_config_test.py`:

```python
def test_pi0_config_subtask_defaults_preserve_existing_behavior():
    config = _config.get_config("pi05_wuji_spray_water_rot6d_pytorch")

    assert config.model.subtask_loss_weight == 0.0
    assert config.model.fast_token_loss_weight == 0.0
    assert config.model.flow_matching_loss_weight == 1.0
    assert config.model.fast_tokenizer_path == "physical-intelligence/fast"
    assert config.model.stop_gradient_flow_to_prefix is False
    assert config.model.pytorch_freeze_filter is None
```

- [x] **Step 2: Run the focused test and verify it fails**

Run:

```bash
uv run pytest src/openpi/training/wuji_config_test.py::test_pi0_config_subtask_defaults_preserve_existing_behavior -q
```

Expected: FAIL with an `AttributeError` for the first missing field.

- [x] **Step 3: Add fields to `Pi0Config`**

In `src/openpi/models/pi0_config.py`, add the fields after `discrete_state_input`:

```python
    subtask_loss_weight: float = 0.0
    fast_token_loss_weight: float = 0.0
    flow_matching_loss_weight: float = 1.0
    fast_tokenizer_path: str = "physical-intelligence/fast"
    stop_gradient_flow_to_prefix: bool = False
    pytorch_freeze_filter: str | None = None
```

- [x] **Step 4: Run the focused test and verify it passes**

Run:

```bash
uv run pytest src/openpi/training/wuji_config_test.py::test_pi0_config_subtask_defaults_preserve_existing_behavior -q
```

Expected: PASS.

---

### Task 2: Preserve Subtask Region Masks In Observations

**Files:**
- Modify: `src/openpi/models/model.py`
- Modify: `src/openpi/models_pytorch/preprocessing_pytorch.py`
- Test: `src/openpi/models/model_test.py`

- [x] **Step 1: Write the failing observation mask tests**

Add to `src/openpi/models/model_test.py`:

```python
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch import preprocessing_pytorch


def _observation_dict():
    return {
        "image": {
            "base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
            "left_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
        },
        "image_mask": {
            "base_0_rgb": np.array([True]),
            "left_wrist_0_rgb": np.array([True]),
            "right_wrist_0_rgb": np.array([True]),
        },
        "state": np.zeros((1, 58), dtype=np.float32),
        "tokenized_prompt": np.zeros((1, 8), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((1, 8), dtype=bool),
        "subtask_region_mask": np.array([[False, True, True, False, False, False, False, False]]),
        "action_region_mask": np.array([[False, False, False, True, True, False, False, False]]),
    }


def test_observation_from_dict_preserves_subtask_region_masks():
    obs = _model.Observation.from_dict(_observation_dict())

    assert obs.subtask_region_mask.tolist() == [[False, True, True, False, False, False, False, False]]
    assert obs.action_region_mask.tolist() == [[False, False, False, True, True, False, False, False]]
    assert obs.to_dict()["subtask_region_mask"].tolist() == obs.subtask_region_mask.tolist()
    assert obs.to_dict()["action_region_mask"].tolist() == obs.action_region_mask.tolist()


def test_preprocess_observation_pytorch_preserves_subtask_region_masks():
    batch = {
        key: torch.as_tensor(value) if not isinstance(value, dict) else {k: torch.as_tensor(v) for k, v in value.items()}
        for key, value in _observation_dict().items()
    }
    obs = _model.Observation.from_dict(batch)

    processed = preprocessing_pytorch.preprocess_observation_pytorch(obs, train=False)

    assert torch.equal(processed.subtask_region_mask, obs.subtask_region_mask)
    assert torch.equal(processed.action_region_mask, obs.action_region_mask)
```

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest src/openpi/models/model_test.py::test_observation_from_dict_preserves_subtask_region_masks src/openpi/models/model_test.py::test_preprocess_observation_pytorch_preserves_subtask_region_masks -q
```

Expected: FAIL because `Observation` has no `subtask_region_mask` attribute.

- [x] **Step 3: Add optional fields to `Observation`**

In `src/openpi/models/model.py`, add fields after `token_loss_mask`:

```python
    subtask_region_mask: at.Bool[ArrayT, "*b l"] | None = None
    action_region_mask: at.Bool[ArrayT, "*b l"] | None = None
```

Update `from_dict`:

```python
            subtask_region_mask=data.get("subtask_region_mask"),
            action_region_mask=data.get("action_region_mask"),
```

Update `preprocess_observation` return:

```python
        subtask_region_mask=observation.subtask_region_mask,
        action_region_mask=observation.action_region_mask,
```

- [x] **Step 4: Preserve masks in PyTorch preprocessing**

In `src/openpi/models_pytorch/preprocessing_pytorch.py`, add to `SimpleProcessedObservation` construction:

```python
        subtask_region_mask=getattr(observation, "subtask_region_mask", None),
        action_region_mask=getattr(observation, "action_region_mask", None),
```

- [x] **Step 5: Run tests and verify pass**

Run:

```bash
uv run pytest src/openpi/models/model_test.py::test_observation_from_dict_preserves_subtask_region_masks src/openpi/models/model_test.py::test_preprocess_observation_pytorch_preserves_subtask_region_masks -q
```

Expected: PASS.

---

### Task 3: Add WUJI Subtask Policy Transforms

**Files:**
- Modify: `src/openpi/policies/wuji_policy.py`
- Test: `src/openpi/policies/wuji_policy_test.py`

- [x] **Step 1: Write failing tests for subtask WUJI inputs and metadata mapping**

Add to `src/openpi/policies/wuji_policy_test.py`:

```python
def test_wuji_subtask_inputs_map_prompts_and_58d_arrays():
    data = _example()
    data["high_prompt"] = b"spray the flowers"
    data["low_prompt"] = "pump"

    result = wuji_policy.WujiSubtaskInputs(model_type=_model.ModelType.PI05)(data)

    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["state"].shape == (58,)
    assert result["actions"].shape == (16, 58)
    assert result["high_prompt"] == "spray the flowers"
    assert result["low_prompt"] == "pump"
    assert "prompt" not in result


def test_wuji_subtask_prompts_from_indices_accepts_scalar_values():
    transform = wuji_policy.WujiSubtaskPromptsFromIndices(
        tasks={0: "spray water"},
        subtasks={0: "pick up bottle", 1: "pump", 2: "spray"},
    )

    result = transform({"task_index": np.array(0), "subtask_index": np.array([2])})

    assert result["high_prompt"] == "spray water"
    assert result["low_prompt"] == "spray"


def test_wuji_subtask_prompts_reject_unlabeled_or_unknown_subtask():
    transform = wuji_policy.WujiSubtaskPromptsFromIndices(tasks={0: "spray water"}, subtasks={0: "pick up bottle"})

    with pytest.raises(ValueError, match="unlabeled"):
        transform({"task_index": 0, "subtask_index": -1})

    with pytest.raises(ValueError, match="subtask_index=9"):
        transform({"task_index": 0, "subtask_index": 9})
```

- [x] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest src/openpi/policies/wuji_policy_test.py::test_wuji_subtask_inputs_map_prompts_and_58d_arrays src/openpi/policies/wuji_policy_test.py::test_wuji_subtask_prompts_from_indices_accepts_scalar_values src/openpi/policies/wuji_policy_test.py::test_wuji_subtask_prompts_reject_unlabeled_or_unknown_subtask -q
```

Expected: FAIL because the new classes do not exist.

- [x] **Step 3: Add helper and transforms**

Add to `src/openpi/policies/wuji_policy.py`:

```python
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
```

Add `WujiSubtaskInputs` as a sibling of `WujiInputs`:

```python
@dataclasses.dataclass(frozen=True)
class WujiSubtaskInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = WujiInputs(model_type=self.model_type)({**data, "prompt": data.get("prompt", "")})
        base.pop("prompt", None)
        if "high_prompt" not in data or "low_prompt" not in data:
            raise ValueError("WujiSubtaskInputs requires high_prompt and low_prompt")
        base["high_prompt"] = _decode_prompt(data["high_prompt"])
        base["low_prompt"] = _decode_prompt(data["low_prompt"])
        return base
```

- [x] **Step 4: Run focused tests and verify pass**

Run:

```bash
uv run pytest src/openpi/policies/wuji_policy_test.py::test_wuji_subtask_inputs_map_prompts_and_58d_arrays src/openpi/policies/wuji_policy_test.py::test_wuji_subtask_prompts_from_indices_accepts_scalar_values src/openpi/policies/wuji_policy_test.py::test_wuji_subtask_prompts_reject_unlabeled_or_unknown_subtask -q
```

Expected: PASS.

---

### Task 4: Add Unlabeled Subtask Dataset Filtering

**Files:**
- Modify: `src/openpi/training/config.py`
- Modify: `src/openpi/training/data_loader.py`
- Test: `src/openpi/training/data_loader_test.py`

- [x] **Step 1: Write failing dataset wrapper tests**

Add to `src/openpi/training/data_loader_test.py`:

```python
class _TinySubtaskDataset:
    def __init__(self):
        self.items = [
            {"subtask_index": -1, "value": "drop"},
            {"subtask_index": 0, "value": "keep0"},
            {"subtask_index": np.array([2]), "value": "keep2"},
        ]

    def __getitem__(self, index):
        return self.items[int(index)]

    def __len__(self):
        return len(self.items)


def test_filter_unlabeled_subtask_dataset_keeps_nonnegative_indices():
    dataset = _data_loader.FilterUnlabeledSubtaskDataset(_TinySubtaskDataset())

    assert len(dataset) == 2
    assert dataset[0]["value"] == "keep0"
    assert dataset[1]["value"] == "keep2"
```

- [x] **Step 2: Run the focused test and verify failure**

Run:

```bash
uv run pytest src/openpi/training/data_loader_test.py::test_filter_unlabeled_subtask_dataset_keeps_nonnegative_indices -q
```

Expected: FAIL because `FilterUnlabeledSubtaskDataset` does not exist.

- [x] **Step 3: Add `filter_unlabeled_subtasks` to `DataConfig`**

In `src/openpi/training/config.py`, add to `DataConfig`:

```python
    filter_unlabeled_subtasks: bool = False
```

- [x] **Step 4: Add the dataset wrapper**

Add to `src/openpi/training/data_loader.py` after `TransformedDataset`:

```python
class FilterUnlabeledSubtaskDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._valid_indices = [
            index for index in range(len(dataset)) if self._subtask_index_is_labeled(dataset[index].get("subtask_index"))
        ]

    @staticmethod
    def _subtask_index_is_labeled(value) -> bool:
        array = np.asarray(value)
        if array.shape == ():
            scalar = array.item()
        elif array.size == 1:
            scalar = array.reshape(()).item()
        else:
            raise ValueError(f"subtask_index must be scalar-like; got shape {array.shape}")
        return int(scalar) >= 0

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[self._valid_indices[index.__index__()]]

    def __len__(self) -> int:
        return len(self._valid_indices)
```

In `create_torch_dataset`, after optional `PromptFromLeRobotTask`:

```python
    if data_config.filter_unlabeled_subtasks:
        dataset = FilterUnlabeledSubtaskDataset(dataset)
```

- [x] **Step 5: Run the focused test and verify pass**

Run:

```bash
uv run pytest src/openpi/training/data_loader_test.py::test_filter_unlabeled_subtask_dataset_keeps_nonnegative_indices -q
```

Expected: PASS.

---

### Task 5: Add High/Low Prompt Tokenization And Masks

**Files:**
- Modify: `src/openpi/models/tokenizer.py`
- Modify: `src/openpi/transforms.py`
- Test: `src/openpi/transforms_test.py`

- [x] **Step 1: Write failing transform tests with a fake tokenizer**

Add to `src/openpi/transforms_test.py`:

```python
class _FakeHighLowTokenizer:
    def tokenize_high_low_prompt(self, high_prompt, low_prompt, state, actions=None):
        assert high_prompt == "spray water"
        assert low_prompt == "pump"
        assert state.shape == (58,)
        if actions is None:
            action_mask = np.array([False, False, False, False])
        else:
            action_mask = np.array([False, False, True, True])
        return (
            np.array([1, 2, 3, 0], dtype=np.int32),
            np.array([True, True, True, False]),
            np.array([1, 1, 1, 0], dtype=np.int32),
            np.array([False, True, True, False]),
            np.array([False, True, False, False]),
            action_mask,
        )


def test_tokenize_high_low_prompt_emits_region_masks_without_fast_actions():
    transform = transforms.TokenizeHighLowPrompt(_FakeHighLowTokenizer(), use_fast_tokens=False)

    result = transform({
        "high_prompt": np.array("spray water"),
        "low_prompt": np.array("pump"),
        "state": np.zeros((58,), dtype=np.float32),
        "actions": np.ones((16, 58), dtype=np.float32),
    })

    assert result["tokenized_prompt"].tolist() == [1, 2, 3, 0]
    assert result["subtask_region_mask"].tolist() == [False, True, False, False]
    assert result["action_region_mask"].tolist() == [False, False, False, False]
    assert "high_prompt" not in result
    assert "low_prompt" not in result


def test_tokenize_high_low_prompt_emits_action_mask_with_fast_actions():
    transform = transforms.TokenizeHighLowPrompt(_FakeHighLowTokenizer(), use_fast_tokens=True)

    result = transform({
        "high_prompt": "spray water",
        "low_prompt": "pump",
        "state": np.zeros((58,), dtype=np.float32),
        "actions": np.ones((16, 58), dtype=np.float32),
    })

    assert result["action_region_mask"].tolist() == [False, False, True, True]
```

- [x] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest src/openpi/transforms_test.py::test_tokenize_high_low_prompt_emits_region_masks_without_fast_actions src/openpi/transforms_test.py::test_tokenize_high_low_prompt_emits_action_mask_with_fast_actions -q
```

Expected: FAIL because `TokenizeHighLowPrompt` does not exist.

- [x] **Step 3: Add `PaligemmaTokenizer` high/low support**

In `src/openpi/models/tokenizer.py`, add `import string`.

Change `PaligemmaTokenizer.__init__` to accept an optional FAST tokenizer:

```python
    def __init__(self, max_len: int = 48, fast_tokenizer_path: str | None = None):
        self._max_len = max_len
        self._fast_skip_tokens = 128
        self._fast_tokenizer = None
```

Then load it only when requested:

```python
        if fast_tokenizer_path is not None:
            self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
```

Add these methods to `PaligemmaTokenizer`:

```python
    def tokenize_high_low_prompt(
        self,
        high_prompt: str,
        low_prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_high_text = high_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        cleaned_low_text = low_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))

        if cleaned_high_text and cleaned_high_text[-1] in string.punctuation:
            cleaned_high_text = cleaned_high_text[:-1]
        cleaned_high_text += "."
        prompt_1 = f"Task: {cleaned_high_text}; State: {state_str}; Subtask: "
        tokens_1 = self._tokenizer.encode(prompt_1, add_bos=True)
        ar_mask = [True] * len(tokens_1)
        loss_mask = [False] * len(tokens_1)
        subtask_region_mask = [False] * len(tokens_1)
        action_region_mask = [False] * len(tokens_1)

        if cleaned_low_text and cleaned_low_text[-1] in string.punctuation:
            cleaned_low_text = cleaned_low_text[:-1]
        cleaned_low_text += "."

        if actions is None or self._fast_tokenizer is None:
            prompt_2 = f"{cleaned_low_text};\nAction: "
            tokens_2 = self._tokenizer.encode(prompt_2, add_eos=True)
        else:
            prompt_2 = f"{cleaned_low_text};"
            tokens_2 = self._tokenizer.encode(prompt_2)

        ar_mask += [True] * len(tokens_2)
        loss_mask += [True] * len(tokens_2)
        subtask_region_mask += [True] * len(tokens_2)
        action_region_mask += [False] * len(tokens_2)
        tokens = tokens_1 + tokens_2

        if actions is not None and self._fast_tokenizer is not None:
            action_tokens_fast = self._fast_tokenizer(actions[None])[0]
            action_tokens_pg = self._act_tokens_to_paligemma_tokens(action_tokens_fast)
            action_seq = (
                self._tokenizer.encode("\nAction: ")
                + action_tokens_pg.tolist()
                + self._tokenizer.encode("|", add_eos=True)
            )
            tokens += action_seq
            ar_mask += [True] * len(action_seq)
            loss_mask += [True] * len(action_seq)
            subtask_region_mask += [False] * len(action_seq)
            action_region_mask += [True] * len(action_seq)

        return self._pad_high_low_tokens(tokens, ar_mask, loss_mask, subtask_region_mask, action_region_mask)

    def _pad_high_low_tokens(self, tokens, ar_mask, loss_mask, subtask_region_mask, action_region_mask):
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
            subtask_region_mask = subtask_region_mask + padding
            action_region_mask = action_region_mask + padding
        else:
            if tokens_len > self._max_len:
                logging.warning(
                    f"Token length ({tokens_len}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]
            subtask_region_mask = subtask_region_mask[: self._max_len]
            action_region_mask = action_region_mask[: self._max_len]
        return (
            np.asarray(tokens),
            np.asarray(mask),
            np.asarray(ar_mask, dtype=np.int32),
            np.asarray(loss_mask),
            np.asarray(subtask_region_mask),
            np.asarray(action_region_mask),
        )

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
```

- [x] **Step 4: Add the transform**

In `src/openpi/transforms.py`, add after `TokenizePrompt`:

```python
@dataclasses.dataclass(frozen=True)
class TokenizeHighLowPrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    use_fast_tokens: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        high_prompt = data.pop("high_prompt", None)
        low_prompt = data.pop("low_prompt", None)
        if high_prompt is None or low_prompt is None:
            raise ValueError("Both high_prompt and low_prompt are required for TokenizeHighLowPrompt")
        if not isinstance(high_prompt, str):
            high_prompt = high_prompt.item()
        if not isinstance(low_prompt, str):
            low_prompt = low_prompt.item()
        if (state := data.get("state")) is None:
            raise ValueError("State is required for TokenizeHighLowPrompt")
        actions = data.get("actions") if self.use_fast_tokens else None
        tokens, token_mask, ar_mask, loss_mask, subtask_region_mask, action_region_mask = (
            self.tokenizer.tokenize_high_low_prompt(high_prompt, low_prompt, state, actions)
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "subtask_region_mask": subtask_region_mask,
            "action_region_mask": action_region_mask,
        }
```

- [x] **Step 5: Run focused tests and verify pass**

Run:

```bash
uv run pytest src/openpi/transforms_test.py::test_tokenize_high_low_prompt_emits_region_masks_without_fast_actions src/openpi/transforms_test.py::test_tokenize_high_low_prompt_emits_action_mask_with_fast_actions -q
```

Expected: PASS.

---

### Task 6: Add WUJI Subtask Data Config And Train Configs

**Files:**
- Modify: `src/openpi/training/config.py`
- Test: `src/openpi/training/wuji_config_test.py`

- [x] **Step 1: Write failing train config tests**

Add to `src/openpi/training/wuji_config_test.py`:

```python
def test_wuji_subtask_train_configs_exist_and_point_to_subtask_dataset(monkeypatch):
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())

    expected = {
        "pi05_wuji_spray_water_rot6d_subtask_flow_pytorch": (0.15, 0.0, 1.0, 320, 20_000, None),
        "pi05_wuji_spray_water_rot6d_subtask_fast_pytorch": (10.0, 1.0, 0.0, 384, 20_000, None),
        "pi05_wuji_spray_water_rot6d_action_expert_pytorch": (0.0, 0.0, 1.0, 320, 8_000, "vlm_except_action_expert"),
        "pi05_wuji_spray_water_rot6d_subtask_hybrid_pytorch": (0.15, 0.15, 1.0, 384, 40_000, None),
    }

    for name, (subtask_weight, fast_weight, flow_weight, max_len, steps, freeze_filter) in expected.items():
        config = _config.get_config(name)
        data_config = config.data.create(config.assets_dirs, config.model)

        assert config.model.action_dim == 58
        assert config.model.action_horizon == 16
        assert config.model.subtask_loss_weight == subtask_weight
        assert config.model.fast_token_loss_weight == fast_weight
        assert config.model.flow_matching_loss_weight == flow_weight
        assert config.model.max_token_len == max_len
        assert config.model.pytorch_freeze_filter == freeze_filter
        assert config.pytorch_weight_path == "./checkpoints/pi05_base_pytorch_converted"
        assert config.batch_size == 16
        assert config.num_train_steps == steps
        assert data_config.repo_id == "/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask"
        assert data_config.asset_id == "wuji_spray_water_rot6d_subtask"
        assert data_config.prompt_from_task is False
        assert data_config.filter_unlabeled_subtasks is True
        assert data_config.lerobot_tolerance_s == 0.08
```

- [x] **Step 2: Run focused test and verify failure**

Run:

```bash
uv run pytest src/openpi/training/wuji_config_test.py::test_wuji_subtask_train_configs_exist_and_point_to_subtask_dataset -q
```

Expected: FAIL because the configs and data config do not exist.

- [x] **Step 3: Add `SubtaskModelTransformFactory`**

In `src/openpi/training/config.py`, add near `ModelTransformFactory`:

```python
@dataclasses.dataclass(frozen=True)
class SubtaskModelTransformFactory(GroupFactory):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        assert isinstance(model_config, pi0_config.Pi0Config)
        fast_path = model_config.fast_tokenizer_path if model_config.fast_token_loss_weight > 0 else None
        return _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizeHighLowPrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len, fast_tokenizer_path=fast_path),
                    use_fast_tokens=model_config.fast_token_loss_weight > 0,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )
```

- [x] **Step 4: Add metadata loaders and `LeRobotWujiSubtaskDataConfig`**

In `src/openpi/training/config.py`, add near the WUJI data configs:

```python
def _load_jsonl_text_map(path: pathlib.Path, *, index_key: str, text_key: str) -> dict[int, str]:
    import json

    result = {}
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            result[int(item[index_key])] = str(item[text_key])
    return result


def _load_wuji_subtask_metadata(repo_id: str) -> tuple[dict[int, str], dict[int, str]]:
    repo_path = pathlib.Path(repo_id)
    tasks_path = repo_path / "meta" / "tasks.jsonl"
    subtasks_path = repo_path / "meta" / "subtasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"WUJI subtask dataset is missing {tasks_path}")
    if not subtasks_path.exists():
        raise FileNotFoundError(f"WUJI subtask dataset is missing {subtasks_path}")
    return (
        _load_jsonl_text_map(tasks_path, index_key="task_index", text_key="task"),
        _load_jsonl_text_map(subtasks_path, index_key="subtask_index", text_key="subtask"),
    )
```

Then add after `LeRobotWujiDataConfig`:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotWujiSubtaskDataConfig(DataConfigFactory):
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
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
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            prompt_from_task=False,
            filter_unlabeled_subtasks=True,
        )
        tasks, subtasks = _load_wuji_subtask_metadata(self.repo_id)
        data_transforms = _transforms.Group(
            inputs=[
                wuji_policy.WujiSubtaskPromptsFromIndices(tasks=tasks, subtasks=subtasks),
                wuji_policy.WujiSubtaskInputs(model_type=model_config.model_type),
            ],
            outputs=[wuji_policy.WujiOutputs()],
        )
        norm_stats_transforms = _transforms.Group(inputs=[wuji_policy.WujiNormStatsInputs()])
        norm_stats_repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform({"state": "observation.state", "actions": "action"})]
        )
        return dataclasses.replace(
            base,
            repack_transforms=self.repack_transforms,
            norm_stats_repack_transforms=norm_stats_repack_transform,
            data_transforms=data_transforms,
            norm_stats_transforms=norm_stats_transforms,
            model_transforms=SubtaskModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )
```

- [x] **Step 5: Add four train configs**

Add after `pi05_wuji_spray_water_rot6d_pytorch` in `_CONFIGS`:

```python
    TrainConfig(
        name="pi05_wuji_spray_water_rot6d_subtask_flow_pytorch",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=58,
            action_horizon=16,
            max_token_len=320,
            subtask_loss_weight=0.15,
            fast_token_loss_weight=0.0,
            flow_matching_loss_weight=1.0,
        ),
        data=LeRobotWujiSubtaskDataConfig(
            repo_id="/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask",
            assets=AssetsConfig(asset_id="wuji_spray_water_rot6d_subtask"),
            base_config=DataConfig(prompt_from_task=False, lerobot_tolerance_s=0.08),
        ),
        pytorch_weight_path="./checkpoints/pi05_base_pytorch_converted",
        batch_size=16,
        num_train_steps=20_000,
        save_interval=1000,
    ),
    TrainConfig(
        name="pi05_wuji_spray_water_rot6d_subtask_fast_pytorch",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=58,
            action_horizon=16,
            max_token_len=384,
            subtask_loss_weight=10.0,
            fast_token_loss_weight=1.0,
            flow_matching_loss_weight=0.0,
        ),
        data=LeRobotWujiSubtaskDataConfig(
            repo_id="/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask",
            assets=AssetsConfig(asset_id="wuji_spray_water_rot6d_subtask"),
            base_config=DataConfig(prompt_from_task=False, lerobot_tolerance_s=0.08),
        ),
        pytorch_weight_path="./checkpoints/pi05_base_pytorch_converted",
        batch_size=16,
        num_train_steps=20_000,
        save_interval=1000,
    ),
    TrainConfig(
        name="pi05_wuji_spray_water_rot6d_action_expert_pytorch",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=58,
            action_horizon=16,
            max_token_len=320,
            subtask_loss_weight=0.0,
            fast_token_loss_weight=0.0,
            flow_matching_loss_weight=1.0,
            pytorch_freeze_filter="vlm_except_action_expert",
        ),
        data=LeRobotWujiSubtaskDataConfig(
            repo_id="/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask",
            assets=AssetsConfig(asset_id="wuji_spray_water_rot6d_subtask"),
            base_config=DataConfig(prompt_from_task=False, lerobot_tolerance_s=0.08),
        ),
        pytorch_weight_path="./checkpoints/pi05_base_pytorch_converted",
        batch_size=16,
        num_train_steps=8_000,
        save_interval=1000,
    ),
    TrainConfig(
        name="pi05_wuji_spray_water_rot6d_subtask_hybrid_pytorch",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=58,
            action_horizon=16,
            max_token_len=384,
            subtask_loss_weight=0.15,
            fast_token_loss_weight=0.15,
            flow_matching_loss_weight=1.0,
        ),
        data=LeRobotWujiSubtaskDataConfig(
            repo_id="/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask",
            assets=AssetsConfig(asset_id="wuji_spray_water_rot6d_subtask"),
            base_config=DataConfig(prompt_from_task=False, lerobot_tolerance_s=0.08),
        ),
        pytorch_weight_path="./checkpoints/pi05_base_pytorch_converted",
        batch_size=16,
        num_train_steps=40_000,
        save_interval=1000,
    ),
```

- [x] **Step 6: Run focused test and verify pass**

Run:

```bash
uv run pytest src/openpi/training/wuji_config_test.py::test_wuji_subtask_train_configs_exist_and_point_to_subtask_dataset -q
```

Expected: PASS.

---

### Task 7: Add PyTorch Token Losses And Hybrid Flow Masking

**Files:**
- Modify: `src/openpi/models_pytorch/pi0_pytorch.py`
- Test: add lightweight tests if possible in `src/openpi/models_pytorch/pi0_pytorch_test.py`

- [x] **Step 1: Write unit tests around small helper functions**

Add helper-oriented tests to avoid instantiating the full PaliGemma model:

```python
import torch

from openpi.models_pytorch import pi0_pytorch


def test_masked_token_cross_entropy_normalizes_per_sample():
    logits = torch.tensor([[[0.0, 2.0], [3.0, 0.0]], [[2.0, 0.0], [0.0, 3.0]]])
    targets = torch.tensor([[1, 0], [1, 1]])
    mask = torch.tensor([[True, True], [False, True]])

    loss = pi0_pytorch.masked_token_cross_entropy(logits, targets, mask)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    assert loss[1] > loss[0]


def test_mask_action_tokens_for_flow_removes_hybrid_fast_tokens():
    tokens = torch.tensor([[10, 11, 12, 13]])
    token_mask = torch.tensor([[True, True, True, True]])
    action_region_mask = torch.tensor([[False, False, True, True]])

    flow_tokens, flow_mask = pi0_pytorch.mask_action_tokens_for_flow(tokens, token_mask, action_region_mask)

    assert flow_tokens.tolist() == [[10, 11, 0, 0]]
    assert flow_mask.tolist() == [[True, True, False, False]]
```

- [x] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest src/openpi/models_pytorch/pi0_pytorch_test.py::test_masked_token_cross_entropy_normalizes_per_sample src/openpi/models_pytorch/pi0_pytorch_test.py::test_mask_action_tokens_for_flow_removes_hybrid_fast_tokens -q
```

Expected: FAIL because helper functions do not exist.

- [x] **Step 3: Add helper functions**

Add to `src/openpi/models_pytorch/pi0_pytorch.py` near `make_att_2d_masks`:

```python
def masked_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    losses = losses.reshape(targets.shape)
    mask = mask.to(dtype=losses.dtype, device=losses.device)
    denom = torch.clamp(mask.sum(dim=-1), min=1.0)
    return (losses * mask).sum(dim=-1) / denom


def mask_action_tokens_for_flow(
    tokenized_prompt: torch.Tensor,
    tokenized_prompt_mask: torch.Tensor,
    action_region_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_region_mask is None:
        return tokenized_prompt, tokenized_prompt_mask
    return (
        torch.where(action_region_mask, torch.zeros_like(tokenized_prompt), tokenized_prompt),
        torch.where(action_region_mask, torch.zeros_like(tokenized_prompt_mask, dtype=torch.bool), tokenized_prompt_mask),
    )
```

- [x] **Step 4: Refactor `PI0Pytorch.forward` into component losses**

Keep existing flow code behavior in a private method:

```python
    def compute_flow_loss(self, observation, actions, noise=None, time=None) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        ...
        return F.mse_loss(u_t, v_t, reduction="none")
```

Add a token loss method:

```python
    def compute_token_losses(self, observation) -> tuple[torch.Tensor, torch.Tensor]:
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        prefix_out = prefix_out[:, :-1].to(dtype=torch.float32)
        logits = self.paligemma_with_expert.paligemma.language_model.lm_head(prefix_out)
        targets = observation.tokenized_prompt[:, 1:].to(device=logits.device, dtype=torch.long)
        subtask_mask = observation.subtask_region_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
        action_mask = observation.action_region_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
        return (
            masked_token_cross_entropy(logits[:, -targets.shape[1] :], targets, subtask_mask),
            masked_token_cross_entropy(logits[:, -targets.shape[1] :], targets, action_mask),
        )
```

If the actual language-head attribute differs, inspect `src/openpi/models_pytorch/gemma_pytorch.py` and replace `language_model.lm_head` with the correct local projection call before implementation.

Update `forward`:

```python
    def forward(self, observation, actions, noise=None, time=None):
        metrics = {}
        total = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)

        if self.config.subtask_loss_weight > 0 or self.config.fast_token_loss_weight > 0:
            subtask_loss, fast_token_loss = self.compute_token_losses(observation)
        else:
            subtask_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
            fast_token_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
        total = total + self.config.subtask_loss_weight * subtask_loss
        total = total + self.config.fast_token_loss_weight * fast_token_loss

        if self.config.flow_matching_loss_weight > 0:
            flow_observation = observation
            if self.config.fast_token_loss_weight > 0 and getattr(observation, "action_region_mask", None) is not None:
                flow_tokens, flow_token_mask = mask_action_tokens_for_flow(
                    observation.tokenized_prompt,
                    observation.tokenized_prompt_mask,
                    observation.action_region_mask,
                )
                flow_observation = dataclasses.replace(
                    observation,
                    tokenized_prompt=flow_tokens,
                    tokenized_prompt_mask=flow_token_mask,
                )
            flow_loss = self.compute_flow_loss(flow_observation, actions, noise=noise, time=time).mean(dim=(-1, -2))
        else:
            flow_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
        total = total + self.config.flow_matching_loss_weight * flow_loss

        metrics["loss"] = total.mean()
        metrics["flow_loss"] = flow_loss.mean()
        metrics["subtask_loss"] = subtask_loss.mean()
        metrics["fast_token_loss"] = fast_token_loss.mean()
        return metrics
```

- [x] **Step 5: Run helper tests and existing WUJI tests**

Run:

```bash
uv run pytest src/openpi/models_pytorch/pi0_pytorch_test.py src/openpi/policies/wuji_policy_test.py src/openpi/training/wuji_config_test.py -q
```

Expected: PASS.

---

### Task 8: Add PyTorch Trainer Metrics And Freeze Helper

**Files:**
- Modify: `scripts/train_pytorch.py`
- Test: add helper tests if a scripts test pattern exists; otherwise verify with focused import snippets.

- [x] **Step 1: Add freeze helper tests**

Add near existing train script tests or create `scripts/train_pytorch_test.py`:

```python
import torch

import scripts.train_pytorch as train_pytorch


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


def test_apply_pytorch_freeze_filter_keeps_action_expert_trainable():
    model = _TinyModel()

    train_pytorch.apply_pytorch_freeze_filter(model, "vlm_except_action_expert")

    named = dict(model.named_parameters())
    assert named["paligemma_with_expert.paligemma.weight"].requires_grad is False
    assert named["paligemma_with_expert.gemma_expert.weight"].requires_grad is True
    assert named["action_in_proj.weight"].requires_grad is True
    assert named["action_out_proj.weight"].requires_grad is True
```

- [x] **Step 2: Run focused test and verify failure**

Run:

```bash
uv run pytest scripts/train_pytorch_test.py::test_apply_pytorch_freeze_filter_keeps_action_expert_trainable -q
```

Expected: FAIL because `apply_pytorch_freeze_filter` does not exist.

- [x] **Step 3: Add freeze and trainable parameter helpers**

Add to `scripts/train_pytorch.py`:

```python
def apply_pytorch_freeze_filter(model: torch.nn.Module, freeze_filter: str | None) -> None:
    if freeze_filter is None:
        return
    if freeze_filter != "vlm_except_action_expert":
        raise ValueError(f"Unsupported pytorch_freeze_filter={freeze_filter!r}")
    unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    trainable_prefixes = (
        "paligemma_with_expert.gemma_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_time_mlp_in",
        "action_time_mlp_out",
    )
    for name, param in unwrapped.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)


def trainable_parameters(model: torch.nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def log_trainable_parameter_counts(model: torch.nn.Module) -> None:
    trainable = 0
    frozen = 0
    for param in model.parameters():
        count = param.numel()
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
    logging.info("PyTorch parameters: trainable=%d frozen=%d", trainable, frozen)
```

- [x] **Step 4: Apply freeze before optimizer construction**

In `train_loop`, after optional base checkpoint load and before optimizer construction:

```python
    apply_pytorch_freeze_filter(
        model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
        getattr(model_cfg, "pytorch_freeze_filter", None),
    )
    log_trainable_parameter_counts(model)
```

Change optimizer construction:

```python
    optim = torch.optim.AdamW(
        trainable_parameters(model),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
```

- [x] **Step 5: Support dictionary loss metrics**

Replace the forward/loss block:

```python
            outputs = model(observation, actions)
            if isinstance(outputs, dict):
                loss = outputs["loss"]
                loss_metrics = {
                    name: value.detach().item()
                    for name, value in outputs.items()
                    if torch.is_tensor(value) and value.ndim == 0
                }
            else:
                losses = outputs
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)
                loss = losses.mean()
                loss_metrics = {"loss": loss.detach().item()}
```

Update `infos.append` to include:

```python
                        **loss_metrics,
```

Update log payload to use `train/` metric names:

```python
                    log_payload = {
                        "train/loss": avg_loss,
                        "train/learning_rate": avg_lr,
                        "step": global_step,
                        "train/time_per_step": elapsed / config.log_interval,
                    }
                    for key in ("flow_loss", "subtask_loss", "fast_token_loss"):
                        vals = [info[key] for info in infos if key in info]
                        if vals:
                            log_payload[f"train/{key}"] = sum(vals) / len(vals)
                    if avg_grad_norm is not None:
                        log_payload["train/grad_norm"] = avg_grad_norm
```

- [x] **Step 6: Run focused tests**

Run:

```bash
uv run pytest scripts/train_pytorch_test.py::test_apply_pytorch_freeze_filter_keeps_action_expert_trainable -q
```

Expected: PASS.

---

### Task 9: Verify Dataloader Instantiation For Current Dataset

**Files:**
- No code edits unless this verification reveals issues.

- [x] **Step 1: Run config and dataloader smoke script**

Run:

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
    loader = data_loader.create_data_loader(
        cfg,
        framework="pytorch",
        shuffle=False,
        num_batches=1,
        skip_norm_stats=True,
    )
    obs, actions = next(iter(loader))
    print(name, obs.state.shape, actions.shape)
PY
```

Expected: prints four config names with state shape ending in `58` and actions shape ending in `(16, 58)`.

- [x] **Step 2: Fix metadata loading if the smoke script exposes task/subtask mismatches**

If hardcoded metadata is insufficient, add a loader in `src/openpi/training/config.py`:

```python
def _load_jsonl_index_text(path: pathlib.Path) -> dict[int, str]:
    import json

    result = {}
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            result[int(item["task_index"] if "task_index" in item else item["subtask_index"])] = item["task"]
    return result
```

Then call it from `LeRobotWujiSubtaskDataConfig.create` when `repo_id` is a local path and `meta/tasks.jsonl` plus `meta/subtasks.jsonl` exist. Preserve the fallback `{0: ..., 1: ..., 2: ...}` subtask map only when metadata files are unavailable.

---

### Task 10: Add One-Batch Forward Smoke Tests

**Files:**
- Test: add a focused PyTorch smoke test if environment has dummy model support.

- [x] **Step 1: Create a lightweight forward smoke test**

If `paligemma_variant="dummy"` and `action_expert_variant="dummy"` work with `PI0Pytorch`, add:

```python
def test_pi0_pytorch_subtask_modes_return_finite_losses():
    import openpi.models.pi0_config as pi0_config
    from openpi.models import model as _model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    for subtask_weight, fast_weight, flow_weight in [(0.15, 0.0, 1.0), (10.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.15, 0.15, 1.0)]:
        cfg = pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=58,
            action_horizon=16,
            max_token_len=32,
            subtask_loss_weight=subtask_weight,
            fast_token_loss_weight=fast_weight,
            flow_matching_loss_weight=flow_weight,
            pytorch_compile_mode=None,
        )
        model = PI0Pytorch(cfg)
        batch = {
            "image": {key: torch.zeros((2, 224, 224, 3), dtype=torch.float32) for key in _model.IMAGE_KEYS},
            "image_mask": {key: torch.ones((2,), dtype=torch.bool) for key in _model.IMAGE_KEYS},
            "state": torch.zeros((2, 58), dtype=torch.float32),
            "tokenized_prompt": torch.ones((2, 32), dtype=torch.long),
            "tokenized_prompt_mask": torch.ones((2, 32), dtype=torch.bool),
            "subtask_region_mask": torch.zeros((2, 32), dtype=torch.bool),
            "action_region_mask": torch.zeros((2, 32), dtype=torch.bool),
        }
        batch["subtask_region_mask"][:, 4:8] = True
        batch["action_region_mask"][:, 8:12] = fast_weight > 0
        observation = _model.Observation.from_dict(batch)
        actions = torch.zeros((2, 16, 58), dtype=torch.float32)

        outputs = model(observation, actions)

        assert torch.isfinite(outputs["loss"])
        assert set(outputs) >= {"loss", "flow_loss", "subtask_loss", "fast_token_loss"}
```

- [x] **Step 2: Run the smoke test**

Run:

```bash
uv run pytest src/openpi/models_pytorch/pi0_pytorch_test.py::test_pi0_pytorch_subtask_modes_return_finite_losses -q
```

Expected: PASS. If the dummy model path is not viable, document the blocker in the test file and rely on the real two-step smoke runs in Task 11.

---

### Task 11: Full Verification

**Files:**
- No code edits unless verification fails.

- [x] **Step 1: Run required unit tests**

Run:

```bash
uv run pytest src/openpi/policies/wuji_policy_test.py src/openpi/training/wuji_config_test.py src/openpi/transforms_test.py
```

Expected: PASS.

- [x] **Step 2: Run model/data unit tests touched by this work**

Run:

```bash
uv run pytest src/openpi/models/model_test.py src/openpi/models_pytorch/pi0_pytorch_test.py src/openpi/training/data_loader_test.py
```

Expected: PASS.

- [x] **Step 3: Run one-batch dataloader verification**

Run the script from Task 9.

Expected: four printed rows, one per subtask config.

- [x] **Step 4: Run two-step training smoke for subtask+flow**

Run:

```bash
uv run scripts/train_pytorch.py pi05_wuji_spray_water_rot6d_subtask_flow_pytorch \
  --exp-name=smoke_subtask_flow \
  --num-train-steps=2 \
  --save-interval=1 \
  --overwrite \
  --wandb-enabled=false
```

Expected: exits 0 and logs finite `train/loss` plus `train/flow_loss` and `train/subtask_loss`.

- [x] **Step 5: Repeat two-step smoke for the other subtask configs**

Run:

```bash
uv run scripts/train_pytorch.py pi05_wuji_spray_water_rot6d_subtask_fast_pytorch \
  --exp-name=smoke_subtask_fast \
  --num-train-steps=2 \
  --save-interval=1 \
  --overwrite \
  --wandb-enabled=false
```

```bash
uv run scripts/train_pytorch.py pi05_wuji_spray_water_rot6d_action_expert_pytorch \
  --exp-name=smoke_action_expert \
  --num-train-steps=2 \
  --save-interval=1 \
  --overwrite \
  --wandb-enabled=false
```

```bash
uv run scripts/train_pytorch.py pi05_wuji_spray_water_rot6d_subtask_hybrid_pytorch \
  --exp-name=smoke_subtask_hybrid \
  --num-train-steps=2 \
  --save-interval=1 \
  --overwrite \
  --wandb-enabled=false
```

Expected: each exits 0. FAST modes may fail with a clear FAST tokenizer availability error if `physical-intelligence/fast` is not available locally; if that happens, record it as an environment dependency rather than silently skipping the feature.

---

## Self-Review Checklist

- Spec coverage:
  - Four WUJI subtask configs are included in Task 6.
  - Existing non-subtask WUJI config default behavior is protected in Task 1 and existing tests.
  - Subtask data mapping, unlabeled filtering, and future fully labeled compatibility are covered in Tasks 3, 4, 6, and 9.
  - High/low prompt tokenization and masks are covered in Task 5.
  - Optional observation region masks are covered in Task 2.
  - PyTorch component losses, hybrid FAST-token masking, and dictionary metrics are covered in Tasks 7 and 8.
  - Action expert VLM freezing and trainable-parameter optimizer filtering are covered in Task 8.
  - Required verification commands are covered in Task 11.
- Placeholder scan: no placeholder markers or unspecified test commands remain.
- Type consistency: config field names match the spec and are used consistently: `subtask_loss_weight`, `fast_token_loss_weight`, `flow_matching_loss_weight`, `fast_tokenizer_path`, `stop_gradient_flow_to_prefix`, and `pytorch_freeze_filter`.


---

## Execution Notes / Verification Evidence

- Task 9 Step 1 dataloader smoke passed after disabling spawned PyTorch DataLoader workers for `python -` / heredoc scripts; output showed all four subtask configs with `torch.Size([16, 58])` state and `torch.Size([16, 16, 58])` actions.
- Task 9 Step 2 metadata-loading fallback was not needed: the smoke script resolved the configured local subtask dataset and printed all four rows without task/subtask mismatch.
- Task 10 dummy one-batch forward smoke is present but intentionally skipped with an in-test blocker explanation: the dummy HF PaliGemma path has incompatible dummy/image token dimensions. The real executable forward paths are covered by Task 11 two-step training smokes.
- Task 11 Step 1 passed: `uv run pytest src/openpi/policies/wuji_policy_test.py src/openpi/training/wuji_config_test.py src/openpi/transforms_test.py` (also included in the grouped 47-test run).
- Task 11 Step 2 passed: `uv run pytest src/openpi/models/model_test.py src/openpi/models_pytorch/pi0_pytorch_test.py src/openpi/training/data_loader_test.py -q` -> `27 passed, 1 skipped`.
- Task 11 Step 3 passed: one-batch dataloader verification printed four rows for the flow, FAST, action-expert, and hybrid subtask configs.
- Task 11 Step 4 passed: `smoke_subtask_flow` produced checkpoint steps `1` and `2` with `metadata.pt`, `model.safetensors`, and `optimizer.pt`.
- Task 11 Step 5 passed: `smoke_subtask_fast`, `smoke_action_expert`, and `smoke_subtask_hybrid` each produced checkpoint steps `1` and `2` with `metadata.pt`, `model.safetensors`, and `optimizer.pt`. Hybrid was rerun successfully after GPU memory became available and after fixing trainer loss lifetime / AdamW peak-memory behavior.
- Additional verification passed:
  - `uv run pytest scripts/train_pytorch_test.py -q` -> `4 passed`.
  - `uv run pytest src/openpi/training/data_loader_test.py scripts/train_pytorch_test.py -q` -> `14 passed`.
  - `uv run pytest src/openpi/training/wuji_config_test.py::test_wuji_subtask_train_configs_exist_and_point_to_subtask_dataset -q` -> `1 passed`.
- Placeholder scan found only pre-existing abstract/interface `NotImplementedError`s, unrelated existing TODOs, and the documented Task 10 `pytest.skip` blocker.
