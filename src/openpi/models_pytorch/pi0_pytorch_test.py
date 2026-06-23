import dataclasses
import types

import pytest
import torch

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch


@dataclasses.dataclass
class _Observation:
    images: dict
    image_masks: dict
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
    subtask_region_mask: torch.Tensor | None = None
    action_region_mask: torch.Tensor | None = None

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class _ForwardHarness(torch.nn.Module):
    forward = pi0_pytorch.PI0Pytorch.forward

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flow_observation = None

    def compute_token_losses(self, observation):
        batch_size = observation.state.shape[0]
        return (
            torch.full((batch_size,), 2.0, device=observation.state.device),
            torch.full((batch_size,), 3.0, device=observation.state.device),
        )

    def compute_flow_loss(self, observation, actions, noise=None, time=None):
        self.flow_observation = observation
        return torch.ones_like(actions)


class _FakeLmHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden):
        token_id = 5 if self.calls == 0 else 1
        self.calls += 1
        logits = torch.full((*hidden.shape[:-1], 8), -100.0, device=hidden.device)
        logits[..., token_id] = 100.0
        return logits


class _RecordingLmHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, hidden):
        self.input_shape = tuple(hidden.shape)
        logits = torch.zeros((hidden.shape[0], 3), device=hidden.device)
        logits[:, 1] = 3.0
        return logits


class _FakePaligemmaWithExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_mask_shapes = []
        q_proj = types.SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
        self.paligemma = types.SimpleNamespace(
            language_model=types.SimpleNamespace(
                layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(q_proj=q_proj))],
                config=types.SimpleNamespace(_attn_implementation=None),
            ),
            lm_head=_FakeLmHead(),
        )

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        del position_ids, use_cache, adarms_cond
        self.attention_mask_shapes.append(tuple(attention_mask.shape))
        hidden = torch.ones((*inputs_embeds[0].shape[:-1], 4), device=inputs_embeds[0].device)
        past_len = 0 if past_key_values is None else past_key_values["len"]
        return (hidden, None), {"len": past_len + inputs_embeds[0].shape[1]}

    def embed_language_tokens(self, tokens):
        return torch.ones((*tokens.shape, 4), device=tokens.device)


class _SampleLowHarness(torch.nn.Module):
    sample_low_level_task = pi0_pytorch.PI0Pytorch.sample_low_level_task
    _prepare_attention_masks_4d = pi0_pytorch.PI0Pytorch._prepare_attention_masks_4d

    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _FakePaligemmaWithExpert()

    def _preprocess_observation(self, observation, *, train=True):
        return [], [], observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        del images, img_masks, lang_tokens, lang_masks
        return (
            torch.zeros((1, 3, 4)),
            torch.tensor([[True, True, False]]),
            torch.tensor([[False, False, False]]),
        )


class _ActionsWithSubtaskHarness(torch.nn.Module):
    sample_actions_with_subtask = pi0_pytorch.PI0Pytorch.sample_actions_with_subtask

    def __init__(self):
        super().__init__()
        self.config = pi0_config.Pi0Config(action_dim=58, action_horizon=16)
        self.calls = []

    def _preprocess_observation(self, observation, *, train=True):
        return [], [], observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state

    def sample_low_level_task(self, device, observation, **kwargs):
        del device, observation, kwargs
        return torch.tensor([[5, 1]]), {"cache": True}, torch.tensor([[True, True, True]]), torch.tensor([[0, 1, 1]])

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        self.calls.append((state, prefix_pad_masks, past_key_values, timestep))
        return torch.ones_like(x_t)


def _make_observation():
    tokenized_prompt = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    tokenized_prompt_mask = torch.ones_like(tokenized_prompt, dtype=torch.bool)
    action_region_mask = torch.tensor([[False, False, True, True], [False, True, True, False]])
    return _Observation(
        images={},
        image_masks={},
        state=torch.zeros((2, 58)),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        action_region_mask=action_region_mask,
    )


def test_masked_token_cross_entropy_normalizes_per_sample():
    logits = torch.tensor([[[0.0, 2.0], [3.0, 0.0]], [[2.0, 0.0], [0.0, 3.0]]])
    targets = torch.tensor([[1, 0], [1, 1]])
    mask = torch.tensor([[True, True], [False, True]])

    loss = pi0_pytorch.masked_token_cross_entropy(logits, targets, mask)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    assert loss[0] > loss[1]


def test_masked_lm_head_cross_entropy_projects_only_masked_tokens():
    lm_head = _RecordingLmHead()
    hidden = torch.randn((2, 4, 5))
    targets = torch.ones((2, 4), dtype=torch.long)
    mask = torch.tensor([[True, False, False, True], [False, True, False, False]])

    loss = pi0_pytorch.masked_lm_head_cross_entropy(lm_head, hidden, targets, mask)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    assert lm_head.input_shape == (3, 5)


def test_mask_action_tokens_for_flow_removes_hybrid_fast_tokens():
    tokens = torch.tensor([[10, 11, 12, 13]])
    token_mask = torch.tensor([[True, True, True, True]])
    action_region_mask = torch.tensor([[False, False, True, True]])

    flow_tokens, flow_mask = pi0_pytorch.mask_action_tokens_for_flow(tokens, token_mask, action_region_mask)

    assert flow_tokens.tolist() == [[10, 11, 0, 0]]
    assert flow_mask.tolist() == [[True, True, False, False]]


@pytest.mark.parametrize(
    ("config", "expected_loss"),
    [
        (pi0_config.Pi0Config(subtask_loss_weight=0.15, fast_token_loss_weight=0.0, flow_matching_loss_weight=1.0), 1.3),
        (pi0_config.Pi0Config(subtask_loss_weight=10.0, fast_token_loss_weight=1.0, flow_matching_loss_weight=0.0), 23.0),
        (pi0_config.Pi0Config(subtask_loss_weight=0.0, fast_token_loss_weight=0.0, flow_matching_loss_weight=1.0), 1.0),
        (pi0_config.Pi0Config(subtask_loss_weight=0.15, fast_token_loss_weight=0.15, flow_matching_loss_weight=1.0), 1.75),
    ],
)
def test_forward_returns_finite_weighted_subtask_losses(config, expected_loss):
    model = _ForwardHarness(config)
    observation = _make_observation()
    actions = torch.zeros((2, 16, 58))

    outputs = model(observation, actions)

    assert set(outputs) == {"loss", "flow_loss", "subtask_loss", "fast_token_loss"}
    assert all(torch.isfinite(value) for value in outputs.values())
    assert torch.isclose(outputs["loss"], torch.tensor(expected_loss))


def test_forward_masks_hybrid_fast_tokens_before_flow():
    config = pi0_config.Pi0Config(subtask_loss_weight=0.15, fast_token_loss_weight=0.15, flow_matching_loss_weight=1.0)
    model = _ForwardHarness(config)
    observation = _make_observation()

    model(observation, torch.zeros((2, 16, 58)))

    assert model.flow_observation.tokenized_prompt.tolist() == [[10, 11, 0, 0], [20, 0, 0, 23]]
    assert model.flow_observation.tokenized_prompt_mask.tolist() == [
        [True, True, False, False],
        [True, False, False, True],
    ]


def test_sample_low_level_task_decodes_until_eos_and_returns_prefix_masks():
    model = _SampleLowHarness()
    observation = _make_observation()
    observation.state = observation.state[:1]
    observation.tokenized_prompt = observation.tokenized_prompt[:1]
    observation.tokenized_prompt_mask = observation.tokenized_prompt_mask[:1]

    output_tokens, past_key_values, prefix_mask, prefix_ar_mask = model.sample_low_level_task(
        torch.device("cpu"),
        observation,
        max_decoding_steps=4,
        paligemma_eos_token=1,
    )

    assert output_tokens.tolist() == [[5, 1]]
    assert past_key_values == {"len": 5}
    assert prefix_mask.tolist() == [[True, True, False, True, True]]
    assert prefix_ar_mask.tolist() == [[False, False, False, True, True]]
    assert model.paligemma_with_expert.attention_mask_shapes == [(1, 1, 3, 3), (1, 1, 1, 4), (1, 1, 1, 5)]


def test_sample_actions_with_subtask_uses_generated_prefix_for_denoising():
    model = _ActionsWithSubtaskHarness()
    observation = _make_observation()
    observation.state = observation.state[:1]
    observation.tokenized_prompt = observation.tokenized_prompt[:1]
    observation.tokenized_prompt_mask = observation.tokenized_prompt_mask[:1]
    noise = torch.zeros((1, 16, 58))

    actions = model.sample_actions_with_subtask(torch.device("cpu"), observation, noise=noise, num_steps=1)

    assert actions.tolist() == torch.full((1, 16, 58), -1.0).tolist()
    assert len(model.calls) == 1
    _, prefix_pad_masks, past_key_values, _ = model.calls[0]
    assert prefix_pad_masks.tolist() == [[True, True, True]]
    assert past_key_values == {"cache": True}


def test_pi0_pytorch_subtask_modes_return_finite_losses():
    pytest.skip(
        "PI0Pytorch dummy PaliGemma is not viable for an end-to-end image forward: "
        "the HF PaliGemma vision tower still projects image tokens to 2048 dims while "
        "dummy language tokens are 64 dims. Real two-step training smoke tests cover "
        "the executable forward path."
    )
    from openpi.models import model as _model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    for subtask_weight, fast_weight, flow_weight in [
        (0.15, 0.0, 1.0),
        (10.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.15, 0.15, 1.0),
    ]:
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
            "image": {key: torch.zeros((2, 3, 224, 224), dtype=torch.float32) for key in _model.IMAGE_KEYS},
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
