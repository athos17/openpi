from flax import nnx
import jax
import numpy as np
import pytest
import torch

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.models_pytorch import preprocessing_pytorch
from openpi.shared import download
from openpi.shared import nnx_utils


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


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
