from openpi import transforms
from openpi.training import config as _config


def test_pi05_wuji_config_instantiates_with_58d_action_space(monkeypatch):
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())

    config = _config.get_config("pi05_wuji_spray_water_rot6d_pytorch")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.action_dim == 58
    assert config.model.action_horizon == 16
    assert config.model.max_token_len == 280
    assert config.pytorch_weight_path == "./checkpoints/pi05_base_pytorch_converted"
    assert config.batch_size == 16
    assert config.num_train_steps == 20_000
    assert data_config.repo_id == "/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter"
    assert data_config.asset_id == "wuji_spray_water_rot6d"
    assert data_config.prompt_from_task is True
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.use_quantile_norm is True
    assert data_config.lerobot_tolerance_s == 0.08


def test_pi0_config_subtask_defaults_preserve_existing_behavior():
    config = _config.get_config("pi05_wuji_spray_water_rot6d_pytorch")

    assert config.model.subtask_loss_weight == 0.0
    assert config.model.fast_token_loss_weight == 0.0
    assert config.model.flow_matching_loss_weight == 1.0
    assert config.model.fast_tokenizer_path == "physical-intelligence/fast"
    assert config.model.stop_gradient_flow_to_prefix is False
    assert config.model.pytorch_freeze_filter is None


def test_wuji_repack_maps_lerobot_keys_to_wuji_inputs(monkeypatch):
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())

    config = _config.get_config("pi05_wuji_spray_water_rot6d_pytorch")
    data_config = config.data.create(config.assets_dirs, config.model)

    repacked = data_config.repack_transforms.inputs[0](
        {
            "observation.images.head_view": "head",
            "observation.images.left_wrist_view": "left",
            "observation.images.right_wrist_view": "right",
            "observation.state": "state",
            "action": "actions",
            "prompt": "prompt",
        }
    )

    assert repacked == {
        "images": {
            "head_view": "head",
            "left_wrist_view": "left",
            "right_wrist_view": "right",
        },
        "state": "state",
        "actions": "actions",
        "prompt": "prompt",
    }


def test_wuji_subtask_train_configs_exist_and_point_to_subtask_dataset(monkeypatch):
    class _NoopSubtaskModelTransformFactory:
        def __call__(self, model_config):
            return transforms.Group()

    monkeypatch.setattr(_config, "SubtaskModelTransformFactory", _NoopSubtaskModelTransformFactory, raising=False)

    expected = {
        "pi05_wuji_spray_water_rot6d_subtask_flow_pytorch": (0.15, 0.0, 1.0, 320, 20_000, None),
        "pi05_wuji_spray_water_rot6d_subtask_fast_pytorch": (10.0, 1.0, 0.0, 384, 20_000, None),
        "pi05_wuji_spray_water_rot6d_action_expert_pytorch": (0.0, 0.0, 1.0, 320, 8_000, "vlm_except_action_expert"),
        "pi05_wuji_spray_water_rot6d_subtask_hybrid_pytorch": (0.15, 0.15, 1.0, 384, 40_000, None),
    }

    expected_repo_id = "/data_all/liyunhao/openpi/data/spray_water_rot6d_rosbag_ts_filter_subtask"

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
        assert data_config.repo_id == expected_repo_id
        assert data_config.asset_id == "wuji_spray_water_rot6d_subtask"
        assert data_config.prompt_from_task is False
        assert data_config.filter_unlabeled_subtasks is True
        assert data_config.lerobot_tolerance_s == 0.08


def test_wuji_subtask_config_reuses_base_wuji_norm_stats(monkeypatch):
    class _NoopSubtaskModelTransformFactory:
        def __call__(self, model_config):
            return transforms.Group()

    monkeypatch.setattr(_config, "SubtaskModelTransformFactory", _NoopSubtaskModelTransformFactory, raising=False)

    config = _config.get_config("pi05_wuji_spray_water_rot6d_subtask_flow_pytorch")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert data_config.norm_stats is not None
    assert "state" in data_config.norm_stats
    assert "actions" in data_config.norm_stats
