from openpi import transforms
from openpi.training import config as _config


def test_pi05_wuji_config_instantiates_with_54d_joint_action_space(monkeypatch):
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())

    config = _config.get_config("pi05_wuji_tactile_vla_joint_absolute_pytorch")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.action_dim == 54
    assert config.model.action_horizon == 32
    assert config.model.max_token_len == 280
    assert config.pytorch_weight_path == "./checkpoints/pi05_base_pytorch_converted"
    assert config.batch_size == 32
    assert config.num_workers == 4
    assert config.num_train_steps == 60_000
    assert config.lr_schedule.warmup_steps == 3_000
    assert config.lr_schedule.peak_lr == 2.5e-5
    assert config.lr_schedule.decay_lr == 2.5e-6
    assert data_config.repo_id == "/data_all/liyunhao/openpi/data/tactile_vla_teleop_test_joint_absolute_merged"
    assert data_config.asset_id == "wuji_tactile_vla_joint_absolute"
    assert data_config.prompt_from_task is True
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.use_quantile_norm is True
    assert data_config.lerobot_tolerance_s == 0.04
    assert len(data_config.data_transforms.inputs) == 1
    assert data_config.data_transforms.inputs[0].action_dim == 54
    assert not any(isinstance(transform, transforms.DeltaActions) for transform in data_config.data_transforms.inputs)


def test_wuji_repack_maps_lerobot_keys_to_wuji_inputs(monkeypatch):
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())

    config = _config.get_config("pi05_wuji_tactile_vla_joint_absolute_pytorch")
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
