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
