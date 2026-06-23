import dataclasses
import pathlib

import numpy as np
import pytest

from openpi.training import config as _config
import openpi.transforms as _transforms

from . import open_loop_eval


class FakeDataset:
    def __init__(self, *, length: int, action_horizon: int):
        self.episode_data_index = {
            "from": np.asarray([0]),
            "to": np.asarray([length]),
        }
        self._length = length
        self._action_horizon = action_horizon

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        action_steps = [min(index + offset, self._length - 1) for offset in range(self._action_horizon)]
        return {
            "state": np.asarray([index, index + 0.5], dtype=np.float32),
            "actions": np.asarray([[step, -step] for step in action_steps], dtype=np.float32),
        }


class OffsetPolicy:
    def __init__(self, *, action_horizon: int):
        self.action_horizon = action_horizon
        self.calls = []

    def infer(self, obs):
        start = int(obs["state"][0])
        self.calls.append(start)
        return {
            "actions": np.asarray(
                [[start + offset + 1, -(start + offset) + 1] for offset in range(self.action_horizon)],
                dtype=np.float32,
            )
        }


def test_evaluate_single_trajectory_chunks_predictions_and_computes_metrics():
    action_horizon = 2
    dataset = FakeDataset(length=5, action_horizon=action_horizon)
    policy = OffsetPolicy(action_horizon=action_horizon)
    data_config = _config.DataConfig(
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform({"state": "state", "actions": "actions"})]
        )
    )

    result = open_loop_eval.evaluate_single_trajectory(
        policy=policy,
        dataset=dataset,
        data_config=data_config,
        traj_id=0,
        steps=5,
        action_horizon=action_horizon,
        save_plot_path=None,
        plot=False,
    )

    assert policy.calls == [0, 2, 4]
    assert result.actual_steps == 5
    np.testing.assert_allclose(result.mse, 1.0)
    np.testing.assert_allclose(result.mae, 1.0)
    assert result.gt_actions.shape == (5, 2)
    assert result.pred_actions.shape == (5, 2)


def test_resolve_plot_path_uses_directory_for_multiple_trajectories(tmp_path: pathlib.Path):
    args = open_loop_eval.Args(
        config_name="debug",
        traj_ids=[0, 1],
        save_plot_path=str(tmp_path / "plots"),
    )

    assert open_loop_eval.resolve_plot_path(args, traj_id=1) == tmp_path / "plots" / "traj_1.jpeg"


def test_resolve_plot_path_preserves_file_path_for_single_trajectory(tmp_path: pathlib.Path):
    path = tmp_path / "plot.jpeg"
    args = dataclasses.replace(
        open_loop_eval.Args(config_name="debug"),
        traj_ids=[0],
        save_plot_path=str(path),
    )

    assert open_loop_eval.resolve_plot_path(args, traj_id=0) == path


def test_create_data_config_overrides_dataset_repo_id():
    args = open_loop_eval.Args(config_name="unused", dataset_repo_id="/tmp/eval_dataset")
    train_config = _config.TrainConfig(
        name="test",
        data=_config.LeRobotAlohaDataConfig(
            repo_id="/tmp/train_dataset",
            assets=_config.AssetsConfig(asset_id="trossen"),
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
    )

    data_config = open_loop_eval.create_data_config(args, train_config)

    assert data_config.repo_id == "/tmp/eval_dataset"
    assert data_config.asset_id == "trossen"
    assert data_config.prompt_from_task is True


def test_create_lerobot_dataset_rejects_local_path_without_metadata(tmp_path: pathlib.Path):
    incomplete_dataset = tmp_path / "filtered_out"
    (incomplete_dataset / "data").mkdir(parents=True)
    data_config = _config.DataConfig(repo_id=str(incomplete_dataset))

    with pytest.raises(ValueError, match="meta/info.json"):
        open_loop_eval.create_lerobot_dataset(data_config, action_horizon=16)
