import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


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


class _ColumnBackedSubtaskDataset(_TinySubtaskDataset):
    def __init__(self):
        super().__init__()
        self.hf_dataset = {"subtask_index": [item["subtask_index"] for item in self.items]}
        self.init_getitem_calls = 0

    def __getitem__(self, index):
        self.init_getitem_calls += 1
        return super().__getitem__(index)


def test_filter_unlabeled_subtask_dataset_uses_hf_column_without_loading_items():
    source = _ColumnBackedSubtaskDataset()

    dataset = _data_loader.FilterUnlabeledSubtaskDataset(source)

    assert len(dataset) == 2
    assert source.init_getitem_calls == 0
    assert dataset[0]["value"] == "keep0"


def test_effective_num_workers_disables_spawn_for_stdin_main(monkeypatch):
    import __main__

    monkeypatch.setattr(__main__, "__file__", "<stdin>", raising=False)

    assert _data_loader._effective_num_workers(2) == 0


def test_effective_num_workers_preserves_file_backed_main(monkeypatch):
    import __main__

    monkeypatch.setattr(__main__, "__file__", __file__, raising=False)

    assert _data_loader._effective_num_workers(2) == 2


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_torch_dataset_passes_lerobot_tolerance(monkeypatch):
    captured = {}

    class FakeMetadata:
        fps = 30

    class FakeLeRobotDataset:
        def __init__(self, repo_id, **kwargs):
            captured["repo_id"] = repo_id
            captured.update(kwargs)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda repo_id: FakeMetadata())
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeLeRobotDataset)

    data_config = _config.DataConfig(repo_id="local/wuji", action_sequence_keys=("action",), lerobot_tolerance_s=0.015)

    dataset = _data_loader.create_torch_dataset(data_config, action_horizon=16, model_config=pi0_config.Pi0Config())

    assert isinstance(dataset, FakeLeRobotDataset)
    assert captured["repo_id"] == "local/wuji"
    assert captured["tolerance_s"] == 0.015


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
