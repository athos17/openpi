import dataclasses
import pathlib

from . import open_loop_eval_subtask


def test_resolve_video_path_uses_directory_for_multiple_trajectories(tmp_path: pathlib.Path):
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        traj_ids=[0, 1],
        save_video_path=str(tmp_path / "videos"),
    )

    assert open_loop_eval_subtask.resolve_video_path(args, traj_id=1) == tmp_path / "videos" / "traj_1.mp4"


def test_resolve_video_path_preserves_file_path_for_single_trajectory(tmp_path: pathlib.Path):
    path = tmp_path / "eval.mp4"
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        traj_ids=[0],
        save_video_path=str(path),
    )

    assert open_loop_eval_subtask.resolve_video_path(args, traj_id=0) == path


def test_resolve_plot_path_returns_none_in_fast_mode(tmp_path: pathlib.Path):
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        mode="fast",
        save_plot_path=str(tmp_path / "plots"),
    )

    assert open_loop_eval_subtask.resolve_plot_path(args, traj_id=0) is None


def test_resolve_plot_path_delegates_to_original_open_loop_in_full_mode(tmp_path: pathlib.Path):
    path = tmp_path / "plot.jpeg"
    args = open_loop_eval_subtask.Args(
        config_name="debug",
        checkpoint_dir="/tmp/checkpoint",
        mode="full",
        traj_ids=[0],
        save_plot_path=str(path),
    )

    assert open_loop_eval_subtask.resolve_plot_path(args, traj_id=0) == path


def test_checkpoint_dir_is_required_for_local_subtask_eval():
    args = open_loop_eval_subtask.Args(config_name="debug", checkpoint_dir=None)

    try:
        open_loop_eval_subtask.validate_args(args)
    except ValueError as exc:
        assert "checkpoint_dir" in str(exc)
    else:
        raise AssertionError("validate_args should reject missing checkpoint_dir")

import numpy as np


class FrameDataset:
    def __init__(self):
        self.items = [
            {"observation.images.head_view": np.zeros((2, 3, 3), dtype=np.uint8)},
            {"observation.images.head_view": np.ones((2, 3, 3), dtype=np.float32)},
        ]

    def __getitem__(self, index):
        return self.items[index]


def test_expand_subtask_schedule_repeats_chunk_text_until_next_inference_point():
    schedule = open_loop_eval_subtask.expand_subtask_schedule(
        actual_steps=5,
        action_horizon=2,
        chunk_subtasks=["pick up bottle", "pump", "spray"],
    )

    assert schedule == ["pick up bottle", "pick up bottle", "pump", "pump", "spray"]


def test_extract_main_frame_supports_raw_dataset_key_and_float_images():
    dataset = FrameDataset()

    frame0 = open_loop_eval_subtask.extract_main_frame(dataset[0], "observation.images.head_view")
    frame1 = open_loop_eval_subtask.extract_main_frame(dataset[1], "observation.images.head_view")

    assert frame0.dtype == np.uint8
    assert frame0.shape == (2, 3, 3)
    assert frame1.dtype == np.uint8
    assert frame1.max() == 255


def test_overlay_subtask_text_changes_pixels_in_top_right_region():
    frame = np.zeros((80, 160, 3), dtype=np.uint8)

    annotated = open_loop_eval_subtask.overlay_subtask_text(frame, "pump")

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated[:40, 80:] != frame[:40, 80:])
    assert np.all(frame == 0), "overlay_subtask_text must not mutate the input frame"

import torch


class FakeLocalSubtaskPolicy:
    def __init__(self):
        self.calls = []

    def infer_subtask_then_actions(
        self,
        repacked,
        *,
        action_horizon,
        num_steps,
        max_subtask_decoding_steps,
        subtask_temperature,
    ):
        start = int(repacked["state"][0])
        self.calls.append((start, action_horizon, num_steps, max_subtask_decoding_steps, subtask_temperature))
        return open_loop_eval_subtask.SubtaskInferenceResult(
            actions=np.asarray([[start + offset, -(start + offset)] for offset in range(action_horizon)], dtype=np.float32),
            subtask=f"subtask-{start}",
            subtask_tokens=np.asarray([start, start + 1], dtype=np.int64),
        )


def test_fake_subtask_policy_adapter_returns_action_chunk_and_subtask():
    policy = FakeLocalSubtaskPolicy()

    result = policy.infer_subtask_then_actions(
        {"state": np.asarray([3.0, 0.0], dtype=np.float32)},
        action_horizon=2,
        num_steps=4,
        max_subtask_decoding_steps=5,
        subtask_temperature=0.0,
    )

    assert result.subtask == "subtask-3"
    assert result.subtask_tokens.tolist() == [3, 4]
    np.testing.assert_allclose(result.actions, np.asarray([[3, -3], [4, -4]], dtype=np.float32))
    assert policy.calls == [(3, 2, 4, 5, 0.0)]


def test_decode_subtask_tokens_strips_eos_when_tokenizer_has_decode():
    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=True):
            assert skip_special_tokens is True
            return " pump\n"

    text = open_loop_eval_subtask.decode_subtask_tokens(Tokenizer(), torch.tensor([[7, 1, 0]]))

    assert text == "pump"

from openpi.training import config as _config
import openpi.transforms as _transforms


class FakeSubtaskDataset:
    def __init__(self, *, length: int, action_horizon: int):
        self.episode_data_index = {"from": np.asarray([0]), "to": np.asarray([length])}
        self._length = length
        self._action_horizon = action_horizon

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        action_steps = [min(index + offset, self._length - 1) for offset in range(self._action_horizon)]
        return {
            "state": np.asarray([index, index + 0.5], dtype=np.float32),
            "actions": np.asarray([[step, -step] for step in action_steps], dtype=np.float32),
            "observation.images.head_view": np.full((4, 5, 3), index, dtype=np.uint8),
        }


def test_evaluate_single_trajectory_records_subtasks_and_metrics_without_plot_or_video():
    action_horizon = 2
    dataset = FakeSubtaskDataset(length=5, action_horizon=action_horizon)
    policy = FakeLocalSubtaskPolicy()
    data_config = _config.DataConfig(
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform({"state": "state", "actions": "actions"})]
        )
    )

    result = open_loop_eval_subtask.evaluate_single_trajectory(
        policy=policy,
        dataset=dataset,
        data_config=data_config,
        traj_id=0,
        steps=5,
        action_horizon=action_horizon,
        num_steps=4,
        max_subtask_decoding_steps=5,
        subtask_temperature=0.0,
        save_video_path=None,
        save_plot_path=None,
        main_camera_key="observation.images.head_view",
        video_fps=30,
        plot_state=True,
    )

    assert result.actual_steps == 5
    assert result.chunk_subtasks == ["subtask-0", "subtask-2", "subtask-4"]
    assert result.frame_subtasks == ["subtask-0", "subtask-0", "subtask-2", "subtask-2", "subtask-4"]
    np.testing.assert_allclose(result.mse, 0.0)
    np.testing.assert_allclose(result.mae, 0.0)
    assert policy.calls == [(0, 2, 4, 5, 0.0), (2, 2, 4, 5, 0.0), (4, 2, 4, 5, 0.0)]


def test_write_annotated_video_creates_mp4(tmp_path: pathlib.Path):
    dataset = FakeSubtaskDataset(length=3, action_horizon=1)
    path = tmp_path / "traj_0.mp4"

    open_loop_eval_subtask.write_annotated_video(
        dataset=dataset,
        start=0,
        actual_steps=3,
        frame_subtasks=["pick", "pump", "spray"],
        save_video_path=path,
        main_camera_key="observation.images.head_view",
        fps=10,
    )

    assert path.exists()
    assert path.stat().st_size > 0


class DatasetWithMetaFps:
    class Meta:
        fps = 17

    meta = Meta()


def test_resolve_video_fps_prefers_args_then_dataset_meta():
    assert open_loop_eval_subtask.resolve_video_fps(24, DatasetWithMetaFps()) == 24
    assert open_loop_eval_subtask.resolve_video_fps(None, DatasetWithMetaFps()) == 17
    assert open_loop_eval_subtask.resolve_video_fps(None, object()) == 30


def test_prepare_subtask_generation_input_uses_empty_low_prompt_instead_of_gt_subtask():
    class PromptFromIndices:
        tasks = {0: "spray water"}
        subtasks = {2: "spray"}

        def __call__(self, data):
            return {**data, "high_prompt": self.tasks[int(data["task_index"])], "low_prompt": self.subtasks[int(data["subtask_index"])]}

    class SubtaskInputs:
        def __call__(self, data):
            assert data["low_prompt"] == ""
            return {"high_prompt": data["high_prompt"], "low_prompt": data["low_prompt"], "state": data["state"]}

    class Tokenizer:
        def tokenize_high_low_prompt_infer(self, high_prompt, state):
            assert high_prompt == "spray water"
            return (
                np.asarray([1, 2, 3]),
                np.asarray([True, True, True]),
                np.asarray([1, 1, 1], dtype=np.int32),
                np.asarray([False, False, False]),
            )

        def tokenize_high_low_prompt(self, high_prompt, low_prompt, state, actions=None):
            raise AssertionError("generation input must not tokenize the ground-truth low_prompt")

    class TokenizeHighLowPrompt:
        tokenizer = Tokenizer()

        def __call__(self, data):
            raise AssertionError("generation input must bypass training low_prompt tokenization")

    transform = open_loop_eval_subtask._CompositeTransformForTest(
        [PromptFromIndices(), SubtaskInputs(), TokenizeHighLowPrompt()]
    )

    result = open_loop_eval_subtask.prepare_subtask_generation_input(
        transform,
        {
            "task_index": np.asarray(0),
            "subtask_index": np.asarray(2),
            "state": np.asarray([0.0, 1.0], dtype=np.float32),
        },
    )

    assert result["high_prompt"] == "spray water"
    assert "low_prompt" not in result
    assert result["tokenized_prompt"].tolist() == [1, 2, 3]
    assert result["token_ar_mask"].tolist() == [1, 1, 1]
