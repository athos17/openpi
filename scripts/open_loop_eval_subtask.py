"""Subtask-aware open-loop evaluation for local WUJI OpenPI checkpoints."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

import tyro

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from . import open_loop_eval


@dataclasses.dataclass
class Args:
    """Arguments for local subtask open-loop evaluation."""

    config_name: str = tyro.MISSING
    """Training config name used to build dataset transforms and checkpoint policy."""

    checkpoint_dir: str | None = None
    """Local checkpoint directory. Required; websocket policies are intentionally unsupported."""

    default_prompt: str | None = None
    """Default prompt injected by locally loaded policies when the dataset does not provide one."""

    dataset_repo_id: str | None = None
    """Optional LeRobot repo ID or local dataset path overriding the selected config's dataset."""

    traj_ids: list[int] = dataclasses.field(default_factory=lambda: [0])
    """Episode IDs to evaluate."""

    steps: int = 1000
    """Maximum number of steps per episode. This is capped by the episode length."""

    action_horizon: int | None = None
    """Number of actions to consume from each predicted chunk. Defaults to the model action horizon."""

    num_steps: int | None = None
    """Optional denoising steps passed to local policy sampling."""

    mode: Literal["fast", "full"] = "fast"
    """fast writes only annotated video; full writes annotated video and action plots."""

    save_video_path: str | None = None
    """File or directory path for annotated main-view videos."""

    save_plot_path: str | None = None
    """File or directory path for trajectory plots in full mode."""

    plot_state: bool = True
    """Whether full-mode plots include state traces when state/action dimensions match."""

    video_backend: str | None = None
    """Optional LeRobot video backend."""

    video_fps: int | None = None
    """Output video FPS. Defaults to dataset metadata FPS when available, otherwise 30."""

    main_camera_key: str = "observation.images.head_view"
    """Raw dataset key for the video main camera."""

    max_subtask_decoding_steps: int = 25
    """Maximum generated low-level subtask token count per inference point."""

    subtask_temperature: float = 0.0
    """Temperature for subtask token sampling."""


def validate_args(args: Args) -> None:
    if args.checkpoint_dir is None:
        raise ValueError("Subtask open-loop evaluation requires checkpoint_dir for a local checkpoint.")


def resolve_video_path(args: Args, traj_id: int) -> Path | None:
    if args.save_video_path is None:
        return None
    path = Path(args.save_video_path)
    if len(args.traj_ids) > 1 or path.suffix == "" or path.is_dir():
        return path / f"traj_{traj_id}.mp4"
    return path


def resolve_plot_path(args: Args, traj_id: int) -> Path | None:
    if args.mode != "full" or args.save_plot_path is None:
        return None
    original_args = open_loop_eval.Args(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        dataset_repo_id=args.dataset_repo_id,
        traj_ids=args.traj_ids,
        steps=args.steps,
        action_horizon=args.action_horizon,
        num_steps=args.num_steps,
        save_plot_path=args.save_plot_path,
        plot=True,
        plot_state=args.plot_state,
        video_backend=args.video_backend,
    )
    return open_loop_eval.resolve_plot_path(original_args, traj_id)



def expand_subtask_schedule(*, actual_steps: int, action_horizon: int, chunk_subtasks: list[str]) -> list[str]:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive; got {action_horizon}")
    schedule: list[str] = []
    for chunk_index, subtask in enumerate(chunk_subtasks):
        start = chunk_index * action_horizon
        end = min(start + action_horizon, actual_steps)
        if start >= actual_steps:
            break
        schedule.extend([subtask] * (end - start))
    if len(schedule) != actual_steps:
        raise ValueError(
            f"Subtask schedule has {len(schedule)} frames, expected {actual_steps}; "
            f"got {len(chunk_subtasks)} chunk subtasks with horizon {action_horizon}."
        )
    return schedule


def _lookup_nested(data: dict[str, Any], dotted_key: str) -> Any:
    if dotted_key in data:
        return data[dotted_key]
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Could not find camera key '{dotted_key}' in dataset item.")
        current = current[part]
    return current


def extract_main_frame(item: dict[str, Any], main_camera_key: str) -> np.ndarray:
    frame = np.asarray(_lookup_nested(item, main_camera_key))
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8, copy=False)
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Main camera frame must have shape HxWxC or CxHxW; got {frame.shape}")
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.ascontiguousarray(frame)


def overlay_subtask_text(frame: np.ndarray, subtask: str) -> np.ndarray:
    import cv2

    annotated = np.asarray(frame).copy()
    label = f"Subtask: {subtask}" if subtask else "Subtask: <empty>"
    if len(label) > 48:
        label = label[:45] + "..."

    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(width, height) / 450.0)
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 8
    x0 = max(0, width - text_w - 2 * pad)
    y0 = 0
    x1 = width
    y1 = min(height, text_h + baseline + 2 * pad)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        label,
        (x0 + pad, y0 + pad + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


@dataclasses.dataclass(frozen=True)
class SubtaskInferenceResult:
    actions: np.ndarray
    subtask: str
    subtask_tokens: np.ndarray


def decode_subtask_tokens(tokenizer: Any, tokens: Any) -> str:
    token_array = open_loop_eval._to_numpy(tokens)
    if token_array.ndim == 2:
        token_array = token_array[0]
    token_list = [int(token) for token in token_array.tolist()]
    if 1 in token_list:
        token_list = token_list[: token_list.index(1)]
    if tokenizer is None:
        return " ".join(str(token) for token in token_list).strip()
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_list, skip_special_tokens=True)).strip()
    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "decode"):
        return str(tokenizer._tokenizer.decode(token_list, skip_special_tokens=True)).strip()
    return " ".join(str(token) for token in token_list).strip()



@dataclasses.dataclass(frozen=True)
class _CompositeTransformForTest:
    transforms: list[Any]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data


def prepare_subtask_generation_input(input_transform: Any, repacked: dict[str, Any]) -> dict[str, Any]:
    """Apply policy input transforms while building an empty low-level prompt for subtask generation."""
    transforms = getattr(input_transform, "transforms", None)
    if transforms is None:
        raise ValueError("Subtask generation requires an inspectable composite input transform.")

    data = dict(repacked)
    generated_prefix = False
    for transform in transforms:
        tokenizer = getattr(transform, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "tokenize_high_low_prompt_infer"):
            high_prompt = data.get("high_prompt")
            data.pop("low_prompt", None)
            if high_prompt is None:
                raise ValueError("Subtask generation input is missing high_prompt before tokenization.")
            if not isinstance(high_prompt, str):
                high_prompt = np.asarray(high_prompt).item()
            tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize_high_low_prompt_infer(
                str(high_prompt), data["state"]
            )
            data.update(
                {
                    "tokenized_prompt": tokens,
                    "tokenized_prompt_mask": token_mask,
                    "token_ar_mask": ar_mask,
                    "token_loss_mask": loss_mask,
                }
            )
            generated_prefix = True
            continue

        data = transform(data)
        if "high_prompt" in data and "low_prompt" in data and not generated_prefix:
            data = {**data, "low_prompt": ""}

    if not generated_prefix:
        raise ValueError("Input transform did not contain a subtask tokenizer with tokenize_high_low_prompt_infer().")
    return data

class LocalSubtaskPolicyAdapter:
    def __init__(self, policy):
        self._policy = policy
        self._model = getattr(policy, "_model", None)
        self._device = getattr(policy, "_pytorch_device", None)
        self._input_transform = getattr(policy, "_input_transform", None)
        self._output_transform = getattr(policy, "_output_transform", None)
        self._sample_kwargs = getattr(policy, "_sample_kwargs", {}) or {}
        self._tokenizer = _find_tokenizer(self._model)
        if self._model is None or self._device is None or self._input_transform is None or self._output_transform is None:
            raise ValueError("Local subtask eval requires a PyTorch policy with model, device, and transforms.")
        if not getattr(policy, "_is_pytorch_model", False):
            raise ValueError("Local subtask eval requires a PyTorch checkpoint policy.")
        if not hasattr(self._model, "sample_low_level_task") or not hasattr(self._model, "denoise_step"):
            raise ValueError("Checkpoint model does not expose subtask generation APIs.")

    def infer_subtask_then_actions(
        self,
        repacked: dict[str, Any],
        *,
        action_horizon: int,
        num_steps: int,
        max_subtask_decoding_steps: int,
        subtask_temperature: float,
    ) -> SubtaskInferenceResult:
        import torch

        transformed = prepare_subtask_generation_input(self._input_transform, repacked)
        batched = {key: _batch_value(value) for key, value in transformed.items()}
        observation = _model.Observation.from_dict(batched)
        observation = _model.preprocess_observation(None, observation, train=False)
        observation = _move_observation_to_device(observation, self._device)

        output_tokens, past_key_values, prefix_pad_masks, _ = self._model.sample_low_level_task(
            self._device,
            observation,
            max_decoding_steps=max_subtask_decoding_steps,
            paligemma_eos_token=1,
            temperature=subtask_temperature,
        )
        subtask = decode_subtask_tokens(self._tokenizer, output_tokens)

        bsize = observation.state.shape[0]
        actions_shape = (bsize, self._model.config.action_horizon, self._model.config.action_dim)
        noise = self._model.sample_noise(actions_shape, self._device)
        _, _, _, _, state = self._model._preprocess_observation(observation, train=False)  # noqa: SLF001

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=self._device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=self._device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt

        raw_actions = open_loop_eval._to_numpy(x_t)[0]
        unnormalized = self._output_transform({"actions": raw_actions})
        actions = open_loop_eval._extract_action_chunk(unnormalized, action_horizon, source="subtask policy")
        return SubtaskInferenceResult(
            actions=actions,
            subtask=subtask,
            subtask_tokens=open_loop_eval._to_numpy(output_tokens[0]),
        )


def _find_tokenizer(model: Any) -> Any:
    for attr_path in (
        ("tokenizer",),
        ("paligemma_with_expert", "tokenizer"),
        ("paligemma", "tokenizer"),
    ):
        current = model
        for attr in attr_path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current
    return None


def _batch_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _batch_value(child) for key, child in value.items()}
    return np.expand_dims(value, axis=0)


def _move_observation_to_device(observation: _model.Observation, device) -> _model.Observation:
    import torch

    def move(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {key: move(child) for key, child in value.items()}
        tensor = torch.as_tensor(value, device=device)
        if tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)
        return tensor

    return _model.Observation.from_dict({key: move(value) for key, value in observation.to_dict().items()})


def create_local_subtask_policy(args: Args, train_config: _config.TrainConfig) -> LocalSubtaskPolicyAdapter:
    validate_args(args)
    sample_kwargs = {"num_steps": args.num_steps} if args.num_steps is not None else None
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs=sample_kwargs,
        default_prompt=args.default_prompt,
    )
    return LocalSubtaskPolicyAdapter(policy)


@dataclasses.dataclass(frozen=True)
class SubtaskTrajectoryResult:
    traj_id: int
    mse: float
    mae: float
    actual_steps: int
    gt_actions: np.ndarray
    pred_actions: np.ndarray
    chunk_subtasks: list[str]
    frame_subtasks: list[str]


def evaluate_single_trajectory(
    *,
    policy,
    dataset,
    data_config: _config.DataConfig,
    traj_id: int,
    steps: int,
    action_horizon: int,
    num_steps: int,
    max_subtask_decoding_steps: int,
    subtask_temperature: float,
    save_video_path: str | Path | None,
    save_plot_path: str | Path | None,
    main_camera_key: str,
    video_fps: int,
    plot_state: bool = True,
) -> SubtaskTrajectoryResult:
    start, end = open_loop_eval.get_episode_bounds(dataset, traj_id)
    actual_steps = min(steps, end - start)

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    chunk_subtasks: list[str] = []
    for offset in range(0, actual_steps, action_horizon):
        global_index = start + offset
        repacked = open_loop_eval.prepare_repacked_item(dataset[global_index], data_config, dataset)
        gt_chunks.append(open_loop_eval._extract_action_chunk(repacked, action_horizon, source="ground truth"))
        inference = policy.infer_subtask_then_actions(
            repacked,
            action_horizon=action_horizon,
            num_steps=num_steps,
            max_subtask_decoding_steps=max_subtask_decoding_steps,
            subtask_temperature=subtask_temperature,
        )
        pred_chunks.append(inference.actions)
        chunk_subtasks.append(inference.subtask)

    gt_actions = np.concatenate(gt_chunks, axis=0)[:actual_steps]
    pred_actions = np.concatenate(pred_chunks, axis=0)[:actual_steps]
    if gt_actions.shape != pred_actions.shape:
        raise ValueError(f"gt_actions shape {gt_actions.shape} does not match pred_actions shape {pred_actions.shape}")

    mse = float(np.mean((gt_actions - pred_actions) ** 2))
    mae = float(np.mean(np.abs(gt_actions - pred_actions)))
    frame_subtasks = expand_subtask_schedule(
        actual_steps=actual_steps,
        action_horizon=action_horizon,
        chunk_subtasks=chunk_subtasks,
    )

    if save_video_path is not None:
        write_annotated_video(
            dataset=dataset,
            start=start,
            actual_steps=actual_steps,
            frame_subtasks=frame_subtasks,
            save_video_path=Path(save_video_path),
            main_camera_key=main_camera_key,
            fps=video_fps,
        )

    if save_plot_path is not None:
        state = open_loop_eval.collect_state_trajectory(dataset, data_config, traj_id, actual_steps)
        open_loop_eval.plot_trajectory_results(
            state_across_time=state,
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            traj_id=traj_id,
            action_horizon=action_horizon,
            save_plot_path=Path(save_plot_path),
            plot_state=plot_state,
        )

    return SubtaskTrajectoryResult(
        traj_id=traj_id,
        mse=mse,
        mae=mae,
        actual_steps=actual_steps,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        chunk_subtasks=chunk_subtasks,
        frame_subtasks=frame_subtasks,
    )


def write_annotated_video(
    *,
    dataset,
    start: int,
    actual_steps: int,
    frame_subtasks: list[str],
    save_video_path: Path,
    main_camera_key: str,
    fps: int,
) -> None:
    import cv2

    if len(frame_subtasks) != actual_steps:
        raise ValueError(f"Expected {actual_steps} frame subtasks, got {len(frame_subtasks)}")
    if actual_steps <= 0:
        raise ValueError("actual_steps must be positive to write a video")

    save_video_path.parent.mkdir(parents=True, exist_ok=True)
    first = overlay_subtask_text(extract_main_frame(dataset[start], main_camera_key), frame_subtasks[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(save_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {save_video_path}")
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for offset in range(1, actual_steps):
            frame = extract_main_frame(dataset[start + offset], main_camera_key)
            annotated = overlay_subtask_text(frame, frame_subtasks[offset])
            if annotated.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {offset} has shape {annotated.shape[:2]}, expected {(height, width)}"
                )
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def resolve_video_fps(arg_fps: int | None, dataset) -> int:
    if arg_fps is not None:
        return int(arg_fps)
    fps = getattr(getattr(dataset, "meta", None), "fps", None)
    if fps is not None:
        return int(fps)
    return 30

def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    validate_args(args)
    train_config = _config.get_config(args.config_name)
    action_horizon = args.action_horizon or train_config.model.action_horizon
    num_steps = args.num_steps or 10
    data_config = open_loop_eval.create_data_config(
        open_loop_eval.Args(
            config_name=args.config_name,
            checkpoint_dir=args.checkpoint_dir,
            default_prompt=args.default_prompt,
            dataset_repo_id=args.dataset_repo_id,
            action_horizon=args.action_horizon,
            num_steps=args.num_steps,
            video_backend=args.video_backend,
        ),
        train_config,
    )
    policy = create_local_subtask_policy(args, train_config)
    dataset = open_loop_eval.create_lerobot_dataset(data_config, action_horizon, video_backend=args.video_backend)
    video_fps = resolve_video_fps(args.video_fps, dataset)

    logging.info("Dataset length: %d", len(dataset))
    logging.info("Running subtask evaluation on trajectories: %s", args.traj_ids)
    logging.info("Mode: %s", args.mode)

    results: list[SubtaskTrajectoryResult] = []
    for traj_id in args.traj_ids:
        try:
            result = evaluate_single_trajectory(
                policy=policy,
                dataset=dataset,
                data_config=data_config,
                traj_id=traj_id,
                steps=args.steps,
                action_horizon=action_horizon,
                num_steps=num_steps,
                max_subtask_decoding_steps=args.max_subtask_decoding_steps,
                subtask_temperature=args.subtask_temperature,
                save_video_path=resolve_video_path(args, traj_id),
                save_plot_path=resolve_plot_path(args, traj_id),
                main_camera_key=args.main_camera_key,
                video_fps=video_fps,
                plot_state=args.plot_state,
            )
        except ValueError as exc:
            logging.warning("Skipping trajectory %d: %s", traj_id, exc)
            continue
        logging.info(
            "Trajectory %d: MSE=%s MAE=%s subtasks=%s",
            traj_id,
            result.mse,
            result.mae,
            result.chunk_subtasks,
        )
        results.append(result)

    if not results:
        logging.info("No valid trajectories were evaluated.")
        return

    logging.info("Average MSE across all trajectories: %s", np.mean([result.mse for result in results]))
    logging.info("Average MAE across all trajectories: %s", np.mean([result.mae for result in results]))
    logging.info("Done")


if __name__ == "__main__":
    main(tyro.cli(Args))
