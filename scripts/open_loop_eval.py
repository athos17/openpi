"""Open-loop evaluation for OpenPI policies on LeRobot episodes."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import base_policy
from openpi_client import websocket_client_policy
import tyro

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    """Arguments for open-loop evaluation."""

    config_name: str = tyro.MISSING
    """Training config name used to build dataset transforms and optional checkpoint policy."""

    checkpoint_dir: str | None = None
    """Checkpoint directory. If omitted, connect to a websocket policy server."""

    host: str = "127.0.0.1"
    """Policy server host used when checkpoint_dir is omitted."""

    port: int = 8000
    """Policy server port used when checkpoint_dir is omitted."""

    api_key: str | None = None
    """Optional API key for websocket policy server."""

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

    save_plot_path: str | None = None
    """File or directory path for trajectory plots."""

    plot: bool = True
    """Whether to save trajectory plots."""

    plot_state: bool = True
    """Whether to include state traces when state/action dimensions match."""

    video_backend: str | None = None
    """Optional LeRobot video backend."""


@dataclasses.dataclass(frozen=True)
class TrajectoryResult:
    traj_id: int
    mse: float
    mae: float
    actual_steps: int
    gt_actions: np.ndarray
    pred_actions: np.ndarray


def create_policy(args: Args, train_config: _config.TrainConfig) -> base_policy.BasePolicy:
    if args.checkpoint_dir is None:
        if args.num_steps is not None:
            logging.warning("--num-steps is ignored when using a websocket policy server.")
        return websocket_client_policy.WebsocketClientPolicy(args.host, args.port, args.api_key)

    sample_kwargs = {"num_steps": args.num_steps} if args.num_steps is not None else None
    return _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs=sample_kwargs,
        default_prompt=args.default_prompt,
    )


def create_data_config(args: Args, train_config: _config.TrainConfig) -> _config.DataConfig:
    data_factory = train_config.data
    if args.dataset_repo_id is not None:
        data_factory = dataclasses.replace(data_factory, repo_id=args.dataset_repo_id)
    return data_factory.create(train_config.assets_dirs, train_config.model)


def create_lerobot_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    *,
    video_backend: str | None = None,
):
    if data_config.repo_id is None:
        raise ValueError("Open-loop evaluation requires a LeRobot repo_id in the selected config.")
    if data_config.rlds_data_dir is not None:
        raise ValueError("Open-loop evaluation currently supports LeRobot datasets, not RLDS datasets.")

    _validate_local_lerobot_root(data_config.repo_id)
    dataset_meta = _data_loader.lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    return _data_loader.lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [step / dataset_meta.fps for step in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        tolerance_s=data_config.lerobot_tolerance_s,
        video_backend=video_backend,
    )


def _validate_local_lerobot_root(repo_id: str) -> None:
    path = Path(repo_id)
    if not path.is_absolute() and not path.exists():
        return
    if not path.exists():
        raise ValueError(f"Local dataset path does not exist: {repo_id}")
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError(
            f"Local dataset path is not a complete LeRobot dataset root: {repo_id}. "
            f"Expected metadata file: {info_path}"
        )


def get_episode_bounds(dataset, traj_id: int) -> tuple[int, int]:
    try:
        episode_data_index = dataset.episode_data_index
        start = _to_int(episode_data_index["from"][traj_id])
        end = _to_int(episode_data_index["to"][traj_id])
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Trajectory ID {traj_id} is out of range.") from exc
    if end <= start:
        raise ValueError(f"Trajectory ID {traj_id} has no frames.")
    return start, end


def evaluate_single_trajectory(
    *,
    policy: base_policy.BasePolicy,
    dataset,
    data_config: _config.DataConfig,
    traj_id: int,
    steps: int,
    action_horizon: int,
    save_plot_path: str | Path | None,
    plot: bool = True,
    plot_state: bool = True,
) -> TrajectoryResult:
    start, end = get_episode_bounds(dataset, traj_id)
    actual_steps = min(steps, end - start)
    logging.info("Using %d steps (requested: %d, trajectory length: %d)", actual_steps, steps, end - start)

    pred_chunks = []
    gt_chunks = []
    for offset in range(0, actual_steps, action_horizon):
        global_index = start + offset
        logging.info("Inferencing trajectory %d at step %d", traj_id, offset)
        repacked = prepare_repacked_item(dataset[global_index], data_config, dataset)
        gt_chunks.append(_extract_action_chunk(repacked, action_horizon, source="ground truth"))
        pred_chunks.append(_extract_action_chunk(policy.infer(_observation_from_repacked(repacked)), action_horizon))

    gt_actions = np.concatenate(gt_chunks, axis=0)[:actual_steps]
    pred_actions = np.concatenate(pred_chunks, axis=0)[:actual_steps]
    if gt_actions.shape != pred_actions.shape:
        raise ValueError(f"gt_actions shape {gt_actions.shape} does not match pred_actions shape {pred_actions.shape}")

    mse = float(np.mean((gt_actions - pred_actions) ** 2))
    mae = float(np.mean(np.abs(gt_actions - pred_actions)))
    logging.info("Unnormalized action MSE for trajectory %d: %s", traj_id, mse)
    logging.info("Unnormalized action MAE for trajectory %d: %s", traj_id, mae)

    if plot and save_plot_path is not None:
        state = collect_state_trajectory(dataset, data_config, traj_id, actual_steps)
        plot_trajectory_results(
            state_across_time=state,
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            traj_id=traj_id,
            action_horizon=action_horizon,
            save_plot_path=Path(save_plot_path),
            plot_state=plot_state,
        )

    return TrajectoryResult(
        traj_id=traj_id,
        mse=mse,
        mae=mae,
        actual_steps=actual_steps,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
    )


def prepare_repacked_item(item: dict[str, Any], data_config: _config.DataConfig, dataset) -> dict[str, Any]:
    transforms: list[_transforms.DataTransformFn] = []
    if data_config.prompt_from_task:
        transforms.append(_transforms.PromptFromLeRobotTask(dataset.meta.tasks))
    transforms.extend(data_config.repack_transforms.inputs)
    return _transforms.compose(transforms)(dict(item))


def collect_state_trajectory(
    dataset,
    data_config: _config.DataConfig,
    traj_id: int,
    actual_steps: int,
) -> np.ndarray | None:
    start, _ = get_episode_bounds(dataset, traj_id)
    states = []
    for offset in range(actual_steps):
        repacked = prepare_repacked_item(dataset[start + offset], data_config, dataset)
        if "state" not in repacked:
            return None
        states.append(np.ravel(_to_numpy(repacked["state"])))
    try:
        return np.stack(states, axis=0)
    except ValueError:
        logging.warning("State shapes are inconsistent; skipping state plot.")
        return None


def plot_trajectory_results(
    *,
    state_across_time: np.ndarray | None,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    traj_id: int,
    action_horizon: int,
    save_plot_path: Path,
    plot_state: bool = True,
) -> None:
    from matplotlib import pyplot as plt

    action_dim = gt_actions.shape[1]
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(8, 4 * action_dim))
    if action_dim == 1:
        axes = [axes]

    fig.suptitle(f"Trajectory {traj_id}", fontsize=16, color="blue")
    can_plot_state = plot_state and state_across_time is not None and state_across_time.shape == gt_actions.shape
    for action_idx, ax in enumerate(axes):
        if can_plot_state:
            ax.plot(state_across_time[:, action_idx], label="state")
        ax.plot(gt_actions[:, action_idx], label="gt action")
        ax.plot(pred_actions[:, action_idx], label="pred action")
        for step in range(0, len(gt_actions), action_horizon):
            label = "inference point" if step == 0 else None
            ax.plot(step, gt_actions[step, action_idx], "ro", label=label)
        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()
    save_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)
    plt.close(fig)


def resolve_plot_path(args: Args, traj_id: int) -> Path | None:
    if args.save_plot_path is None:
        return None
    path = Path(args.save_plot_path)
    if len(args.traj_ids) > 1 or path.suffix == "" or path.is_dir():
        return path / f"traj_{traj_id}.jpeg"
    return path


def _observation_from_repacked(repacked: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in repacked.items() if key != "actions"}


def _extract_action_chunk(
    data: dict[str, Any],
    action_horizon: int,
    *,
    source: str = "policy",
) -> np.ndarray:
    if "actions" not in data:
        raise ValueError(f"{source} output is missing an 'actions' key.")
    actions = _to_numpy(data["actions"]).astype(np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"{source} actions must have shape [horizon, dim]; got {actions.shape}.")
    if actions.shape[0] < action_horizon:
        raise ValueError(f"{source} action chunk has horizon {actions.shape[0]}, expected at least {action_horizon}.")
    return actions[:action_horizon]


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_int(value) -> int:
    array = _to_numpy(value)
    if array.shape == ():
        return int(array.item())
    if array.size == 1:
        return int(array.reshape(()).item())
    raise ValueError(f"Expected scalar-like value, got shape {array.shape}.")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    train_config = _config.get_config(args.config_name)
    action_horizon = args.action_horizon or train_config.model.action_horizon
    data_config = create_data_config(args, train_config)
    policy = create_policy(args, train_config)
    dataset = create_lerobot_dataset(data_config, action_horizon, video_backend=args.video_backend)

    logging.info("Dataset length: %d", len(dataset))
    logging.info("Running evaluation on trajectories: %s", args.traj_ids)

    results: list[TrajectoryResult] = []
    for traj_id in args.traj_ids:
        try:
            result = evaluate_single_trajectory(
                policy=policy,
                dataset=dataset,
                data_config=data_config,
                traj_id=traj_id,
                steps=args.steps,
                action_horizon=action_horizon,
                save_plot_path=resolve_plot_path(args, traj_id),
                plot=args.plot,
                plot_state=args.plot_state,
            )
        except ValueError as exc:
            logging.warning("Skipping trajectory %d: %s", traj_id, exc)
            continue
        logging.info("MSE for trajectory %d: %s, MAE: %s", traj_id, result.mse, result.mae)
        results.append(result)

    if not results:
        logging.info("No valid trajectories were evaluated.")
        return

    logging.info("Average MSE across all trajectories: %s", np.mean([result.mse for result in results]))
    logging.info("Average MAE across all trajectories: %s", np.mean([result.mae for result in results]))
    logging.info("Done")


if __name__ == "__main__":
    main(tyro.cli(Args))
