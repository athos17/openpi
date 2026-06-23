# Open Loop Eval Design

## Goal

Add an OpenPI open-loop evaluation script modeled after the GR00T reference: evaluate policy action chunks on recorded LeRobot episodes, compare predicted actions with recorded actions, report MSE/MAE, and optionally save trajectory plots.

## Architecture

The script uses the selected OpenPI training config as the source of truth for dataset keys, action horizon, tolerance, prompts, and transforms. It builds a LeRobot dataset with action delta timestamps, applies only dataset-side pre-policy transforms (`prompt_from_task` and `repack_transforms`) before calling the policy, and compares policy outputs against the repacked raw action chunks in the same external action space used by policy inference.

Policies are loaded in two ways: a local checkpoint through `policy_config.create_trained_policy`, or a websocket policy server through `WebsocketClientPolicy` when no checkpoint is supplied.

## Components

- `scripts/open_loop_eval.py`: CLI, dataset creation, policy creation, trajectory evaluation, metric aggregation, plotting.
- `scripts/open_loop_eval_test.py`: unit tests for trajectory slicing, chunk truncation, metrics, and plot path handling with fake datasets and policies.

## Error Handling

The script validates trajectory IDs, missing action chunks, action dimensions, and short policy chunks with explicit `ValueError` messages. If no valid trajectories are evaluated, it logs that outcome instead of producing averages.

## Testing

Tests use fake dataset and policy objects so they do not require a real LeRobot dataset or checkpoint. Verification runs the targeted pytest file and ruff on the new script and test.
