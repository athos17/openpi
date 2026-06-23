# Open Loop Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a script that evaluates OpenPI policies open-loop on LeRobot episodes and reports action MSE/MAE.

**Architecture:** Reuse OpenPI training configs and transforms instead of hard-coding robot-specific field names. Compare predictions and ground truth in policy external action space by applying prompt/repack transforms before inference and keeping the repacked raw action chunk for metrics.

**Tech Stack:** Python, tyro, numpy, matplotlib, LeRobot, OpenPI policy/config modules, pytest.

---

### Task 1: Tests

**Files:**
- Create: `scripts/open_loop_eval_test.py`

- [ ] Add a fake dataset with `episode_data_index` and action chunks.
- [ ] Add a fake policy whose predictions are offset from ground truth by a known amount.
- [ ] Assert `evaluate_single_trajectory` truncates chunked predictions to requested steps and computes expected MSE/MAE.
- [ ] Assert plot path resolution treats a multi-trajectory path as a directory.
- [ ] Run `pytest scripts/open_loop_eval_test.py -q` and confirm it fails before implementation.

### Task 2: Script

**Files:**
- Create: `scripts/open_loop_eval.py`

- [ ] Add dataclass CLI arguments for config name, checkpoint/server policy source, trajectory IDs, steps, action horizon, plotting, and default prompt.
- [ ] Create LeRobot datasets from `DataConfig` with `delta_timestamps` matching the eval action horizon.
- [ ] Implement per-episode evaluation: bounds lookup, policy inference every horizon steps, prediction/GT concatenation, truncation, MSE/MAE.
- [ ] Add optional plotting for state, GT action, predicted action, and inference points.
- [ ] Aggregate metrics across valid trajectory IDs.

### Task 3: Verification

**Files:**
- Test: `scripts/open_loop_eval_test.py`
- Verify: `scripts/open_loop_eval.py`

- [ ] Run `pytest scripts/open_loop_eval_test.py -q`.
- [ ] Run `ruff check scripts/open_loop_eval.py scripts/open_loop_eval_test.py`.
- [ ] Report exact verification results.
