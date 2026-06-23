# WUJI Subtask Policy Server Design

Date: 2026-06-23

## Goal

Add a dedicated WUJI subtask websocket server for the current `openpi` repository. The server will expose the trained mechanical-arm + dexterous-hand WUJI 58D subtask model over the existing `msgpack_numpy` websocket protocol.

The server should support:

- default one-request pipeline: generate a low-level subtask, then generate a 58D action chunk;
- explicit `subtask_only` mode for debugging and monitoring;
- explicit `actions_only` mode using a caller-provided low-level prompt;
- optional periodic subtask refresh messages on an open websocket connection;
- existing checkpoint/config loading through `policy_config.create_trained_policy()`.

Client implementation is out of scope for this design. The protocol should remain compatible with a future real-robot WUJI client.

## Current Context

The repository already has WUJI subtask training support:

- `src/openpi/policies/wuji_policy.py` defines WUJI 58D input/output transforms.
- `src/openpi/training/config.py` defines WUJI subtask configs such as `pi05_wuji_spray_water_rot6d_subtask_flow_pytorch` and `pi05_wuji_spray_water_rot6d_subtask_hybrid_pytorch`.
- `src/openpi/models/tokenizer.py` supports high-level task + low-level subtask tokenization and region masks.
- `src/openpi/models_pytorch/pi0_pytorch.py` implements `sample_low_level_task()` and `sample_actions_with_subtask()`.

The missing piece is serving integration. The default `Policy.infer()` path calls `model.sample_actions()` and does not expose the subtask pipeline or return generated subtask text/tokens.

## Chosen Approach

Use a dedicated WUJI subtask server with an inference-engine layer.

Rejected alternatives:

1. Single-file server script: fastest to write, but couples model logic, transforms, websocket handling, and refresh state in one file.
2. Extending the generic `WebsocketPolicyServer`: keeps one server abstraction, but the current generic server assumes `policy.infer(obs)` and does not fit mode dispatch or server-initiated refresh pushes cleanly.

The chosen design keeps boundaries clear and testable.

## Modules

### `src/openpi/policies/wuji_subtask_inference.py`

Defines `WujiSubtaskInferenceEngine`.

Responsibilities:

- load or accept an already-created `Policy` from `policy_config.create_trained_policy()`;
- validate that the policy is PyTorch-backed and WUJI-compatible;
- reuse the policy's input transforms, output transforms, model, metadata, and PyTorch device;
- provide synchronous model APIs used by the websocket server:
  - `infer_subtask_then_actions(request)`;
  - `infer_subtask_only(request)`;
  - `infer_actions_only(request)`.

The engine owns model-level details and transform details. It does not own websocket connection state.

### `src/openpi/serving/wuji_subtask_websocket_server.py`

Defines the dedicated websocket server.

Responsibilities:

- use `openpi_client.msgpack_numpy` for requests and responses;
- send metadata after client connection;
- parse request mode and options;
- dispatch to `WujiSubtaskInferenceEngine`;
- attach timing information;
- return structured error responses for non-fatal request errors;
- manage optional per-connection `subtask_refresh` background tasks;
- cancel refresh tasks on new requests or connection close.

The server does not manipulate PyTorch tensors or tokenizer internals directly.

### `scripts/serve_wuji_subtask_policy.py`

Command-line entry point.

Responsibilities:

- parse arguments such as `--config`, `--checkpoint-dir`, `--host`, `--port`, and `--pytorch-device`;
- call `policy_config.create_trained_policy()`;
- create the inference engine;
- create and run the websocket server.

## Policy Loading

The script will reuse existing loading:

```python
policy = policy_config.create_trained_policy(
    config.get_config(args.config),
    args.checkpoint_dir,
    pytorch_device=args.pytorch_device,
)
```

The engine will then validate:

- `policy._is_pytorch_model` is true;
- `policy._model` has `sample_low_level_task()`;
- `policy._model` has either `sample_actions_with_subtask()` or the lower-level helpers needed to denoise from a generated prefix;
- `policy._model.config.action_dim == 58`;
- WUJI output transforms preserve a returned action chunk with trailing dimension 58.

Private policy attributes are used deliberately in the first version because the existing public `Policy.infer()` interface does not expose subtask generation. If this pattern becomes common, a later refactor can add public accessors or a specialized policy subclass.

## Request Protocol

The server uses msgpack over websocket. A connected client sends a Python dict encoded with `msgpack_numpy`.

Default request:

```python
{
    "type": "infer",
    "mode": "subtask_then_actions",
    "images": {
        "head_view": np.ndarray,
        "left_wrist_view": np.ndarray,
        "right_wrist_view": np.ndarray,
    },
    "state": np.ndarray,        # shape (58,)
    "prompt": str,              # high-level task

    "low_prompt": str,          # required only for actions_only
    "max_subtask_decoding_steps": 25,
    "subtask_temperature": 0.0,
    "num_action_steps": 10,
    "noise": np.ndarray | None,
    "subtask_refresh_interval": float | None,
}
```

`type` may be omitted and defaults to `infer`.

Supported modes:

- `subtask_then_actions`: default mode. Generate subtask, then generate actions using the generated subtask prefix.
- `subtask_only`: generate subtask text and tokens only.
- `actions_only`: generate actions using request-provided `low_prompt`; do not generate a new subtask.

The public input keys are WUJI robot keys: `head_view`, `left_wrist_view`, `right_wrist_view`, `state`, and `prompt`. Clients should not need to know model-internal image names such as `base_0_rgb`.

## Response Protocol

Normal inference response:

```python
{
    "type": "infer_result",
    "mode": str,
    "state": np.ndarray,
    "actions": np.ndarray | None,       # shape (16, 58) after output transforms
    "subtask": str | None,
    "subtask_tokens": np.ndarray | None,
    "action_schema": {
        "format": "wuji_rot6d_58d",
        "horizon": 16,
        "slices": {
            "left_eef": [0, 9],
            "right_eef": [9, 18],
            "left_hand_joints": [18, 38],
            "right_hand_joints": [38, 58],
        },
    },
    "timing": {
        "total_ms": float,
        "transform_ms": float,
        "subtask_ms": float,
        "action_ms": float,
    },
    "subtask_refresh_enabled": bool,
    "subtask_refresh_interval": float | None,
}
```

The first version returns actions as a flat `(T, 58)` array. It does not split actions into `left_eef`, `right_eef`, `left_hand_joints`, and `right_hand_joints`; that split belongs in the future real-robot client because client-side safety, interpolation, clipping, and controller timing may differ by deployment.

Error response for non-fatal request errors:

```python
{
    "type": "error",
    "error": "ValueError",
    "message": "state must have shape (58,)",
    "traceback": "debug traceback string",   # only when debug tracebacks are enabled
}
```

Non-fatal request errors do not close the websocket. Protocol corruption or unrecoverable server errors may close the connection.

## Inference Data Flow

### Input preparation

The engine uses the existing policy input transform chain rather than duplicating WUJI preprocessing.

For `subtask_then_actions` and `subtask_only`:

1. Convert request `prompt` to high-level prompt.
2. Provide an empty low-level prompt when building a model observation so the tokenizer produces the `Task: ...; State: ...; Subtask:` prefix.
3. Apply `policy._input_transform`.
4. Batch the transformed data.
5. Convert arrays to torch tensors on `policy._pytorch_device`.
6. Build `openpi.models.model.Observation`.

For `actions_only`:

1. Require `low_prompt`.
2. Build an observation containing high-level prompt and the provided low-level prompt.
3. Apply the same transform/batch/tensor steps.

### Subtask generation

Call:

```python
model.sample_low_level_task(
    device,
    observation,
    max_decoding_steps=max_subtask_decoding_steps,
    paligemma_eos_token=1,
    temperature=subtask_temperature,
)
```

This returns generated token IDs, `past_key_values`, and prefix masks that include the generated subtask tokens.

The repository should add a PaliGemma detokenization helper equivalent to the one in `/data_all/liyunhao/openpi_subtask`, because the current tokenizer has inference tokenization but not a plain subtask `detokenize()` method.

### `subtask_then_actions`

The server needs both generated subtask text/tokens and actions. Therefore, the engine should avoid treating the current `sample_actions_with_subtask()` as a complete black box because that method returns only actions.

Preferred implementation design:

1. Call `sample_low_level_task()` once.
2. Detokenize generated subtask tokens for the response.
3. Denoise actions using the returned `past_key_values` and `prefix_pad_masks`.
4. Run policy output transforms on the action tensor.
5. Return subtask, tokens, transformed actions, and timing.

The denoising helper can initially live in the engine by reusing the second half of `sample_actions_with_subtask()`. If implementation review shows it is cleaner, refactor the PyTorch model to expose a helper such as `sample_actions_from_generated_prefix()`.

### `subtask_only`

Call only `sample_low_level_task()`, detokenize, and return no actions.

### `actions_only`

Use the request-provided `low_prompt` as the low-level prompt and generate actions without generating a new subtask. The semantics are: "condition on this provided subtask text".

The implementation should ensure this mode does not call `sample_low_level_task()`.

## Periodic Subtask Refresh

Refresh is included in the first version.

When a request sets `subtask_refresh_interval > 0` and mode is `subtask_then_actions` or `subtask_only`, the server starts a per-connection background refresh task.

Refresh behavior:

- uses the latest request snapshot that enabled refresh;
- the snapshot includes images, state, prompt/high prompt, and decoding options;
- periodically runs `subtask_only` logic;
- pushes a server-initiated message:

```python
{
    "type": "subtask_refresh",
    "subtask": str,
    "subtask_tokens": np.ndarray,
    "refresh_count": int,
    "timing": {"subtask_ms": float},
}
```

Refresh does not automatically replace currently executing actions and does not automatically generate a new action chunk. The client decides whether to use the refreshed subtask.

Connection state rules:

- A new request on the same connection cancels the old refresh task.
- If the new request includes a positive refresh interval, the server starts a new refresh task from the new snapshot.
- If the new request omits refresh or sets it to `None`/`<=0`, refresh remains disabled.
- Connection close cancels refresh.
- Refresh exceptions are logged and reported as an error message if possible, then that refresh task stops.

## Concurrency

Model calls should not run concurrently on the same model instance.

The engine or server will maintain a lock so that main inference requests and refresh tasks serialize access to the PyTorch model. The websocket event loop should not block during model calls; the server should use `asyncio.to_thread()` or an executor, guarded by the model lock.

This design favors correctness and stable GPU memory behavior over maximum throughput. Multi-client concurrent throughput can be revisited later.

## WUJI 58D Action Compatibility

The server returns actions as `(T, 58)` after output transforms.

The action schema is:

- `left_eef`: indices `[0, 9)`, `xyz + rot6d`;
- `right_eef`: indices `[9, 18)`, `xyz + rot6d`;
- `left_hand_joints`: indices `[18, 38)`, 20D hand joints;
- `right_hand_joints`: indices `[38, 58)`, 20D hand joints.

The action output must go through the existing policy output transform chain so scale and normalization match the ordinary trained policy path.

## Out of Scope

This design does not include:

- a real-robot client implementation;
- RTC overlap/frozen-step action guidance;
- JSON websocket protocol;
- automatic action replanning triggered by refresh messages;
- splitting response actions into GR00T-style four-key action dicts;
- high-frequency observation update streaming.

## Testing Strategy

### Engine tests

Use fake policy/model objects where possible.

Verify:

- `subtask_then_actions` calls subtask generation before action denoising;
- `subtask_only` returns subtask and tokens, with no actions;
- `actions_only` requires `low_prompt` and does not call subtask generation;
- output actions pass through output transforms;
- invalid state/action dimensions raise clear errors;
- detokenization truncates EOS and ignores padding.

### Server protocol tests

Use a fake engine and an in-process websocket server.

Verify:

- initial metadata is sent on connection;
- msgpack `mode=subtask_then_actions` returns `type=infer_result`;
- invalid requests return `type=error` without closing the connection;
- `subtask_refresh_interval > 0` produces at least one `subtask_refresh` message;
- a new request cancels the previous refresh task;
- connection close cancels refresh.

### Script tests

Verify:

- `scripts/serve_wuji_subtask_policy.py --help` runs;
- argument parsing supports config, checkpoint directory, host, port, PyTorch device, decoding defaults, and refresh options;
- server construction can be tested with mocked policy loading.

A real checkpoint/GPU smoke test can be run separately when hardware is available.

## Success Criteria

The implementation is successful when:

1. A WUJI checkpoint can be served with the new script.
2. A msgpack websocket client can send WUJI images, 58D state, and high-level prompt.
3. The default response contains generated subtask text, subtask tokens, and unnormalized `(16, 58)` actions.
4. `subtask_only` and `actions_only` modes work as explicit debug modes.
5. Optional refresh pushes `subtask_refresh` messages without interrupting normal requests.
6. Existing generic `serve_policy.py` behavior remains unchanged.
