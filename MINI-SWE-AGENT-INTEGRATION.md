# Agent Interface Integration Plan

Date: 2026-07-05

## Goal

Add evaluation backends that can test whether Pangu's RefactorBench drop is
caused by an agent-interface mismatch, without breaking the existing SWE-agent
harness. The primary new backend should be a native OpenAI-compatible
`bash(command: string)` tool-calling runner. The already-added Mini-SWE-Agent
backend remains useful as a bash-only baseline, but it is not the closest match
to the model's served tool-calling interface.

Every backend must produce the same `preds.json` contract that
`scripts/score.py` already scores:

```json
{
  "task-id": {
    "model_name_or_path": "openai/pangu_auto",
    "instance_id": "task-id",
    "model_patch": "diff --git ..."
  }
}
```

This lets SWE-agent, Mini-SWE-Agent, and native bash-tool runs be compared with
the same scorer and report tooling.

## Current Harness Facts

- `scripts/run_model.py` is a SWE-agent wrapper. It loads
  `scripts/<variant>_instances.yaml`, runs `sweagent run-batch`, then calls
  `scripts/score.py`.
- `scripts/score.py` is already backend-agnostic enough for this plan: it only
  requires a `preds.json` with `instance_id` and `model_patch`.
- RefactorBench instance YAMLs are in SWE-agent expert-file format. The fields
  alternate backends need are already present:
  - task id: `problem_statement.id`
  - prompt text: `problem_statement.text`
  - repo name: `env.repo.repo_name`
  - image override target: `env.deployment.image`
- The default Docker image is now `rb-swerex:py311-tree-sitter`; `setup.sh`
  installs `swe-rex`, `tree-sitter==0.21.3`, and
  `tree-sitter-languages==1.10.2` into that image.
- The current SWE-agent prompt in `scripts/rb_agent.yaml` exposes bash plus
  SWE-agent edit tools, including `str_replace_editor`. The training data did
  not expose `str_replace_editor`, so the key interface test should use a
  single bash tool only.

## Training And Serving Facts Checked

The actual training file checked was:

```text
C:\Users\a84414458\Downloads\20260630_easy_medium_hard_refactoring_panguml2_slow_ml15_v3.json
```

Observed facts from that file:

- 2,381 trajectories.
- All observed task ids are `__lazy`; `lazy` evaluation is useful later, but is
  out of scope for this immediate plan.
- The meta prompt defines exactly one tool: `bash(command: string)`.
- There are 52,029 parsed tool calls, all named `bash`.
- There are no SWE-agent edit tools such as `str_replace_editor`.
- There are no Mini fenced commands such as `mswea_bash_command`.
- There are no OpenAI `tool_calls` fields in the raw file. Tool calls are
  serialized in assistant text with the model's special-token format, for
  example:

```text
[unused11]
[{"name": "bash", "arguments": {"command": "ls -la"}}]
[unused12]
```

Do not overfit the eval prompt to those raw special tokens. A live Pangu gateway
curl showed the served model can emit normal OpenAI-compatible `tool_calls`
when the request includes a `tools` definition. Therefore the correct
training-interface test is a native tool-calling runner with one `bash` tool,
not a runner that asks the model to literally print `[unused11]` and
`[unused12]`.

## Design Decision

Add a separate native bash-tool runner instead of replacing `scripts/run_model.py`.
Keep the Mini runner as a second baseline.

Reasons:

- It keeps the existing SWE-agent baseline stable.
- It allows apples-to-apples comparison between agent backends.
- It avoids mixing SWE-agent-specific arguments such as `--parse`,
  `--agent.model.litellm_model_registry`, and edit-tool bundles with bash-tool
  or Mini settings.
- It makes failures easier to attribute: same model, same tasks, same scorer,
  different agent interface.
- It tests the interface Pangu is actually served through: OpenAI-compatible
  `tools` and `tool_calls`.

Target new files:

- `scripts/run_common.py`
- `scripts/run_toolcall_model.py`
- `scripts/run_mini_model.py`
- `scripts/rb_mini_agent.yaml`

Target updates:

- `setup.sh`
- `SETUP.md`
- `HANDOVER.md`
- optionally `scripts/run_status.py`, `scripts/token_usage.py`,
  `scripts/plot_run_health.py`

## External Mini-SWE-Agent Facts Checked

Primary docs checked:

- Repository: https://github.com/SWE-agent/mini-swe-agent
- Quickstart: https://mini-swe-agent.com/latest/quickstart/
- Output files: https://mini-swe-agent.com/latest/usage/output_files/
- SWE-bench batch runner: https://mini-swe-agent.com/latest/usage/swebench/
- Docker environment reference:
  https://mini-swe-agent.com/latest/reference/environments/docker/
- YAML configuration:
  https://mini-swe-agent.com/latest/advanced/yaml_configuration/

Important implications for the Mini baseline:

- Mini-SWE-Agent is bash-oriented by default, so it is much closer than
  SWE-agent's edit-tool bundle.
- Mini's fenced `mswea_bash_command` action format is still not the training or
  serving interface seen for Pangu, so Mini should be treated as an intermediate
  bash-only baseline, not the final training-matched backend.
- Mini writes trajectory JSON files with a different schema than SWE-agent.
- Mini's SWE-bench tooling already understands `preds.json`/`model_patch`, but
  RefactorBench should not call Mini's SWE-bench runner directly because
  RefactorBench uses baked local repos such as `/flask_refactor` and
  `/django_refactor`, not SWE-bench instances.
- The Docker environment supports setting the container image, working
  directory, environment variables, timeout, and Docker run args. Those map
  directly to the current RefactorBench image and proxy/offline setup needs.

## Implementation Plan

### 1. Extract mandatory shared machinery into `scripts/run_common.py`

Before adding the native tool-call runner, move the verified backend-agnostic
pieces out of `scripts/run_mini_model.py` into `scripts/run_common.py` and make
both alternate runners import them. This is mandatory, not an optional cleanup:
the A/B is confounded if Mini and native tool-call runs collect patches, score,
resume, or compute control metrics differently.

Shared helpers should include:

- model/env/sampling/registry resolution;
- Docker image presence checks and startup helpers where backend-neutral;
- base tree hashing and precheck logic;
- command execution wrapper inputs shared across bash backends:
  `--command-timeout`, output truncation limit, output truncation format, and
  return-code/exception serialization;
- patch extraction:
  `git add -A -- .` then `git diff --cached --binary -- .`;
- per-task `result.json` writing and atomic `preds.json` rebuild with
  `os.replace`;
- scorer invocation with `--only`;
- resume/redo-existing logic;
- `control_metrics_from_actions` and the exact `control` dict schema stored in
  `preds.json`;
- rough accumulated-context token counting and `max_input_tokens` guardrail.

Do not let `run_toolcall_model.py` reimplement these pieces. It should differ
from `run_mini_model.py` in the model interaction layer only: native
OpenAI-compatible `tool_calls` versus Mini's fenced text action parser.

### 2. Add `scripts/run_toolcall_model.py`

Implement a custom RefactorBench batch runner that talks to the model through
the OpenAI-compatible chat-completions API with exactly one tool:

```json
{
  "type": "function",
  "function": {
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
      "type": "object",
      "required": ["command"],
      "properties": {
        "command": {
          "type": "string",
          "description": "The bash command to execute"
        }
      },
      "additionalProperties": false
    }
  }
}
```

This is the primary fix for the training/eval mismatch. Do not ask the model to
emit Mini fenced commands, SWE-agent edit tool calls, or literal `[unused11]`
special-token blocks. The gateway/vLLM parser should expose the model's tool
selection as normal `message.tool_calls`.

The loop per instance should be:

1. Start a Docker container from `rb-swerex:py311-tree-sitter` with working
   directory `/<repo>`.
2. Send a system prompt plus the RefactorBench task as a chat-completions
   request with the single `bash` tool above.
3. If the response has `tool_calls`, execute each `bash` command sequentially
   inside the container and append one `role: tool` message per call.
4. If the response has no `tool_calls` and `finish_reason == "stop"`, treat it
   as normal completion and extract the patch from git state.
5. If the response has no `tool_calls` and `finish_reason == "length"`, do not
   submit. Record `length_truncated`, send one concise correction/retry if
   budget remains, and otherwise fail the task with a truncation stop reason.
6. If the response has no `tool_calls` and any other `finish_reason`, treat it
   as a malformed/unknown completion, not as submission.
7. If the model calls `bash` with exactly
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, treat that as a backup
   completion marker and extract the patch from git state.
8. If the model calls an unknown tool, omits `command`, returns malformed JSON
   arguments, or produces unusable tool calls, record a format/tool-call error
   and feed back a concise correction message. Cap consecutive format errors.

The runner should support multiple tool calls in one assistant message because
the training data contains occasional multi-call assistant turns. Execute them
in order and preserve each API-provided `tool_call_id` in the corresponding
tool-result message.

Prompt requirements:

- Keep the system prompt short and aligned with the served tool interface: the
  model has one `bash` tool and should use it to inspect/edit files.
- Do not mention `str_replace_editor`, `mswea_bash_command`, or raw
  `[unused11]`/`[unused12]` tokens.
- The user prompt can reuse the existing RefactorBench task text and command
  rules, but should avoid requiring the sentinel as the only valid completion
  path. A final no-tool assistant answer is a valid stop condition for this
  backend only when the served `finish_reason` is `stop`.
- Keep the current safety rules from the Mini prompt where they matter for this
  benchmark: do not start servers, do not run full test suites, and do not edit
  tests unless explicitly requested.

CLI should mirror the useful parts of `scripts/run_model.py` and
`scripts/run_mini_model.py`:

```bash
python scripts/run_toolcall_model.py --model pangu --variant descriptive \
  --image rb-swerex:py311-tree-sitter \
  --workers 8 \
  --slug pangu-toolcall-attempt1 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100
```

Required arguments:

- `--model {claude,deepseek,glm,pangu}`
- `--variant {base,descriptive,lazy}`; `lazy` is supported by the runner but is
  out of scope for the immediate comparison
- `--slug`
- `--model-name`
- `--api-base`
- `--api-key`
- `--env-file`, default `scripts/models.env`
- `--sampling-file`, default `scripts/sampling.yaml`
- `--instances`, for smoke subsets
- `--image`, default `rb-swerex:py311-tree-sitter`
- `--workers`
- `--startup-timeout`
- `--command-timeout`
- `--docker-arg`, repeatable
- `--per-instance-call-limit`
- `--max-input-tokens`
- `--max-output-tokens`
- `--temperature`
- `--top-p`
- `--top-k`
- `--reasoning-effort`
- `--redo-existing`
- `--no-score`
- `--score-checkout`, default `local`
- `--dry-run`

Model/gateway behavior:

- Use the same `openai/<gateway-model-name>` naming convention and env/preset
  resolution as `scripts/run_model.py`.
- Send the OpenAI-compatible `tools` field on every model request.
- Before every request, enforce `max_input_tokens` with the shared
  accumulated-context guardrail from `run_common.py`; do not wait for gateway
  context errors.
- Preserve standard sampling and `extra_body` settings from
  `scripts/sampling.yaml`, including `top_k`, `min_p`,
  `repetition_penalty`, `frequency_penalty`, `presence_penalty`, and
  `reasoning_effort` when set.
- Capture `reasoning` / `reasoning_content` from responses when present, but do
  not require visible assistant prose before a tool call.
- Save `reasoning` / `reasoning_content` in the trajectory, but do not resend
  those fields in later chat history unless the gateway explicitly requires
  them. Resend assistant history as content plus `tool_calls` only.

Recommended output directory:

```text
runs/<slug>__<variant>/
  toolcall_run.config.yaml
  preds.json
  <instance_id>/
    <instance_id>.traj.json
    <instance_id>.debug.log
    patch.diff
    result.json
```

Trajectory JSON should store the full request/response control surface needed
for debugging: assistant content, reasoning fields, tool calls, tool outputs,
finish reason, stop reason, token usage, commands, return codes, output
truncation, timing, and format/tool-call errors.

The runner must write the same `control` dict schema into `preds.json` as the
Mini runner. `scripts/compare_agent_control.py` should be able to consume
tool-call runs without new trajectory parsing.

### 3. Pin and install Mini-SWE-Agent

Update `setup.sh` after the SWE-agent install step:

```bash
"$VPY" -m pip install -q mini-swe-agent==2.4.4
```

Do not put Mini-SWE-Agent inside the task Docker image unless the runner needs
it inside the container. The Mini runner runs on the host venv; the Docker image
is only the task execution environment.

Because company proxy issues have already affected container package installs,
verify the host venv can install Mini-SWE-Agent during setup. If this fails,
fix the host `pip`/proxy configuration before running any Mini experiment; do
not try to solve it from inside the task Docker container.

### 4. Keep `scripts/rb_mini_agent.yaml` for the Mini baseline

Create a Mini-specific prompt/config that mirrors the RefactorBench refactoring
prompt but exposes only bash actions.

Required prompt behavior:

- Tell the model it is doing refactoring, not bug fixing.
- Tell it the repository is already checked out at the working directory.
- Tell it not to edit tests unless the task explicitly asks for test edits.
- Tell it not to run the whole app or full test suite.
- Tell it to search, inspect, edit with shell commands, verify references, then
  submit.
- Tell it to review its changes with `git diff` if useful, but finish by
  emitting the completion marker only.

Submission instruction:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

Mini completion wiring is required, not optional. In Mini-SWE-Agent 2.4.4,
the Docker environment treats `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as a
submission sentinel when it is the first output line of a successful command.
`rb_mini_agent.yaml` must instruct the model to run exactly
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as the final command.
`scripts/run_mini_model.py` should also defensively subclass/wrap the Mini
Docker environment so this marker raises Mini's terminating `Submitted`
exception even if upstream behavior drifts. Without this hook, Mini may run to
the call cap and make normal completions look like repetition loops.

Important: do not include SWE-agent edit tools or any `str_replace_editor`
wording in this config. Mini is a bash-only baseline, not the exact
training-matched backend, because it still requires fenced
`mswea_bash_command` actions. Also do not rely on model-visible stdout to
transport the final patch; command output may be truncated by the agent layer.
The runner, not the model, must collect the patch from git state.

### 5. Keep `scripts/run_mini_model.py` for the Mini baseline

Implement a custom RefactorBench Mini batch runner. Do not use Mini's SWE-bench
runner directly.

CLI should mirror the useful parts of `scripts/run_model.py`:

```bash
python scripts/run_mini_model.py --model pangu --variant descriptive \
  --image rb-swerex:py311-tree-sitter \
  --workers 8 \
  --slug pangu-mini-attempt1 \
  --startup-timeout 1800 \
  --per-instance-call-limit 100
```

Required arguments:

- `--model {claude,deepseek,glm,pangu}`
- `--variant {base,descriptive,lazy}`
- `--slug`
- `--model-name`
- `--api-base`
- `--api-key`
- `--env-file`, default `scripts/models.env`
- `--sampling-file`, default `scripts/sampling.yaml`
- `--instances`, for smoke subsets
- `--image`, default `rb-swerex:py311-tree-sitter`
- `--workers`
- `--startup-timeout`
- `--command-timeout`, explicit per-shell-command timeout
- `--docker-arg`, repeatable
- `--per-instance-call-limit`
- `--max-input-tokens`
- `--max-output-tokens`
- `--temperature`
- `--top-p`
- `--top-k`
- `--reasoning-effort`
- `--redo-existing`
- `--no-score`
- `--score-checkout`, default `local`
- `--dry-run`

Reuse the same preset/env/sampling behavior as `scripts/run_model.py` through
`scripts/run_common.py`. The Mini runner and tool-call runner must use the same
shared helper implementations for backend-neutral behavior.

`--startup-timeout` is kept for CLI symmetry with `scripts/run_model.py`, but it
may not map to Mini's Docker environment if Mini has no separate container-start
timeout. If unsupported, warn clearly and treat it as a no-op. `--command-timeout`
is not optional: it must map to Mini's subprocess/shell-command timeout so a
model cannot wedge a worker by running the full test suite, starting a server,
or launching another long-running command.

Mini model configuration must preserve the same gateway and sampling semantics
as the SWE-agent runner:

- `model_name`: keep the `openai/<gateway-model-name>` form used by LiteLLM.
- API base/key: pass `RB_API_BASE` / `RB_API_KEY`, or the per-model
  `<PREFIX>_API_BASE` / `<PREFIX>_API_KEY`, into Mini's LiteLLM model config.
- Standard sampling: map `temperature`, `top_p`, and `max_output_tokens` to
  Mini's corresponding model/config fields. Enforce `max_input_tokens` in the
  runner with a conservative accumulated-context guardrail if Mini does not
  expose a native context cap.
- Extra body: pass through the same `extra_body` keys from
  `scripts/sampling.yaml`, including `top_k`, `min_p`, `repetition_penalty`,
  `frequency_penalty`, `presence_penalty`, and `reasoning_effort` when set.
- A/B rule: do not change sampling while changing backend. If
  `reasoning_effort` or `repetition_penalty` is enabled for Mini, enable the
  same setting for the SWE-agent comparison arm.

Recommended output directory:

```text
runs/<slug>__<variant>/
  mini_run.config.yaml
  preds.json
  <instance_id>/
    <instance_id>.traj.json
    <instance_id>.debug.log
    patch.diff
    result.json
```

### 6. Shared instance execution logic

For each instance from `scripts/<variant>_instances.yaml`:

1. Read:
   - `iid = instance["problem_statement"]["id"]`
   - `task_text = instance["problem_statement"]["text"]`
   - `repo = instance["env"]["repo"]["repo_name"]`
2. Set the Docker execution environment:
   - image: CLI `--image` or YAML deployment image
   - cwd: `"/" + repo`
   - env:
     - `PAGER=cat`
     - `MANPAGER=cat`
     - `LESS=-R`
     - `PIP_PROGRESS_BAR=off`
     - `TQDM_DISABLE=1`
     - `GIT_PAGER=cat`
     - `PYTHONSAFEPATH=1`
   - run args: from `--docker-arg`
   - offline behavior:
     - use the local `rb-swerex:py311-tree-sitter` image;
     - do not force an image pull;
     - fail loudly if the image is missing;
     - do not clone, fetch, or reset the repo from the network;
     - use the baked preexisting repo at `/<repo>`.
   - command timeout: set each backend's per-command timeout explicitly from
     `--command-timeout`; do not rely only on the prompt telling the model not
     to run the app or full test suite.
   - command-output truncation: both bash backends must use the same truncation
     limit and equivalent head/tail truncation format, matching Mini's current
     observation budget unless changed deliberately for both backends. Record
     the timeout and truncation settings in the run config and each task's
     `result.json`.
3. For the tool-call backend, build chat messages and expose only the native
   `bash(command: string)` tool.
4. For the Mini backend, build the prompt from `scripts/rb_mini_agent.yaml` and
   `task_text`.
5. Stop the tool-call backend when either the assistant returns no tool calls
   with `finish_reason == "stop"` or the model calls `bash` with
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Do not stop on no-tool
   `finish_reason == "length"`.
6. Stop the Mini loop as soon as the configured completion detector sees
   `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Treat the marker only as the stop
   signal, not as patch transport.
7. Before container teardown, extract the patch programmatically from git state:

   ```bash
   git add -A -- .
   git diff --cached --binary -- .
   ```

   This is mandatory. Plain `git diff -- .` omits untracked/new files, which
   would break RefactorBench tasks that create new modules. Do not parse the
   patch out of model-visible stdout; long diffs can be clipped mid-hunk.
   `--binary` is acceptable even though RefactorBench should not create binary
   files. If a binary diff ever appears, `score.py`'s `git apply` attempts can
   handle it, while the final GNU `patch` fallback may not. That is fine because
   binary patches are out of scope for these Python refactoring tasks.
8. Save the collected patch to `patch.diff`.
9. Write per-task artifacts.
10. Write per-task `result.json` first, including the shared `control` dict,
    then merge into `preds.json`.

Use per-task `result.json` files as the source of truth during the run. Build
or update `preds.json` by reading the union of all result files on disk, writing
to a temp file, then replacing the old file with `os.replace`. This avoids
corrupt JSON if multiple workers finish at the same time or the process is
interrupted. If duplicate result files exist for the same task, use
last-writer-wins based on file modification time.

The `control` dict written by Mini and the tool-call backend must have the same
field names and semantics so `scripts/compare_agent_control.py` can compare
them without backend-specific trajectory parsing.

### 7. Base tree alignment precheck

Before running tasks, verify that the container repo content matches the scorer
base for every repo in the selected instance set.

Do not compare commit SHA. The Docker image and `score.py --checkout local`
both create fresh local git commits, so commit hashes can differ even when file
content is identical. Compare git tree hashes instead:

```bash
git -C /<repo> rev-parse HEAD^{tree}
```

The expected tree hash should come from the same local checkout strategy that
`scripts/score.py --checkout local` uses: seed from `repositories/<repo>`, drop
VCS metadata, initialize git, add all files, and commit. If any repo tree hash
differs, fail before spending model calls. A base mismatch otherwise shows up
later as confusing `patch_apply_failed` or false task failures.

### 8. Budget and diagnostic parity

The comparison is only trustworthy if all compared backends use comparable
budgets and expose the failure modes being tested.

Budget policy:

- Use the same model-call cap for SWE-agent, Mini, and tool-call runs where
  possible.
- Log actual model calls, prompt tokens, completion tokens, total tokens, wall
  time, exit status, and patch byte length for every task.
- Do not pretend SWE-agent tool steps, Mini bash steps, and native bash tool
  calls are identical units. Report raw backend step count, executed command
  count, and model-call count.
- Keep per-command timeout and output truncation aligned between Mini and the
  native tool-call runner, or log any deliberate difference as an experiment
  variable.
- Keep sampling identical across arms. A backend comparison should not also
  introduce a new `reasoning_effort`, `repetition_penalty`, temperature, or
  output-token setting.

Loop metrics required from the first implementation:

- action duplicate rate: hash each submitted action/command after whitespace
  normalization and report the fraction of repeated actions;
- max repeated-action streak;
- empty-response rate, with interpretation caveat below;
- malformed-action or malformed-tool-call rate;
- stop reason: completed marker, call cap, timeout, runtime error, or unknown.

The comparison report should show pass rate and these loop/control metrics
together. The point is not just whether Mini or native tool calling changes the
score; it is whether the bash-only interfaces reduce the repetition,
format-error, and empty-output failures seen under SWE-agent.

Interpretation caveat: `empty-response rate` is interface-confounded in the
three-way comparison. Native tool-calling backends often have empty visible
assistant content by design, while Mini's text action format usually cannot.
Use duplicate-command/action rate, max repeated-action streak, call-cap rate,
timeout rate, and malformed-tool/action rate as the primary cross-backend loop
signals. Treat empty-response rate mainly as a within-native-backend health
signal.

### 9. Resume behavior

Default behavior should resume incomplete runs:

- If `preds.json` already contains a non-empty `model_patch` for an instance,
  skip that instance.
- If `<instance_id>/result.json` exists but `preds.json` is missing or stale,
  rebuild `preds.json` from result files before starting.
- `--redo-existing` forces rerun of completed instances.

This is important because 100-task RefactorBench runs are long and company
proxy/Docker issues can interrupt them.

### 10. Scoring integration

After either alternate backend finishes, call the existing scorer exactly like
`run_model.py`:

```bash
python scripts/score.py \
  --preds runs/<slug>__<variant>/preds.json \
  --out runs/<slug>__<variant>/scores.json \
  --variant <variant> \
  --checkout local \
  --model <model_name> \
  --only <comma-separated-instance-ids>
```

No scorer fork should be needed for the first implementation. If a Mini or
tool-call patch fails to apply, that is a model/agent output issue and should
be visible as `patch_apply_failed` in the normal score output.

`score.py --only` already computes `total.n` and per-repo denominators from the
filtered task ids, so smoke and 5-task runs are not diluted by the missing tasks.
Each alternate runner's built-in auto-score step must always pass `--only` with
the exact ids actually attempted in that run. Without `--only`, a smoke or
5-task `preds.json` would be scored against the full benchmark and reported as
missing the other tasks.

### 11. Status tooling updates

`scripts/run_status.py` currently expects SWE-agent `.traj` files and
`run_batch.config.yaml`.

Minimum acceptable first version:

- Tool-call runner and Mini runner work and score.
- `run_status.py` or a new comparison helper reports task status, step count,
  executed command count, model-call count, token totals when available, exit
  status, patch bytes, duplicate-action rate, max repeat streak,
  empty-response rate, malformed-tool/action rate, and stop reason.

Recommended follow-up:

- Detect tool-call runs by `toolcall_run.config.yaml`.
- Detect Mini runs by `mini_run.config.yaml` or `*.traj.json`.
- Parse tool-call and Mini trajectory step counts.
- Parse tool-call no-tool completion, tool-call sentinel completion, Mini exit
  status, and Mini completion marker.
- Parse token/call counts if present in each trajectory schema.
- Parse repeated-action and empty-response metrics for SWE-agent, Mini, and
  tool-call trajectories where possible.
- Reuse existing health flags:
  - `no_trajectory`
  - `empty_patch`
  - `patch_apply_failed`
  - `timeout`
  - `no_prediction`
- Add Mini-specific health flags only if there is clear evidence in artifacts.

### 12. Documentation updates

Update stale docs while adding Mini:

- Replace old `rb-swerex:py311` examples with
  `rb-swerex:py311-tree-sitter`.
- Add a "SWE-agent backend" section with current `scripts/run_model.py` command.
- Add a "Native bash tool-call backend" section with the new
  `scripts/run_toolcall_model.py` command.
- Add a "Mini-SWE-Agent backend" section with the
  `scripts/run_mini_model.py` command.
- Explicitly state that all backends write `preds.json` and use the same
  `scripts/score.py`.
- Explain that the native bash tool-call backend is the primary
  training-interface test for Pangu because the served model emits
  OpenAI-compatible `tool_calls`.
- Explain that Mini is a bash-only baseline, while SWE-agent remains the
  historical benchmark baseline.

## Validation Plan

### A. Static checks

```bash
python -m py_compile scripts/run_common.py scripts/run_toolcall_model.py scripts/run_mini_model.py
python scripts/run_toolcall_model.py --help
python scripts/run_mini_model.py --help
```

Confirm both runners expose `--startup-timeout`, `--command-timeout`,
`--per-instance-call-limit`, `--instances`, `--image`, `--redo-existing`, and
`--no-score`. Confirm `run_toolcall_model.py --help` documents native
OpenAI-compatible bash tool calls and no-tool assistant completion.

### B. Dependency check

```bash
source scripts/env.sh
python -m pip show mini-swe-agent
python -m pip show openai || python -m pip show litellm
python -m pip show pyyaml
```

The tool-call backend can use either the OpenAI SDK or LiteLLM, but it must send
the OpenAI-compatible `tools` field and read `message.tool_calls` from the
served response.

### C. Docker image check

```bash
docker run --rm rb-swerex:py311-tree-sitter \
  python3 -m pip show tree-sitter tree-sitter-languages
```

### D. Patch extraction check

Before using the model, validate the runner's patch collection on a synthetic
container edit that creates a new file:

```bash
docker run --rm rb-swerex:py311-tree-sitter /bin/sh -lc '
  cd /django_refactor &&
  mkdir -p django/apps &&
  printf "x = 1\n" > django/apps/rb_new_file_probe.py &&
  git add -A -- . &&
  git diff --cached --binary -- . | grep -q "new file mode"
'
```

This catches the `git diff -- .` failure mode before any alternate-backend run.

### E. Base tree check

Run both alternate runners in dry-run/precheck mode and confirm they verify tree
hashes for every repo in the selected instance set:

```bash
python scripts/run_toolcall_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug toolcall-smoke-precheck \
  --image rb-swerex:py311-tree-sitter \
  --dry-run

python scripts/run_mini_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug mini-smoke-precheck \
  --image rb-swerex:py311-tree-sitter \
  --dry-run
```

Expected: container tree hash equals local scorer-base tree hash for the smoke
repo.

### F. One-task smoke

```bash
python scripts/run_toolcall_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug toolcall-smoke \
  --image rb-swerex:py311-tree-sitter \
  --workers 1 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100
```

Expected for the native tool-call smoke:

- `runs/toolcall-smoke__descriptive/preds.json` exists.
- `model_patch` is non-empty.
- if the task creates a new file, the patch contains `new file mode`.
- `scores.json` exists.
- `scores.json.total.n` equals the number of actually-run ids, not 100.
- No trajectory mentions `str_replace_editor` or `mswea_bash_command`.
- The trajectory contains OpenAI-compatible `tool_calls` for bash commands.
- The stop reason is no-tool assistant completion with `finish_reason == "stop"`
  or sentinel completion, not call cap.
- Any no-tool response with `finish_reason == "length"` is recorded as
  `length_truncated` or retried; it is not treated as completed.
- `preds.json` contains the same `control` dict schema used by Mini.
- Tool-call command output is truncated with the same configured limit/shape as
  Mini observations.
- The status/comparison output reports duplicate-command rate, max repeat
  streak, empty-response rate, malformed-tool-call rate, stop reason, model
  calls, tokens when available, wall time, and patch bytes.

Check:

```bash
rg -n "str_replace_editor|edit_anthropic|review_on_submit|mswea_bash_command" \
  runs/toolcall-smoke__descriptive
```

Then keep the Mini smoke as the bash-only baseline:

```bash
python scripts/run_mini_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug mini-smoke \
  --image rb-swerex:py311-tree-sitter \
  --workers 1 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100
```

Expected for the Mini smoke:

- `runs/mini-smoke__descriptive/preds.json` exists.
- `model_patch` is non-empty.
- if the task creates a new file, the patch contains `new file mode`.
- `scores.json` exists.
- `scores.json.total.n` equals the number of actually-run ids, not 100.
- No trajectory mentions `str_replace_editor`.
- The trajectory stop reason is completed marker, not call cap.
- The status/comparison output reports duplicate-action rate, max repeat
  streak, empty-response rate, stop reason, model calls, tokens when available,
  wall time, and patch bytes.

Check:

```bash
rg -n "str_replace_editor|edit_anthropic|review_on_submit" \
  runs/mini-smoke__descriptive
```

### G. Small batch

Create a 5-task subset across different repos and run:

```bash
python scripts/run_toolcall_model.py --model pangu --variant descriptive \
  --instances scripts/mini_5_instances.yaml \
  --slug toolcall-5 \
  --image rb-swerex:py311-tree-sitter \
  --workers 2 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100

python scripts/run_mini_model.py --model pangu --variant descriptive \
  --instances scripts/mini_5_instances.yaml \
  --slug mini-5 \
  --image rb-swerex:py311-tree-sitter \
  --workers 2 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100
```

Validate:

```bash
python scripts/report.py runs/toolcall-smoke__descriptive/scores.json \
  runs/toolcall-5__descriptive/scores.json \
  runs/mini-smoke__descriptive/scores.json \
  runs/mini-5__descriptive/scores.json
```

### H. Resume test

1. Start a 5-task tool-call run.
2. Interrupt after one or two completed tasks.
3. Rerun the same command.
4. Confirm completed tasks are skipped and missing tasks continue.
5. Rerun with `--redo-existing` and confirm all selected tasks rerun.
6. Repeat once for Mini if Mini remains in the comparison.

### I. Comparison run

Run the same model, same instances, same sampling, same image, and same model
call cap with all three backends. Do not compare alternate backends against
historical scores when answering the interface-mismatch question; re-run the
SWE-agent arm fresh.

```bash
python scripts/run_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug swe-smoke \
  --image rb-swerex:py311-tree-sitter \
  --workers 1 \
  --startup-timeout 1800 \
  --per-instance-call-limit 100 \
  --parse auto

python scripts/run_toolcall_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug toolcall-smoke \
  --image rb-swerex:py311-tree-sitter \
  --workers 1 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100

python scripts/run_mini_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug mini-smoke \
  --image rb-swerex:py311-tree-sitter \
  --workers 1 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100
```

Then compare:

```bash
python scripts/report.py \
  runs/swe-smoke__descriptive/scores.json \
  runs/toolcall-smoke__descriptive/scores.json \
  runs/mini-smoke__descriptive/scores.json
```

Also run the loop/control comparison helper and report pass rate together with
duplicate-command/action rate, max repeat streak, empty-response rate,
malformed-tool/action rate, stop reason distribution, calls, tokens, and wall
time. The immediate comparison remains on `descriptive`; `lazy` is explicitly
deferred.

## Risks And Mitigations

- Shared-machinery divergence:
  - Make `scripts/run_common.py` mandatory for patch extraction, base checks,
    scoring, resume, output truncation, token guardrails, and control metrics.
  - Do not accept a tool-call runner that reimplements these pieces locally.
  - If a helper needs backend-specific behavior, pass a small explicit option
    and log it in the run config.
- Tool-call parser mismatch:
  - The raw training file uses special-token serialization, but the live gateway
    demonstrated OpenAI-compatible `tool_calls`. Validate this with a one-task
    smoke before any large run.
  - If the served response does not produce `message.tool_calls` for the
    `bash` tool, stop and debug gateway/parser configuration rather than
    falling back to Mini silently.
- Transient gateway failures:
  - Retry 429, 408, 5xx, URL errors, and request timeouts with bounded
    exponential backoff before failing a task.
  - Keep retry settings in the run config and control metrics. Without this,
    the native tool-call arm is less robust than SWE-agent/Mini under shared
    gateway load and the A/B can be biased by infrastructure errors.
- Incorrect stop condition in native tool-call runner:
  - Treat assistant responses with no tool calls as normal completion only when
    `finish_reason == "stop"`.
  - Treat `finish_reason == "length"` as truncation: retry once if budget
    remains or stop with `length_truncated`, but do not submit.
  - Also accept a `bash` call whose command is exactly
    `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  - Do not require the sentinel, because the training trajectories usually end
    with a normal no-tool assistant summary.
- Tool-call history incompatibility:
  - Persist `reasoning` / `reasoning_content` to trajectory files for analysis.
  - Do not resend those reasoning fields in the next request unless the gateway
    explicitly documents that it wants them.
  - Smoke-test the exact history format with the Pangu gateway before a batch.
- Mini-SWE-Agent API drift:
  - Pin the version after verifying the Linux eval host.
  - Keep Mini integration behind `scripts/run_mini_model.py`.
- Patch extraction mismatch:
  - Use completion markers and no-tool assistant completion only as stop
    signals.
  - Extract patches programmatically before container teardown with
    `git add -A -- .` and `git diff --cached --binary -- .`.
  - Save both `patch.diff` and raw trajectory for debugging.
- Completion marker not wired:
  - Configure Mini's finish/sentinel condition for
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, or implement the stop check in the
    runner after every step.
  - Validate smoke trajectories stop by completed marker rather than call cap.
- Hanging shell commands:
  - Set an explicit per-command timeout from `--command-timeout` in both
    alternate backends.
  - Treat `--startup-timeout` as best-effort only if Mini cannot map it.
- Parallel write corruption:
  - Write per-task `result.json`, rebuild `preds.json` from result files,
    write a temp file, then publish with `os.replace`.
- Base mismatch:
  - Compare git tree hashes between container repos and the local scorer base
    before model calls.
- Confounded A/B:
  - Keep sampling, model-call caps, image, instances, and scoring checkout the
    same across all compared arms.
  - Report loop/control metrics beside pass rate.
- Different prompt changes task difficulty:
  - Keep the native tool-call and Mini prompts semantically aligned with the
    RefactorBench task prompt.
  - The intended interface changes are limited to native `bash` tool calls for
    the primary test and Mini fenced bash for the baseline.
  - `lazy` evaluation is deferred for now; the immediate comparison remains on
    the same `descriptive` task slice.
- Model still may fail:
  - Native bash tool calls remove the SWE-agent edit-tool mismatch, but they do
    not guarantee higher scores. If failures remain loops or bad edits under
    native bash tool calls, that is stronger evidence of a model/control issue.

## Definition Of Done

- `bash setup.sh` installs Mini-SWE-Agent in the host venv.
- `scripts/run_common.py` exists and owns shared patch extraction, base-tree
  precheck, env/sampling resolution, scoring, resume, output truncation,
  token guardrails, `preds.json` rebuild, and control-metric computation.
- `scripts/run_toolcall_model.py --help` works.
- `scripts/run_mini_model.py --help` works.
- `scripts/run_toolcall_model.py` sends exactly one OpenAI-compatible tool,
  `bash(command: string)`, and reads returned `message.tool_calls`.
- The tool-call runner retries transient gateway 429/408/5xx/timeouts with
  bounded backoff and records retry counts in control metrics.
- The tool-call runner handles multiple bash tool calls in one assistant turn.
- The tool-call runner treats no-tool assistant responses as completion only
  when `finish_reason == "stop"`.
- The tool-call runner never submits on no-tool `finish_reason == "length"`;
  it records/retries truncation and otherwise exits with `length_truncated`.
- The tool-call runner accepts `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as
  a backup stop marker.
- The tool-call runner strips `reasoning` / `reasoning_content` from re-sent
  assistant history unless the gateway explicitly requires those fields.
- `rb_mini_agent.yaml` or `scripts/run_mini_model.py` wires
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as an actual Mini stop condition.
- Both alternate runners enforce an explicit per-command timeout.
- Both alternate runners use the same configured command-output truncation
  limit and equivalent head/tail truncation format.
- Both alternate runners extract patches with `git add -A -- .` and
  `git diff --cached --binary -- .`, so new files are included.
- Neither alternate runner parses the final patch from model-visible stdout.
- Both alternate runners verify base tree-hash alignment before spending model
  calls.
- Both alternate runners enforce `max_input_tokens` before each model request.
- Both alternate runners write the same `control` dict schema into `preds.json`.
- A one-task native tool-call smoke run produces a non-empty `preds.json`.
- A one-task Mini smoke run produces a non-empty `preds.json`.
- The existing `scripts/score.py` scores both alternate smoke runs.
- The built-in auto-score path passes `--only` for the actually-run task ids.
- Resume works after interruption.
- Docs show SWE-agent, native bash tool-call, and Mini-SWE-Agent commands.
- SWE-agent baseline commands still work unchanged.
- A/B on the same task slice reports pass rate plus duplicate-action rate,
  max repeated-action streak, empty-response rate, malformed-tool/action rate,
  stop reasons, model calls, tokens when available, wall time, and patch size
  under comparable budgets.
- The immediate acceptance comparison uses the same `descriptive` instances;
  `lazy` is not required for this phase.
