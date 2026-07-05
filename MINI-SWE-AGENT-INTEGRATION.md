# Mini-SWE-Agent Integration Plan

Date: 2026-07-05

## Goal

Add Mini-SWE-Agent as a second evaluation backend for RefactorBench without
breaking the existing SWE-agent harness. The Mini backend should produce the
same `preds.json` contract that `scripts/score.py` already scores:

```json
{
  "task-id": {
    "model_name_or_path": "openai/pangu_auto",
    "instance_id": "task-id",
    "model_patch": "diff --git ..."
  }
}
```

This lets old SWE-agent runs and new Mini-SWE-Agent runs be compared with the
same scorer and report tooling.

## Current Harness Facts

- `scripts/run_model.py` is a SWE-agent wrapper. It loads
  `scripts/<variant>_instances.yaml`, runs `sweagent run-batch`, then calls
  `scripts/score.py`.
- `scripts/score.py` is already backend-agnostic enough for this plan: it only
  requires a `preds.json` with `instance_id` and `model_patch`.
- RefactorBench instance YAMLs are in SWE-agent expert-file format. The fields
  Mini needs are already present:
  - task id: `problem_statement.id`
  - prompt text: `problem_statement.text`
  - repo name: `env.repo.repo_name`
  - image override target: `env.deployment.image`
- The default Docker image is now `rb-swerex:py311-tree-sitter`; `setup.sh`
  installs `swe-rex`, `tree-sitter==0.21.3`, and
  `tree-sitter-languages==1.10.2` into that image.
- The current SWE-agent prompt in `scripts/rb_agent.yaml` exposes bash plus
  SWE-agent edit tools, including `str_replace_editor`. Mini-SWE-Agent should
  intentionally use bash-only interaction to match Mini training traces better.

## Design Decision

Add a separate Mini runner instead of replacing `scripts/run_model.py`.

Reasons:

- It keeps the existing SWE-agent baseline stable.
- It allows apples-to-apples comparison between agent backends.
- It avoids mixing SWE-agent-specific arguments such as `--parse`,
  `--agent.model.litellm_model_registry`, and edit-tool bundles with Mini
  settings.
- It makes failures easier to attribute: same model, same tasks, same scorer,
  different agent interface.

Target new files:

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

Important implications:

- Mini-SWE-Agent is bash-oriented by default, which is the point of this
  integration.
- Mini writes trajectory JSON files with a different schema than SWE-agent.
- Mini's SWE-bench tooling already understands `preds.json`/`model_patch`, but
  RefactorBench should not call Mini's SWE-bench runner directly because
  RefactorBench uses baked local repos such as `/flask_refactor` and
  `/django_refactor`, not SWE-bench instances.
- The Docker environment supports setting the container image, working
  directory, environment variables, timeout, and Docker run args. Those map
  directly to the current RefactorBench image and proxy/offline setup needs.

## Implementation Plan

### 1. Pin and install Mini-SWE-Agent

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

### 2. Add `scripts/rb_mini_agent.yaml`

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
wording in this config. The goal is to evaluate the model under the interface it
was trained on. Also do not rely on model-visible stdout to transport the final
patch; command output may be truncated by the agent layer. The runner, not the
model, must collect the patch from git state.

### 3. Add `scripts/run_mini_model.py`

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

Reuse the same preset/env/sampling behavior as `scripts/run_model.py`.
Implementation can either import shared helpers from `run_model.py` or move the
common code into a small `scripts/run_common.py`.

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

### 4. Instance execution logic

For each instance from `scripts/<variant>_instances.yaml`:

1. Read:
   - `iid = instance["problem_statement"]["id"]`
   - `task_text = instance["problem_statement"]["text"]`
   - `repo = instance["env"]["repo"]["repo_name"]`
2. Set Mini Docker environment:
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
   - command timeout: set Mini's per-command timeout explicitly from
     `--command-timeout`; do not rely only on the prompt telling the model not
     to run the app or full test suite.
3. Build the prompt from `scripts/rb_mini_agent.yaml` and `task_text`.
4. Run Mini-SWE-Agent for that task.
5. Stop the Mini loop as soon as the configured completion detector sees
   `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Treat the marker only as the stop
   signal, not as patch transport.
6. Before container teardown, extract the patch programmatically from git state:

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
7. Save the collected patch to `patch.diff`.
8. Write per-task artifacts.
9. Write per-task `result.json` first, then merge into `preds.json`.

Use per-task `result.json` files as the source of truth during the run. Build
or update `preds.json` by reading the union of all result files on disk, writing
to a temp file, then replacing the old file with `os.replace`. This avoids
corrupt JSON if multiple workers finish at the same time or the process is
interrupted. If duplicate result files exist for the same task, use
last-writer-wins based on file modification time.

### 5. Base tree alignment precheck

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

### 6. Budget and diagnostic parity

The A/B is only trustworthy if both backends use comparable budgets and expose
the failure modes being tested.

Budget policy:

- Use the same model-call cap for SWE-agent and Mini where possible.
- Log actual model calls, prompt tokens, completion tokens, total tokens, wall
  time, exit status, and patch byte length for every task.
- Do not pretend SWE-agent tool steps and Mini bash steps are identical units.
  Report both raw backend step count and model-call count.
- Keep sampling identical across arms. A backend comparison should not also
  introduce a new `reasoning_effort`, `repetition_penalty`, temperature, or
  output-token setting.

Loop metrics required from the first implementation:

- action duplicate rate: hash each submitted action/command after whitespace
  normalization and report the fraction of repeated actions;
- max repeated-action streak;
- empty-response rate;
- malformed-action rate, if Mini exposes parse/action errors;
- stop reason: completed marker, call cap, timeout, runtime error, or unknown.

The comparison report should show pass rate and these loop/control metrics
together. The point is not just whether Mini changes the score; it is whether it
reduces the repetition and empty-output failures seen under SWE-agent.

### 7. Resume behavior

Default behavior should resume incomplete runs:

- If `preds.json` already contains a non-empty `model_patch` for an instance,
  skip that instance.
- If `<instance_id>/result.json` exists but `preds.json` is missing or stale,
  rebuild `preds.json` from result files before starting.
- `--redo-existing` forces rerun of completed instances.

This is important because 100-task RefactorBench runs are long and company
proxy/Docker issues can interrupt them.

### 8. Scoring integration

After Mini finishes, call the existing scorer exactly like `run_model.py`:

```bash
python scripts/score.py \
  --preds runs/<slug>__<variant>/preds.json \
  --out runs/<slug>__<variant>/scores.json \
  --variant <variant> \
  --checkout local \
  --model <model_name> \
  --only <comma-separated-instance-ids>
```

No scorer fork should be needed for the first implementation. If a Mini patch
fails to apply, that is a model/agent output issue and should be visible as
`patch_apply_failed` in the normal score output.

`score.py --only` already computes `total.n` and per-repo denominators from the
filtered task ids, so smoke and 5-task runs are not diluted by the missing tasks.
The Mini runner's built-in auto-score step must always pass `--only` with the
exact ids actually attempted in that run. Without `--only`, a smoke or 5-task
`preds.json` would be scored against the full benchmark and reported as missing
the other tasks.

### 9. Status tooling updates

`scripts/run_status.py` currently expects SWE-agent `.traj` files and
`run_batch.config.yaml`.

Minimum acceptable first version:

- Mini runner works and scores.
- `run_status.py` or a new comparison helper reports Mini task status, step
  count, model-call count, token totals when available, exit status, patch
  bytes, duplicate-action rate, max repeat streak, empty-response rate, and
  stop reason.

Recommended follow-up:

- Detect Mini runs by `mini_run.config.yaml` or `*.traj.json`.
- Parse Mini trajectory step count.
- Parse Mini exit status/completion marker.
- Parse token/call counts if present in the Mini trajectory schema.
- Parse repeated-action and empty-response metrics for both SWE-agent and Mini
  trajectories where possible.
- Reuse existing health flags:
  - `no_trajectory`
  - `empty_patch`
  - `patch_apply_failed`
  - `timeout`
  - `no_prediction`
- Add Mini-specific health flags only if there is clear evidence in artifacts.

### 10. Documentation updates

Update stale docs while adding Mini:

- Replace old `rb-swerex:py311` examples with
  `rb-swerex:py311-tree-sitter`.
- Add a "SWE-agent backend" section with current `scripts/run_model.py` command.
- Add a "Mini-SWE-Agent backend" section with the new
  `scripts/run_mini_model.py` command.
- Explicitly state that both backends write `preds.json` and use the same
  `scripts/score.py`.
- Explain that Mini is intended for models trained on Mini-SWE-Agent
  trajectories, while SWE-agent remains the historical benchmark baseline.

## Validation Plan

### A. Static checks

```bash
python -m py_compile scripts/run_mini_model.py
python scripts/run_mini_model.py --help
```

Confirm `--help` exposes both `--startup-timeout` and `--command-timeout`, and
that the implementation warns if `--startup-timeout` cannot be mapped to Mini.

### B. Dependency check

```bash
source scripts/env.sh
python -m pip show mini-swe-agent
python -m pip show pyyaml
```

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

This catches the `git diff -- .` failure mode before any Mini run.

### E. Base tree check

Run the Mini runner in dry-run/precheck mode and confirm it verifies tree hashes
for every repo in the selected instance set:

```bash
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
python scripts/run_mini_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug mini-smoke \
  --image rb-swerex:py311-tree-sitter \
  --workers 1 \
  --startup-timeout 1800 \
  --command-timeout 30 \
  --per-instance-call-limit 100
```

Expected:

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
python scripts/report.py runs/mini-smoke__descriptive/scores.json \
  runs/mini-5__descriptive/scores.json
```

### H. Resume test

1. Start a 5-task Mini run.
2. Interrupt after one or two completed tasks.
3. Rerun the same command.
4. Confirm completed tasks are skipped and missing tasks continue.
5. Rerun with `--redo-existing` and confirm all selected tasks rerun.

### I. Comparison run

Run the same model, same instances, same sampling, same image, and same model
call cap with both backends. Do not compare Mini against historical scores when
answering the interface-mismatch question; re-run the SWE-agent arm fresh.

```bash
python scripts/run_model.py --model pangu --variant descriptive \
  --instances scripts/smoke_instances.yaml \
  --slug swe-smoke \
  --image rb-swerex:py311-tree-sitter \
  --workers 1 \
  --startup-timeout 1800 \
  --per-instance-call-limit 100 \
  --parse auto

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
  runs/mini-smoke__descriptive/scores.json
```

Also run the loop/control comparison helper and report pass rate together with
duplicate-action rate, max repeat streak, empty-response rate, stop reason
distribution, calls, tokens, and wall time.

## Risks And Mitigations

- Mini-SWE-Agent API drift:
  - Pin the version after verifying the Linux eval host.
  - Keep Mini integration behind `scripts/run_mini_model.py`.
- Patch extraction mismatch:
  - Use the completion marker only as a stop signal.
  - Extract patches programmatically before container teardown with
    `git add -A -- .` and `git diff --cached --binary -- .`.
  - Save both `patch.diff` and raw trajectory for debugging.
- Completion marker not wired:
  - Configure Mini's finish/sentinel condition for
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, or implement the stop check in the
    runner after every step.
  - Validate smoke trajectories stop by completed marker rather than call cap.
- Hanging shell commands:
  - Set an explicit Mini per-command timeout from `--command-timeout`.
  - Treat `--startup-timeout` as best-effort only if Mini cannot map it.
- Parallel write corruption:
  - Write per-task `result.json`, rebuild `preds.json` from result files,
    write a temp file, then publish with `os.replace`.
- Base mismatch:
  - Compare git tree hashes between container repos and the local scorer base
    before model calls.
- Confounded A/B:
  - Keep sampling, model-call caps, image, instances, and scoring checkout the
    same across both arms.
  - Report loop/control metrics beside pass rate.
- Different prompt changes task difficulty:
  - Keep the Mini prompt semantically aligned with `scripts/rb_agent.yaml`.
  - The only intended interface change is Mini bash-only interaction.
- Model still may fail:
  - Mini backend removes the SWE-agent tool mismatch, but it does not guarantee
    higher scores. If failures remain loops or bad edits under bash-only Mini,
    that is stronger evidence of a model/control issue.

## Definition Of Done

- `bash setup.sh` installs Mini-SWE-Agent in the host venv.
- `scripts/run_mini_model.py --help` works.
- `rb_mini_agent.yaml` or `scripts/run_mini_model.py` wires
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as an actual Mini stop condition.
- The runner enforces an explicit per-command timeout.
- The runner extracts patches with `git add -A -- .` and
  `git diff --cached --binary -- .`, so new files are included.
- The runner never parses the final patch from model-visible stdout.
- The runner verifies base tree-hash alignment before spending model calls.
- A one-task Mini smoke run produces a non-empty `preds.json`.
- The existing `scripts/score.py` scores the Mini smoke run.
- The built-in auto-score path passes `--only` for the actually-run task ids.
- Resume works after interruption.
- Docs show both SWE-agent and Mini-SWE-Agent commands.
- SWE-agent baseline commands still work unchanged.
- A/B on the same task slice reports pass rate plus duplicate-action rate,
  max repeated-action streak, empty-response rate, stop reasons, model calls,
  tokens when available, wall time, and patch size under comparable budgets.
