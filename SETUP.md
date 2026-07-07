# RefactorBench eval-harness — setup on a fresh machine

The multi-model SWE-agent harness lives under `scripts/`. Bootstrapping is
automated by `setup.sh`; on a new machine it's basically **clone → `setup.sh` →
fill in keys**.

## Prerequisites

- **git**, **Docker** (daemon running), network access
- **Python ≥ 3.11** (for SWE-agent)
- Ideally also a **Python ≤ 3.11** (e.g. 3.10) — the AST checkers use
  `ast.Str` / `ast.NameConstant`, which were **removed in Python 3.12**.
  `setup.sh` auto-detects a suitable checker interpreter and records it as
  `RB_TEST_PYTHON`.

## Steps

### 1. Clone the fork (the `eval-harness` branch)

```bash
git clone -b eval-harness https://github.com/sail-amir/RefactorBench.git
cd RefactorBench
```

This includes everything committed: the harness scripts, `scripts/litellm_registry.json`,
`scripts/{base,lazy}_instances.yaml`, the vendored `repositories/`, `tests/`,
`problems/`, and `setup.sh`.

### 2. Bootstrap

```bash
bash setup.sh
```

Idempotent. It:
- creates `.venv` and installs **SWE-agent v1.1.0** (+ streaming/reasoning-capture patch),
- installs **Mini-SWE-Agent** for the bash-only backend,
- builds the **`rb-swerex:py311-tree-sitter`** Docker image with the **vendored `repositories/`
  baked in** (each at `/<repo>`, as a fresh git repo) so tasks need **no GitHub
  access** — see "Offline mode" below,
- generates the base/lazy instance yamls and a 1-task `smoke_instances.yaml`,
- copies `scripts/models.env.example → scripts/models.env`,
- writes `scripts/env.sh`.

> The image build skips if `rb-swerex:py311-tree-sitter` already exists — run
> `docker rmi rb-swerex:py311-tree-sitter && bash setup.sh` to rebuild with the baked repos.

Override defaults via env vars, e.g. `SWE_AGENT_COMMIT=<sha> IMAGE=rb-swerex:py311-tree-sitter bash setup.sh`.

If the machine is already bootstrapped and you only need to add/verify the
Mini-SWE-Agent backend, use the lighter helper instead:

```bash
bash setup_mini.sh
```

It installs the pinned Mini-SWE-Agent package into `.venv`, verifies the Mini
imports used by `scripts/run_mini_model.py`, checks the Docker image has
tree-sitter packages, and ensures `scripts/smoke_instances.yaml` exists.

### 3. Fill in gateway credentials

`scripts/models.env` is **gitignored** (secrets never leave the machine), so set
it on each machine. Copy the values from your other machine's `scripts/models.env`:

```
RB_API_BASE=...        # shared OpenAI-compatible gateway base URL (…/v1)
RB_API_KEY=...         # shared token (claude/deepseek/glm)
PANGU_API_KEY=...      # pangu's different token (overrides RB_API_KEY for pangu)
CLAUDE_MODEL=openai/...
DEEPSEEK_MODEL=openai/...
GLM_MODEL=openai/...
PANGU_MODEL=openai/...
```

Keep the `openai/` prefix on every `*_MODEL` — that routes litellm to `RB_API_BASE`.

### 4. Activate the environment

```bash
source scripts/env.sh    # puts .venv on PATH, exports RB_TEST_PYTHON
```

### 4b. Verify the gateway (recommended)

Before spending on a SWE-agent run, confirm the endpoint/key/model name work:

```bash
python scripts/check_gateway.py --model deepseek --thinking
```

It probes the gateway (resolving the same `models.env` config) for: basic chat
(reachable + auth + model name), native `tool_calls` (function_calling), and —
with `--thinking` — a `reasoning_content` channel. Exit 0 = healthy. Use
`--model claude|glm|pangu` or `--model-name openai/<name>` for others.

### 5. SWE-agent smoke test (`add-log-parameter-get-debug-flag-task`)

```bash
python scripts/run_model.py --model deepseek --variant descriptive \
  --instances scripts/smoke_instances.yaml --slug smoke \
  --image rb-swerex:py311-tree-sitter --startup-timeout 1200 --workers 1
```

### 6. Native bash tool-call smoke test (`add-log-parameter-get-debug-flag-task`)

Use this backend to test models served with OpenAI-compatible `tools` /
`tool_calls` and a single `bash(command)` tool:

```bash
python scripts/run_toolcall_model.py --model deepseek --variant descriptive \
  --instances scripts/smoke_instances.yaml --slug toolcall-smoke \
  --image rb-swerex:py311-tree-sitter \
  --startup-timeout 1200 --command-timeout 30 --workers 1
```

### 7. Mini-SWE-Agent smoke test (`add-log-parameter-get-debug-flag-task`)

Use this as the bash-only Mini baseline:

```bash
python scripts/run_mini_model.py --model deepseek --variant descriptive \
  --instances scripts/smoke_instances.yaml --slug mini-smoke \
  --image rb-swerex:py311-tree-sitter \
  --startup-timeout 1200 --command-timeout 30 --workers 1
```

To test a Mini prompt closer to the bash-tool training traces, keep the same
runner and use plain fenced `bash` actions instead of Mini's
`mswea_bash_command` tag:

```bash
python scripts/run_mini_model.py --model deepseek --variant descriptive \
  --instances scripts/smoke_instances.yaml --slug mini-bash-smoke \
  --image rb-swerex:py311-tree-sitter \
  --startup-timeout 1200 --command-timeout 30 --workers 1 \
  --agent-config scripts/rb_mini_agent_plain_bash.yaml
```

All backends write `runs/<slug>__<variant>/preds.json` and use the same
`scripts/score.py`.

### 8. Full run + report

```bash
# one model × one variant (100 tasks)
python scripts/run_model.py --model deepseek --variant descriptive \
  --image rb-swerex:py311-tree-sitter --startup-timeout 1200 --workers 4

# same model through native bash tool-calls
python scripts/run_toolcall_model.py --model deepseek --variant descriptive \
  --image rb-swerex:py311-tree-sitter \
  --startup-timeout 1200 --command-timeout 30 --workers 4

# same model through Mini-SWE-Agent
python scripts/run_mini_model.py --model deepseek --variant descriptive \
  --image rb-swerex:py311-tree-sitter \
  --startup-timeout 1200 --command-timeout 30 --workers 4

# cross-model / cross-variant grid
python scripts/report.py runs/*/scores.json

# pass rate + loop/control metrics
python scripts/compare_agent_control.py runs/*/scores.json

# radar chart by refactoring type; fixed 13-type order, frontier/hero/baseline styling
python scripts/plot_radar_by_type.py \
  runs/glm-5-1__descriptive/scores.json \
  runs/pangu-7b-5k-refactoring-mini-full__descriptive/scores.json \
  runs/pangu-35b__descriptive/scores.json \
  runs/pangu-7b__descriptive/scores.json \
  --labels GLM-5.1 pangu-7b-refactoring pangu-35b pangu-7b \
  --out runs/refactorbench_radar.png
```

### 9. Inspect a run (status + health)

After a smoke or full run, get a per-task status/health report:

```bash
python scripts/run_status.py smoke_pangu35b      # slug (searches runs/)
python scripts/run_status.py                     # most recent run
```

It reads the trajectory + preds + scores + logs and reports, per task: parser
used, exit status, step count, patch size, score (PASS/fail + reason), tokens,
wall-clock, and HEALTH flags. It separates **health** (did the job run cleanly)
from **score** (did the model solve it) — e.g. `healthy` + `fail` means the
harness worked and the model simply missed the task. Flags like `runtime_failed`,
`empty_patch`, or `no_trajectory` point at real infra/agent problems; a
first-attempt container retry shows as the informational note `slow_start`.

## Notes

- **Native function-calling works out of the box.** `scripts/litellm_registry.json`
  is committed and `run_model.py` passes it via `--agent.model.litellm_model_registry`
  by default, marking the gateway models `supports_function_calling`. So **omit
  `--parse thought_action`** to use native tool-calls (the better path). Add
  `--parse thought_action` only as a fallback.
- **`--startup-timeout`** matters only on a heavily loaded host (container start
  can exceed the 180s default). On a quiet machine, drop it — runs are much
  faster (~6 min/task vs ~17 min under load).
- **`--reasoning-effort high`** enables thinking models (e.g. `deepseek-v3.2`);
  slower but typically higher quality.
- **Offline mode (no GitHub access).** The instance yamls use the
  `preexisting` repo type with `reset: false`, pointing at the repos **baked into
  the image** at `/<repo>` — so the agent never clones or `git fetch`es from
  GitHub (this avoids the per-task `Connection timed out` failures on
  restricted-network hosts). Scoring matches: **`run_model.py` defaults to
  `--score-checkout local`** (the bundled `repositories/`, same base the agent
  saw). Pass `--score-checkout fork` only if you want a fresh `dhruvji/*` clone
  and have network. Nothing in a run touches GitHub by default.
- **Refactoring prompt (paper-faithful).** `run_model.py` passes
  `--config scripts/rb_agent.yaml` by default — SWE-agent's `default.yaml` with the
  prompt adapted for refactoring per RefactorBench App. B.1: the "create a
  reproduction script and run it" steps are removed (these are refactors, not bug
  fixes, so such scripts launch the app/server and hang on swe-rex's 30s command
  timeout), and the agent is told not to run the app/tests. Pass `--agent-config ""`
  to fall back to SWE-agent's stock bug-fixing prompt.
- **Native bash tool-call backend.** `run_toolcall_model.py` sends an
  OpenAI-compatible `tools` request with exactly one `bash(command)` tool,
  executes returned `message.tool_calls` in the task container, and stops on a
  no-tool assistant response with `finish_reason == "stop"` or the sentinel
  command. It shares patch extraction, scoring, resume, base-tree checks, and
  control metrics with Mini through `scripts/run_common.py`. It retries
  transient gateway 429/408/5xx/timeouts with bounded backoff. Add
  `--require-submit-marker` to test whether Mini-style explicit submission
  improves stopping behavior for models that otherwise keep calling tools.
- **Mini-SWE-Agent backend.** `run_mini_model.py` uses `scripts/rb_mini_agent.yaml`,
  a bash-only prompt with no SWE-agent edit-tool wording. For a prompt-format
  experiment closer to bash-tool training traces, pass
  `--agent-config scripts/rb_mini_agent_plain_bash.yaml`; this keeps the same
  Mini runner but asks for plain fenced `bash` actions instead of the
  `mswea_bash_command` tag. It stops on
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, then the runner extracts the patch
  from git state with `git add -A -- . && git diff --cached --binary -- .` so
  newly created files are included. The built-in score step passes `--only` for
  the actually-run instances, so smoke/subset denominators are correct.
- **Models:** presets are `claude | deepseek | glm | pangu`; variants are
  `base | descriptive | lazy`. Trajectories are saved per task under
  `runs/<slug>__<variant>/<id>/<id>.traj` (gitignored — local only).
