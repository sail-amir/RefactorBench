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
- creates `.venv` and installs **SWE-agent v1.1.0** (+ optional streaming patch),
- builds the **`rb-swerex:py311`** Docker image (swe-rex preinstalled so
  containers start fast),
- generates the base/lazy instance yamls and a 1-task `smoke_instances.yaml`,
- copies `scripts/models.env.example → scripts/models.env`,
- writes `scripts/env.sh`.

Override defaults via env vars, e.g. `SWE_AGENT_REF=main IMAGE=rb-swerex:py311 bash setup.sh`.

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

### 5. Smoke test (1 flask task)

```bash
python scripts/run_model.py --model deepseek --variant descriptive \
  --instances scripts/smoke_instances.yaml --slug smoke \
  --image rb-swerex:py311 --startup-timeout 1200 --workers 1
```

### 6. Full run + report

```bash
# one model × one variant (100 tasks)
python scripts/run_model.py --model deepseek --variant descriptive \
  --image rb-swerex:py311 --startup-timeout 1200 --workers 4

# cross-model / cross-variant grid
python scripts/report.py runs/*/scores.json
```

### 7. Inspect a run (status + health)

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
- **Scoring** defaults to `--checkout fork` (clones `dhruvji/*_refactor`). Use
  `--checkout local` to score against the bundled `repositories/` copies offline.
- **Models:** presets are `claude | deepseek | glm | pangu`; variants are
  `base | descriptive | lazy`. Trajectories are saved per task under
  `runs/<slug>__<variant>/<id>/<id>.traj` (gitignored — local only).
