# RefactorBench eval-harness — session handoff / resume notes

Snapshot of where the work stands so it can be picked up after a restart.
Branch: **`eval-harness`** (fork `sail-amir/RefactorBench`, in sync with `fork/eval-harness`).
Everything below is committed + pushed unless noted.

## What this is
A multi-model agent harness for **RefactorBench** (100 multi-file refactoring
tasks across 9 OSS repos), pointed at a shared **OpenAI-compatible gateway**.
It supports the original SWE-agent backend, a native bash tool-call backend,
and a bash-only Mini-SWE-Agent backend.
Presets: `claude | deepseek | glm | pangu`. Variants: `base | descriptive | lazy`.
Harness lives in `scripts/`; bootstrap with `setup.sh` (see `SETUP.md`).

## Two machines (shared `/shared_workspace_mfs` mount)
- **This box** (`/shared_workspace_mfs/amir/replications/RefactorBench`): dev/edits.
  **Docker is NOT accessible from here** — can't run SWE-agent locally.
- **`a84414458` (pangu-eval-dev)** (`/home/a84414458/sources/RefactorBench`): runs the
  actual jobs (pangu at `http://10.170.22.221:7776`). **Restricted network → GitHub
  unreachable from containers**, which is why we made the harness fully offline.

## State of the run pipeline (key facts)
- **Offline repos**: the `rb-swerex:py311-tree-sitter` image **bakes the vendored `repositories/`**
  at `/<repo>` (fresh git repo, base commit). Instances use `type: preexisting` +
  `reset: false`, so **no clone, no `git fetch`** — fixes the `Connection timed out`
  failures. Each task = a **fresh container** (`remove_container=True`) → pristine repo;
  no in-container reset needed.
  - ⚠️ To pick up baked repos you must rebuild: `docker rmi rb-swerex:py311-tree-sitter && bash setup.sh`
    (or build directly; see SETUP.md "Offline mode").
- **Scoring** defaults to `--score-checkout local` (bundled `repositories/`, same base
  the agent saw, offline). `fork` clones from GitHub (needs network).
- **Caps** live in `scripts/sampling.yaml` (not the registry): `max_output_tokens: 32000`,
  `max_input_tokens: 100000` (defaults; per-model overridable). Registry holds only
  capability/cost metadata.
- **Prompt**: `run_model.py` passes `--config scripts/rb_agent.yaml` (refactoring-tuned,
  **no reproduction-script step** — paper App. B.1) → avoids the `CommandTimeoutError`
  from agents running the app/server. `--agent-config ""` reverts to stock.
- **Reasoning capture**: `scripts/sweagent-streaming.patch` captures `reasoning_content`
  from streaming chunks onto each **trajectory step**; `token_usage.py` counts it.
  - ⚠️ `setup.sh`'s "already patched" check skips re-applying if `stream_chunk_builder`
    is present. To force the updated patch on the run host:
    `git -C .swe-agent-src checkout -- sweagent && git -C .swe-agent-src apply "$(pwd)/scripts/sweagent-streaming.patch"`
- **Parsing**: `--parse auto` probes the gateway for native tool_calls (glm preset
  defaults `thought_action`; others `function_calling`).

## Analysis tooling (all in `scripts/`, committed)
- `token_usage.py <run>` — total + per-step mean/median input/output tokens (incl. reasoning).
- `compare_agent_control.py <run...>` — pass rate plus duplicate-action, empty-response,
  stop-reason, call, token, and loop metrics for SWE-agent/tool-call/Mini comparisons.
- `plot_success_by_type.py <run> [--compare <run2>]` — success rate per Fowler type.
- `plot_run_health.py <run> [--compare]` — step-count box plot + exit-status mix.
- `analysis/descriptive_task_types.jsonl` — the 100 tasks labeled by Fowler refactoring type.

## Canonical run command (pangu, on the run host)
```bash
cd ~/sources/RefactorBench && git pull
# (rebuild image if repos not yet baked; re-apply patch if needed — see warnings above)
source scripts/env.sh
python scripts/run_model.py --model pangu --variant descriptive \
  --image rb-swerex:py311-tree-sitter --workers 8 --slug pangu35b \
  --parse auto --startup-timeout 1800 --per-instance-call-limit 100 \
  --docker-arg=-e --docker-arg PYTHONSAFEPATH=1 --extra --agent.max_requeries 5
# score/report afterward (scoring is local/offline by default):
python scripts/report.py runs/pangu35b__descriptive/scores.json
```
Prereq: `scripts/models.env` must have the real `PANGU_MODEL=openai/<name>` (it's gitignored).
Resumable: re-running the same `--slug` skips completed tasks.

Native bash tool-call equivalent for the same model/interface-mismatch experiment:
```bash
source scripts/env.sh
python scripts/run_toolcall_model.py --model pangu --variant descriptive \
  --image rb-swerex:py311-tree-sitter --workers 8 --slug pangu35b-toolcall \
  --startup-timeout 1800 --command-timeout 30 --per-instance-call-limit 100 \
  --docker-arg=-e --docker-arg PYTHONSAFEPATH=1
python scripts/report.py runs/pangu35b-toolcall__descriptive/scores.json
```

Mini-SWE-Agent baseline for the same model/interface-mismatch experiment:
```bash
source scripts/env.sh
python scripts/run_mini_model.py --model pangu --variant descriptive \
  --image rb-swerex:py311-tree-sitter --workers 8 --slug pangu35b-mini \
  --startup-timeout 1800 --command-timeout 30 --per-instance-call-limit 100 \
  --docker-arg=-e --docker-arg PYTHONSAFEPATH=1
python scripts/report.py runs/pangu35b-mini__descriptive/scores.json
python scripts/compare_agent_control.py \
  runs/pangu35b__descriptive/scores.json \
  runs/pangu35b-toolcall__descriptive/scores.json \
  runs/pangu35b-mini__descriptive/scores.json
```

## Outstanding / next steps
1. **Verify streaming reasoning capture on the pangu host** — run the 1-task `reason-check`
   smoke, confirm `reasoning chars captured > 0` in the `.traj` (see prior commands).
2. **Launch the full pangu run** with the canonical command once #1 is confirmed.
3. **Cosmetic, optional**: silence the false "Model … does not support function calling"
   warning (it's logged before the registry loads in `models.py` — harmless; the probe
   already confirmed FC works).
4. Consider `--exclude-requeries` option for the plots/token tool (requery turns inflate
   step counts and input tokens).

## Gotchas learned (don't relearn the hard way)
- `--insecure-git` only disables TLS verify, **not** reachability — useless for the
  GitHub timeouts (now moot since offline).
- `--docker-arg` values starting with `-` need `=`: `--docker-arg=-e` (not `--docker-arg -e`).
- `--extra` is `REMAINDER` → must come **last**.
- 503/500/429/timeouts ARE retried (up to 20×, 10–120s backoff); auth/context/cost are not.
- `submitted (exit_X)` = had a patch when it died; bare `exit_X` = empty patch.
- Reference solutions are **withheld** by the paper — only AST tests in `tests/` define "correct".
- `.swe-agent-src` and `runs/*.traj` are gitignored (the streaming patch is the reproducible artifact).
