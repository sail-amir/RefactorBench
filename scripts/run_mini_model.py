#!/usr/bin/env python3
"""Run one RefactorBench batch through Mini-SWE-Agent, then score it."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from run_common import (
    DEFAULT_ENV_FILE,
    DEFAULT_IMAGE,
    OUTPUT_TRUNCATION_LIMIT,
    PRESETS,
    REPO_ROOT,
    SCRIPTS_DIR,
    SENTINEL,
    build_registry,
    collect_patch_from_executor,
    control_metrics_from_actions,
    default_container_env,
    docker_image_present,
    iid_of,
    load_instances,
    load_yaml,
    model_kwargs,
    precheck_base_trees,
    rebuild_preds,
    repo_of,
    resolve_common_args,
    rough_tokens_from_messages,
    run_score,
    stop_reason,
    text_of,
    write_json_atomic,
    write_text_atomic,
)

DEFAULT_AGENT_CONFIG = SCRIPTS_DIR / "rb_mini_agent.yaml"
PRED_LOCK = threading.Lock()


def mini_metrics(traj_path: Path) -> dict[str, Any]:
    if not traj_path.exists():
        return control_metrics_from_actions([])
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return control_metrics_from_actions([])

    actions: list[str] = []
    empty = 0
    model_calls = 0
    tokens_sent = 0
    tokens_received = 0
    assistant_messages = 0
    messages = data.get("messages", []) or []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content") or ""
        extra = msg.get("extra") or {}
        if role == "assistant":
            assistant_messages += 1
            model_calls += 1
            if not str(content).strip():
                empty += 1
            for action in extra.get("actions") or []:
                if isinstance(action, dict):
                    actions.append(str(action.get("cmd") or action.get("command") or action))
                else:
                    actions.append(str(action))
            response = extra.get("response") or {}
            usage = response.get("usage") or {}
            tokens_sent += int(usage.get("prompt_tokens") or 0)
            tokens_received += int(usage.get("completion_tokens") or 0)

    info = data.get("info") or {}
    ms = info.get("model_stats") or {}
    tokens_sent = int(ms.get("tokens_sent") or tokens_sent or 0)
    tokens_received = int(ms.get("tokens_received") or tokens_received or 0)
    model_calls = int(ms.get("api_calls") or model_calls or 0)
    metrics = control_metrics_from_actions(
        actions,
        empty_responses=empty,
        model_calls=model_calls,
        tokens_sent=tokens_sent,
        tokens_received=tokens_received,
    )
    metrics["trajectory_steps"] = len(messages)
    if assistant_messages and not metrics["actions"]:
        metrics["schema_warning"] = "no_actions_extracted_from_mini_trajectory"
    if model_calls and not (tokens_sent or tokens_received):
        metrics["token_warning"] = "no_token_usage_found_in_mini_trajectory"
    return metrics


def build_mini_objects(args: Any, inst: dict[str, Any], outdir: Path,
                       cfg: dict[str, Any]):
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.exceptions import LimitsExceeded, Submitted
        from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
    except ImportError as e:
        raise RuntimeError(
            "mini-swe-agent is not installed in this Python environment. "
            "Run setup.sh, setup_mini.sh, or install the pinned package in the harness venv."
        ) from e

    class RefactorBenchDockerEnvironment(DockerEnvironment):
        def _check_finished(self, output: dict):
            lines = output.get("output", "").lstrip().splitlines(keepends=True)
            if lines and lines[0].strip() == SENTINEL and output.get("returncode") == 0:
                submission = "".join(lines[1:])
                raise Submitted(
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {"exit_status": "Submitted", "submission": submission},
                    }
                )
            return super()._check_finished(output)

    class RefactorBenchAgent(DefaultAgent):
        def __init__(self, *pargs, rb_max_input_tokens: int = 0, **kwargs):
            self.rb_max_input_tokens = rb_max_input_tokens
            super().__init__(*pargs, **kwargs)

        def query(self) -> dict:
            if self.rb_max_input_tokens:
                used = rough_tokens_from_messages(self.messages)
                if used > self.rb_max_input_tokens:
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "ContextLimitExceeded",
                            "extra": {
                                "exit_status": "ContextLimitExceeded",
                                "submission": "",
                                "tokens_estimate": used,
                                "max_input_tokens": self.rb_max_input_tokens,
                            },
                        }
                    )
            return super().query()

    repo = repo_of(inst)
    iid = iid_of(inst)
    agent_cfg = dict(cfg.get("agent") or {})
    model_cfg = dict(cfg.get("model") or {})
    env_cfg = dict(cfg.get("environment") or {})

    if args.per_instance_call_limit:
        agent_cfg["step_limit"] = args.per_instance_call_limit
    if args.cost is not None:
        agent_cfg["cost_limit"] = args.cost

    env = default_container_env()
    env.update(dict(env_cfg.get("env") or {}))
    for item in args.env or []:
        key, _, value = item.partition("=")
        if key:
            env[key] = value
    env_cfg.update({
        "image": args.image,
        "cwd": f"/{repo}",
        "env": env,
        "timeout": args.command_timeout,
        "pull_timeout": int(args.startup_timeout or 120),
        "run_args": ["--rm", *(args.docker_arg or [])],
    })

    model_cfg.update({
        "model_name": args.model_name,
        "model_kwargs": model_kwargs(args, cfg),
        "litellm_model_registry": args.registry_json,
    })
    model = LitellmTextbasedModel(**model_cfg)
    env_obj = RefactorBenchDockerEnvironment(**env_cfg)
    agent_cfg["output_path"] = outdir / iid / f"{iid}.traj.json"
    agent = RefactorBenchAgent(
        model=model,
        env=env_obj,
        rb_max_input_tokens=args.max_input_tokens,
        **agent_cfg,
    )
    return agent, env_obj


def collect_mini_patch(env_obj: Any) -> str:
    def execute(command: str) -> str:
        output = env_obj.execute({"command": command})
        return output.get("output") or ""

    return collect_patch_from_executor(execute)


def run_one(args: Any, inst: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    iid = iid_of(inst)
    repo = repo_of(inst)
    task_dir = Path(args.outdir) / iid
    task_dir.mkdir(parents=True, exist_ok=True)
    debug_log = task_dir / f"{iid}.debug.log"
    patch_path = task_dir / "patch.diff"
    traj_path = task_dir / f"{iid}.traj.json"

    started = time.time()
    error = ""
    exit_status = ""
    patch = ""
    env_obj = None
    with debug_log.open("a", encoding="utf-8") as dbg:
        def log(msg: str) -> None:
            print(msg, file=dbg, flush=True)

        log(f"instance={iid} repo={repo} image={args.image}")
        log(f"command_timeout={args.command_timeout} output_truncation_limit={OUTPUT_TRUNCATION_LIMIT}")
        try:
            agent, env_obj = build_mini_objects(args, inst, Path(args.outdir), cfg)
            info = agent.run(
                task=text_of(inst),
                cwd=f"/{repo}",
                instance_id=iid,
                repo=repo,
            )
            exit_status = str((info or {}).get("exit_status") or "")
            log(f"mini exit_status={exit_status}")
            patch = collect_mini_patch(env_obj)
            write_text_atomic(patch_path, patch)
            log(f"patch_bytes={len(patch)}")
        except Exception as e:  # noqa: BLE001
            error = str(e)
            log(f"ERROR: {error}")
            log(traceback.format_exc())
            if env_obj is not None:
                try:
                    patch = collect_mini_patch(env_obj)
                    write_text_atomic(patch_path, patch)
                    log(f"patch_bytes_after_error={len(patch)}")
                except Exception as e2:  # noqa: BLE001
                    log(f"patch collection after error failed: {e2}")
        finally:
            if env_obj is not None:
                with contextlib.suppress(Exception):
                    env_obj.cleanup()

    metrics = mini_metrics(traj_path)
    metrics["command_timeout"] = args.command_timeout
    metrics["output_truncation_limit"] = OUTPUT_TRUNCATION_LIMIT
    schema_warning = metrics.get("schema_warning")
    token_warning = metrics.get("token_warning")
    if schema_warning or token_warning:
        with debug_log.open("a", encoding="utf-8") as dbg:
            if schema_warning:
                print(f"WARNING: {schema_warning}", file=dbg, flush=True)
            if token_warning:
                print(f"WARNING: {token_warning}", file=dbg, flush=True)
    if exit_status and exit_status.lower() != "submitted":
        metrics["exit_status"] = exit_status

    rec = {
        "model_name_or_path": args.model_name,
        "instance_id": iid,
        "model_patch": patch,
        "backend": "mini-swe-agent",
        "repo": repo,
        "exit_status": exit_status,
        "stop_reason": stop_reason(exit_status, patch, error),
        "duration": round(time.time() - started, 2),
        "error": error,
        "control": metrics,
    }
    write_json_atomic(task_dir / "result.json", rec)
    with PRED_LOCK:
        rebuild_preds(Path(args.outdir))
    return rec


def resolve_args(args: Any) -> None:
    resolve_common_args(args, force_stream_false=True)
    write_json_atomic(
        Path(args.registry_json),
        build_registry(args.model_name, supports_function_calling=False),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), help="preset")
    ap.add_argument("--variant", default="descriptive", choices=["base", "descriptive", "lazy"])
    ap.add_argument("--slug")
    ap.add_argument("--model-name")
    ap.add_argument("--api-base")
    ap.add_argument("--api-key")
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    ap.add_argument("--sampling-file")
    ap.add_argument("--instances")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--startup-timeout", type=float, default=1200,
                    help="Docker container start timeout if supported by Mini")
    ap.add_argument("--command-timeout", type=int, default=30,
                    help="per-command timeout inside the Mini Docker environment")
    ap.add_argument("--docker-arg", action="append", default=[],
                    help="extra arg passed to docker run (repeatable)")
    ap.add_argument("--env", action="append", default=[],
                    help="extra KEY=VALUE env var inside the task container")
    ap.add_argument("--per-instance-call-limit", type=int, default=100)
    ap.add_argument("--max-input-tokens", type=int, default=0)
    ap.add_argument("--max-output-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--reasoning-effort")
    ap.add_argument("--stream", action="store_true", default=False,
                    help="accepted for sampling parity; Mini text model runs non-streaming")
    ap.add_argument("--cost", type=float)
    ap.add_argument("--redo-existing", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--score-checkout", default="local", choices=["fork", "local"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--skip-base-tree-check", action="store_true")
    ap.add_argument("--agent-config", default=str(DEFAULT_AGENT_CONFIG))
    ap.add_argument("--registry", default=str(SCRIPTS_DIR / "litellm_registry.json"))
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    resolve_args(args)

    instances_path = Path(args.instances) if args.instances else SCRIPTS_DIR / f"{args.variant}_instances.yaml"
    if not instances_path.exists():
        sys.exit(f"instances file not found: {instances_path}")
    instances = load_instances(instances_path)
    cfg = load_yaml(Path(args.agent_config))

    write_json_atomic(Path(args.outdir) / "mini_run.config.json", {
        "model_name": args.model_name,
        "variant": args.variant,
        "instances": str(instances_path),
        "image": args.image,
        "workers": args.workers,
        "startup_timeout": args.startup_timeout,
        "command_timeout": args.command_timeout,
        "output_truncation_limit": OUTPUT_TRUNCATION_LIMIT,
        "per_instance_call_limit": args.per_instance_call_limit,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": args.extra_body,
        "score_checkout": args.score_checkout,
    })

    print(f"# mini slug={Path(args.outdir).name} variant={args.variant} model={args.model_name}")
    print(f"# instances={len(instances)} image={args.image} workers={args.workers}")
    print(f"# command_timeout={args.command_timeout}s startup_timeout={args.startup_timeout}s")
    if args.dry_run:
        print("# dry-run: not starting Docker or model calls")
        return 0

    if not docker_image_present(args.image):
        sys.exit(f"Docker image not found locally: {args.image}\n"
                 f"Build it first with setup.sh. The Mini runner will not pull images.")

    if not args.skip_base_tree_check:
        print("# checking container/local base tree hashes ...", flush=True)
        try:
            base = precheck_base_trees(args.image, instances, args.startup_timeout)
            write_json_atomic(Path(args.outdir) / "base_tree_check.json", base)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"base tree precheck failed: {e}")

    if args.precheck_only:
        print("# precheck-only complete")
        return 0

    with PRED_LOCK:
        preds = rebuild_preds(Path(args.outdir))
    todo = []
    for inst in instances:
        iid = iid_of(inst)
        patch = (preds.get(iid) or {}).get("model_patch") or ""
        if patch.strip() and not args.redo_existing:
            print(f"# skip existing {iid}")
            continue
        todo.append(inst)
    print(f"# running {len(todo)} tasks ({len(instances) - len(todo)} skipped)")

    failures = 0
    if todo:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, args, inst, cfg): iid_of(inst) for inst in todo}
            for fut in cf.as_completed(futs):
                iid = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"[ERROR] {iid}: {e}", file=sys.stderr)
                    continue
                plen = len(rec.get("model_patch") or "")
                print(f"[{rec.get('stop_reason','?')}] {iid} patch={plen}B "
                      f"calls={(rec.get('control') or {}).get('model_calls', 0)}")

    with PRED_LOCK:
        rebuild_preds(Path(args.outdir))

    if args.no_score:
        print(f"skipping scoring (--no-score). preds at {Path(args.outdir) / 'preds.json'}")
        return 1 if failures else 0
    rc = run_score(args, instances)
    return rc or (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
