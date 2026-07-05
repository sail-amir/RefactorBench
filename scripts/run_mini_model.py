#!/usr/bin/env python3
"""Run one RefactorBench batch through Mini-SWE-Agent, then score it.

This is the Mini-SWE-Agent backend for the eval harness. It deliberately keeps
the same output contract as SWE-agent:

    runs/<slug>__<variant>/preds.json

where each prediction is keyed by instance id and contains a unified diff in
``model_patch``. The existing ``scripts/score.py`` then applies and checks the
patches, so SWE-agent and Mini-SWE-Agent runs are comparable.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from run_model import DEFAULT_ENV_FILE, PRESETS, load_env_file, resolve_sampling

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_AGENT_CONFIG = SCRIPTS_DIR / "rb_mini_agent.yaml"
DEFAULT_IMAGE = "rb-swerex:py311-tree-sitter"
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
PRED_LOCK = threading.Lock()


def sh(args: list[str], cwd: Path | str | None = None, timeout: int = 120,
       check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=str(cwd) if cwd else None, text=True,
        capture_output=True, timeout=timeout, check=check,
    )


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        sys.exit("PyYAML not found. Run setup.sh or install pyyaml in the harness venv.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_instances(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    if not isinstance(data, list) or not data:
        sys.exit(f"instances file is empty or invalid: {path}")
    return data


def repo_of(inst: dict[str, Any]) -> str:
    return inst["env"]["repo"]["repo_name"]


def iid_of(inst: dict[str, Any]) -> str:
    return inst["problem_statement"]["id"]


def text_of(inst: dict[str, Any]) -> str:
    return inst["problem_statement"]["text"]


def load_preds(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list):
        return {d["instance_id"]: d for d in data if isinstance(d, dict) and "instance_id" in d}
    return data if isinstance(data, dict) else {}


def rebuild_preds(outdir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, tuple[float, dict[str, Any]]] = {}
    for p in outdir.glob("*/result.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        iid = rec.get("instance_id")
        if not iid:
            continue
        mtime = p.stat().st_mtime
        prev = records.get(iid)
        if prev is None or mtime >= prev[0]:
            records[iid] = (mtime, rec)
    preds = {}
    for iid, (_, rec) in records.items():
        preds[iid] = {
            "model_name_or_path": rec.get("model_name_or_path", "unknown"),
            "instance_id": iid,
            "model_patch": rec.get("model_patch", ""),
        }
        for key in ("backend", "repo", "exit_status", "stop_reason", "control"):
            if key in rec:
                preds[iid][key] = rec[key]
    write_json_atomic(outdir / "preds.json", preds)
    return preds


def docker_image_present(image: str) -> bool:
    r = sh(["docker", "image", "inspect", image], timeout=60)
    return r.returncode == 0


def docker_tree_hash(image: str, repo: str, startup_timeout: float) -> str:
    cmd = ["docker", "run", "--rm", image, "git", "-C", f"/{repo}",
           "rev-parse", "HEAD^{tree}"]
    r = sh(cmd, timeout=max(30, int(startup_timeout or 120)))
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "").strip())
    return r.stdout.strip()


def local_tree_hash(repo: str) -> str:
    src = REPO_ROOT / "repositories" / repo
    if not src.is_dir():
        raise FileNotFoundError(f"local repo not found: {src}")
    with tempfile.TemporaryDirectory(prefix="rb_tree_") as td:
        dst = Path(td) / repo
        shutil.copytree(src, dst, symlinks=True)
        shutil.rmtree(dst / ".git", ignore_errors=True)
        sh(["git", "init", "-q"], cwd=dst, check=True)
        sh(["git", "config", "user.email", "rb@local"], cwd=dst, check=True)
        sh(["git", "config", "user.name", "RefactorBench"], cwd=dst, check=True)
        sh(["git", "add", "-A"], cwd=dst, check=True)
        r = sh(["git", "write-tree"], cwd=dst, check=True)
        return r.stdout.strip()


def precheck_base_trees(image: str, instances: list[dict[str, Any]],
                        startup_timeout: float) -> dict[str, dict[str, str]]:
    repos = sorted({repo_of(i) for i in instances})
    out: dict[str, dict[str, str]] = {}
    for repo in repos:
        local = local_tree_hash(repo)
        container = docker_tree_hash(image, repo, startup_timeout)
        out[repo] = {"local_tree": local, "container_tree": container}
        if local != container:
            raise RuntimeError(
                f"base tree mismatch for {repo}: local={local} container={container}"
            )
    return out


def normalize_action(action: str) -> str:
    return "\n".join(line.rstrip() for line in action.strip().splitlines()).strip()


def control_metrics_from_actions(actions: list[str], empty_responses: int = 0,
                                 model_calls: int = 0,
                                 tokens_sent: int = 0,
                                 tokens_received: int = 0) -> dict[str, Any]:
    norm = [normalize_action(a) for a in actions if normalize_action(a)]
    hashes = [hashlib.sha256(a.encode("utf-8", "replace")).hexdigest() for a in norm]
    repeated = 0
    seen: set[str] = set()
    streak = 0
    max_streak = 0
    last = None
    for h in hashes:
        if h in seen:
            repeated += 1
        seen.add(h)
        if h == last:
            streak += 1
        else:
            streak = 1
            last = h
        max_streak = max(max_streak, streak)
    n = len(hashes)
    return {
        "actions": n,
        "unique_actions": len(set(hashes)),
        "duplicate_actions": repeated,
        "duplicate_action_rate": round(repeated / n, 4) if n else 0.0,
        "max_repeated_action_streak": max_streak if n else 0,
        "empty_responses": empty_responses,
        "empty_response_rate": round(empty_responses / model_calls, 4) if model_calls else 0.0,
        "model_calls": model_calls,
        "tokens_sent": tokens_sent,
        "tokens_received": tokens_received,
    }


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
    messages = data.get("messages", []) or []
    assistant_messages = 0
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
        actions, empty_responses=empty, model_calls=model_calls,
        tokens_sent=tokens_sent, tokens_received=tokens_received,
    )
    metrics["trajectory_steps"] = len(messages)
    if assistant_messages and not metrics["actions"]:
        metrics["schema_warning"] = "no_actions_extracted_from_mini_trajectory"
    if model_calls and not (tokens_sent or tokens_received):
        metrics["token_warning"] = "no_token_usage_found_in_mini_trajectory"
    return metrics


def stop_reason(exit_status: str | None, patch: str, error: str = "") -> str:
    status = (exit_status or "").lower()
    if status == "submitted":
        return "completed_marker"
    if status == "contextlimitexceeded":
        return "context_limit"
    if status and "limit" in status:
        return "call_cap"
    if "timeout" in error.lower():
        return "timeout"
    if error:
        return "runtime_error"
    if patch.strip():
        return exit_status or "patch_collected"
    return exit_status or "unknown"


def model_kwargs(args, config: dict[str, Any]) -> dict[str, Any]:
    kw = dict((config.get("model") or {}).get("model_kwargs") or {})
    if args.api_base:
        kw["api_base"] = args.api_base
    if args.api_key:
        kw["api_key"] = args.api_key
    if args.temperature is not None:
        kw["temperature"] = args.temperature
    if args.top_p is not None:
        kw["top_p"] = args.top_p
    if args.max_output_tokens:
        kw["max_tokens"] = args.max_output_tokens
    if args.extra_body:
        kw["extra_body"] = args.extra_body
    return kw


def build_mini_objects(args, inst: dict[str, Any], outdir: Path, cfg: dict[str, Any]):
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.exceptions import LimitsExceeded, Submitted
        from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
    except ImportError as e:
        raise RuntimeError(
            "mini-swe-agent is not installed in this Python environment. "
            "Run setup.sh, or install the pinned package in the harness venv."
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

        @staticmethod
        def _rough_tokens(messages: list[dict]) -> int:
            total = 0
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else ""
                if isinstance(content, list):
                    content = " ".join(str(part) for part in content)
                total += max(0, round(len(str(content or "")) / 4))
            return total

        def query(self) -> dict:
            if self.rb_max_input_tokens:
                used = self._rough_tokens(self.messages)
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
    env = dict(env_cfg.get("env") or {})
    for item in args.env or []:
        k, _, v = item.partition("=")
        if k:
            env[k] = v
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
        model=model, env=env_obj, rb_max_input_tokens=args.max_input_tokens,
        **agent_cfg,
    )
    return agent, env_obj


def collect_patch(env_obj) -> str:
    output = env_obj.execute({
        "command": "git add -A -- . && git diff --cached --binary -- ."
    })
    return output.get("output") or ""


def run_one(args, inst: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
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
    agent = env_obj = None
    with debug_log.open("a", encoding="utf-8") as dbg:
        def log(msg: str) -> None:
            print(msg, file=dbg, flush=True)

        log(f"instance={iid} repo={repo} image={args.image}")
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
            try:
                patch = collect_patch(env_obj)
                write_text_atomic(patch_path, patch)
                log(f"patch_bytes={len(patch)}")
            except Exception as e:  # noqa: BLE001
                error = f"patch collection failed: {e}"
                log(error)
                log(traceback.format_exc())
        except Exception as e:  # noqa: BLE001
            error = str(e)
            log(f"ERROR: {error}")
            log(traceback.format_exc())
            if env_obj is not None:
                try:
                    patch = collect_patch(env_obj)
                    write_text_atomic(patch_path, patch)
                    log(f"patch_bytes_after_error={len(patch)}")
                except Exception as e2:  # noqa: BLE001
                    log(f"patch collection after error failed: {e2}")
        finally:
            if env_obj is not None:
                with contextlib.suppress(Exception):
                    env_obj.cleanup()

    metrics = mini_metrics(traj_path)
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
    elapsed = round(time.time() - started, 2)
    rec = {
        "model_name_or_path": args.model_name,
        "instance_id": iid,
        "model_patch": patch,
        "backend": "mini-swe-agent",
        "repo": repo,
        "exit_status": exit_status,
        "stop_reason": stop_reason(exit_status, patch, error),
        "duration": elapsed,
        "error": error,
        "control": metrics,
    }
    write_json_atomic(task_dir / "result.json", rec)
    with PRED_LOCK:
        rebuild_preds(Path(args.outdir))
    return rec


def run_score(args, instances: list[dict[str, Any]]) -> int:
    ids = [iid_of(i) for i in instances]
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "score.py"),
        "--preds", str(Path(args.outdir) / "preds.json"),
        "--out", str(Path(args.outdir) / "scores.json"),
        "--variant", args.variant,
        "--checkout", args.score_checkout,
        "--model", args.model_name,
        "--only", ",".join(ids),
    ]
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def resolve_args(args) -> None:
    load_env_file(args.env_file)
    preset = PRESETS.get(args.model, {})
    pfx = preset.get("env_prefix")

    def env(suffix: str) -> str | None:
        return os.environ.get(f"{pfx}_{suffix}") if pfx else None

    args.model_name = args.model_name or env("MODEL") or preset.get("model_name")
    if not args.model_name:
        sys.exit("need --model <preset> or --model-name")
    args.cost = args.cost if args.cost is not None else preset.get("cost", 0.0)
    args.api_base = args.api_base or env("API_BASE") or os.environ.get("RB_API_BASE")
    args.api_key = args.api_key or env("API_KEY") or os.environ.get("RB_API_KEY")
    if not args.api_base:
        print(f"WARNING: no api_base (set RB_API_BASE in {args.env_file}).", file=sys.stderr)
    if not args.api_key:
        print(f"WARNING: no api_key (set RB_API_KEY in {args.env_file}).", file=sys.stderr)
    args.max_input_tokens = args.max_input_tokens or 0
    resolve_sampling(args)
    if args.max_input_tokens == 0 and preset.get("max_input_tokens"):
        args.max_input_tokens = preset["max_input_tokens"]
    slug = args.slug or preset.get("slug") or args.model_name.split("/")[-1]
    args.outdir = str(REPO_ROOT / "runs" / f"{slug}__{args.variant}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Mini's text model runs non-streaming; keep the flag accepted for sampling
    # parity with run_model.py but do not forward it to LiteLLM.
    args.stream = False

    args.registry_json = str(Path(args.outdir) / "registry.json")
    registry: dict[str, Any] = {}
    if args.registry and Path(args.registry).exists():
        with open(args.registry, "r", encoding="utf-8") as fh:
            with contextlib.suppress(json.JSONDecodeError):
                registry = json.load(fh)
    registry.setdefault(args.model_name, {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": False,
    })
    write_json_atomic(Path(args.registry_json), registry)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), help="preset")
    ap.add_argument("--variant", default="descriptive",
                    choices=["base", "descriptive", "lazy"])
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
