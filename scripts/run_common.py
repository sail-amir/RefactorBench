#!/usr/bin/env python3
"""Shared helpers for RefactorBench alternate agent backends."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from run_model import DEFAULT_ENV_FILE, PRESETS, load_env_file, resolve_sampling

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_IMAGE = "rb-swerex:py311-tree-sitter"
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
OUTPUT_TRUNCATION_LIMIT = 10_000


def sh(args: list[str], cwd: Path | str | None = None, timeout: int = 120,
       check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
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
        return {
            d["instance_id"]: d
            for d in data
            if isinstance(d, dict) and "instance_id" in d
        }
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
    return sh(["docker", "image", "inspect", image], timeout=60).returncode == 0


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


def truncate_output(output: str, limit: int = OUTPUT_TRUNCATION_LIMIT) -> dict[str, Any]:
    if len(output) <= limit:
        return {"output": output}
    keep = max(1, limit // 2)
    return {
        "warning": "output_truncated",
        "output_head": output[:keep],
        "output_tail": output[-keep:],
        "elided_chars": len(output) - (keep * 2),
    }


def command_result_payload(returncode: int, output: str,
                           exception_info: str | None = None,
                           limit: int = OUTPUT_TRUNCATION_LIMIT) -> dict[str, Any]:
    payload: dict[str, Any] = {"returncode": returncode}
    if exception_info:
        payload["exception_info"] = exception_info
    payload.update(truncate_output(output, limit=limit))
    return payload


class DockerSession:
    """Small docker-run/docker-exec wrapper for the native tool-call backend."""

    def __init__(self, image: str, repo: str, env: dict[str, str] | None = None,
                 docker_args: list[str] | None = None, startup_timeout: float = 1200,
                 command_timeout: int = 30, output_limit: int = OUTPUT_TRUNCATION_LIMIT):
        self.image = image
        self.repo = repo
        self.cwd = f"/{repo}"
        self.env = env or {}
        self.docker_args = docker_args or []
        self.startup_timeout = startup_timeout
        self.command_timeout = command_timeout
        self.output_limit = output_limit
        self.name = f"rb-toolcall-{uuid.uuid4().hex[:12]}"
        self.container_id = ""

    def start(self) -> None:
        cmd = ["docker", "run", "-d", "--name", self.name, "-w", self.cwd, "--rm"]
        for key, value in self.env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += self.docker_args
        cmd += [self.image, "sleep", "2h"]
        r = sh(cmd, timeout=max(30, int(self.startup_timeout or 120)))
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or "").strip())
        self.container_id = r.stdout.strip()

    def execute_raw(self, command: str, timeout: int | None = None) -> tuple[int, str, str]:
        timeout = int(timeout or self.command_timeout)
        wrapped = f"timeout {timeout}s /bin/sh -lc {shlex.quote(command)}"
        cmd = ["docker", "exec", "-w", self.cwd, self.name, "/bin/sh", "-lc", wrapped]
        started = time.time()
        try:
            r = sh(cmd, timeout=timeout + 15)
            output = (r.stdout or "") + (r.stderr or "")
            exc = "CommandTimeoutError" if r.returncode == 124 else ""
            return r.returncode, output, exc
        except subprocess.TimeoutExpired:
            return -1, "", f"CommandTimeoutError after {round(time.time() - started, 2)}s"

    def execute(self, command: str) -> dict[str, Any]:
        rc, output, exc = self.execute_raw(command)
        return command_result_payload(
            rc, output, exception_info=exc or None, limit=self.output_limit
        )

    def collect_patch(self) -> str:
        rc, output, exc = self.execute_raw(
            "git add -A -- . && git diff --cached --binary -- .",
            timeout=max(self.command_timeout, 120),
        )
        if rc != 0:
            raise RuntimeError(exc or output or f"patch collection failed rc={rc}")
        return output

    def cleanup(self) -> None:
        sh(["docker", "rm", "-f", self.name], timeout=60)


def collect_patch_from_executor(execute: Callable[[str], str]) -> str:
    return execute("git add -A -- . && git diff --cached --binary -- .")


def normalize_action(action: str) -> str:
    return "\n".join(line.rstrip() for line in action.strip().splitlines()).strip()


def control_metrics_from_actions(actions: list[str], empty_responses: int = 0,
                                 model_calls: int = 0, tokens_sent: int = 0,
                                 tokens_received: int = 0,
                                 malformed_actions: int = 0) -> dict[str, Any]:
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
        "malformed_actions": malformed_actions,
        "malformed_action_rate": round(malformed_actions / model_calls, 4) if model_calls else 0.0,
        "model_calls": model_calls,
        "tokens_sent": tokens_sent,
        "tokens_received": tokens_received,
    }


def rough_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        total += max(0, round(len(str(content)) / 4))
        for call in message.get("tool_calls") or []:
            total += max(0, round(len(json.dumps(call, sort_keys=True)) / 4))
    return total


def stop_reason(exit_status: str | None, patch: str, error: str = "") -> str:
    status = (exit_status or "").lower()
    if status in {"submitted", "completed_marker"}:
        return "completed_marker"
    if status in {"completed_no_tool", "no_tool_stop"}:
        return "completed_no_tool"
    if status == "contextlimitexceeded":
        return "context_limit"
    if status == "length_truncated":
        return "length_truncated"
    if status and "limit" in status:
        return "call_cap"
    if "timeout" in error.lower():
        return "timeout"
    if error:
        return "runtime_error"
    if patch.strip():
        return exit_status or "patch_collected"
    return exit_status or "unknown"


def model_kwargs(args: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    kw = dict(((config or {}).get("model") or {}).get("model_kwargs") or {})
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


def api_model_name(model_name: str) -> str:
    return model_name.split("/", 1)[1] if model_name.startswith("openai/") else model_name


def resolve_common_args(args: Any, *, default_slug: str | None = None,
                        force_stream_false: bool = False) -> None:
    load_env_file(args.env_file)
    preset = PRESETS.get(args.model, {})
    pfx = preset.get("env_prefix")

    def env(suffix: str) -> str | None:
        return os.environ.get(f"{pfx}_{suffix}") if pfx else None

    args.model_name = args.model_name or env("MODEL") or preset.get("model_name")
    if not args.model_name:
        sys.exit("need --model <preset> or --model-name")
    args.cost = args.cost if getattr(args, "cost", None) is not None else preset.get("cost", 0.0)
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
    if force_stream_false:
        args.stream = False
    slug = args.slug or default_slug or preset.get("slug") or args.model_name.split("/")[-1]
    args.outdir = str(REPO_ROOT / "runs" / f"{slug}__{args.variant}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    args.registry_json = str(Path(args.outdir) / "registry.json")
    write_json_atomic(Path(args.registry_json), build_registry(args.model_name))


def build_registry(model_name: str, supports_function_calling: bool = True) -> dict[str, Any]:
    return {
        model_name: {
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": supports_function_calling,
            "supports_tool_choice": supports_function_calling,
        }
    }


def run_score(args: Any, instances: list[dict[str, Any]]) -> int:
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


def default_container_env(extra: list[str] | None = None) -> dict[str, str]:
    env = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
        "PYTHONSAFEPATH": "1",
    }
    for item in extra or []:
        key, _, value = item.partition("=")
        if key:
            env[key] = value
    return env

