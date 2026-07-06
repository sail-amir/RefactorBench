#!/usr/bin/env python3
"""Run one RefactorBench batch through native OpenAI bash tool-calls, then score it."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from run_common import (
    DEFAULT_ENV_FILE,
    DEFAULT_IMAGE,
    OUTPUT_TRUNCATION_LIMIT,
    PRESETS,
    DockerSession,
    SCRIPTS_DIR,
    SENTINEL,
    api_model_name,
    control_metrics_from_actions,
    default_container_env,
    docker_image_present,
    iid_of,
    load_instances,
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

PRED_LOCK = threading.Lock()

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the checked-out repository.",
        "parameters": {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "additionalProperties": False,
        },
    },
}


def endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def is_transient_http_status(status: int) -> bool:
    return status == 429 or status == 408 or 500 <= status <= 599


def chat_completion(args: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    if not args.api_base:
        raise RuntimeError("api_base is required for native tool-call runner")
    if not args.api_key:
        raise RuntimeError("api_key is required for native tool-call runner")
    payload: dict[str, Any] = {
        "model": api_model_name(args.model_name),
        "messages": messages,
        "tools": [BASH_TOOL],
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.max_output_tokens:
        payload["max_tokens"] = args.max_output_tokens
    for key, value in (args.extra_body or {}).items():
        payload[key] = value

    data = json.dumps(payload).encode("utf-8")
    url = endpoint(args.api_base)
    max_attempts = max(1, int(args.request_retries) + 1)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
                response["_rb_retry_count"] = attempt - 1
                return response
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            last_error = f"HTTP {e.code}: {body}"
            if not is_transient_http_status(e.code) or attempt >= max_attempts:
                raise RuntimeError(last_error) from e
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt >= max_attempts:
                raise RuntimeError(last_error) from e

        sleep_s = min(
            float(args.request_retry_max_sleep),
            float(args.request_retry_initial_sleep) * (2 ** (attempt - 1)),
        )
        time.sleep(sleep_s)
    raise RuntimeError(last_error or "chat completion failed")



def build_messages(task_text: str) -> list[dict[str, Any]]:
    system = (
        "You are a helpful assistant that can interact with a computer. "
        "Use the bash tool to inspect and edit the checked-out repository. "
        "When the task is complete, respond with a concise final summary and no tool call."
    )
    user = f"""Please solve this refactoring task:

# Task

{task_text}

The repository is already checked out in the current working directory.

Rules:
- Use the bash tool for commands and file edits.
- Directory and environment-variable changes are not persistent across commands, so include needed `cd` or environment setup in each command.
- Do not start servers or run the full test suite.
- Do not edit tests unless the task explicitly asks for test edits.
- Search and inspect before editing; update definitions, imports, and usages consistently.
- Review your changes with focused commands such as grep, rg, python syntax checks, or git diff.
- When satisfied, either provide a final answer with no tool call, or call bash with exactly: echo {SENTINEL}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def parse_tool_args(call: dict[str, Any]) -> tuple[str, str]:
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw = function.get("arguments") or "{}"
    if isinstance(raw, dict):
        args = raw
    else:
        args = json.loads(raw)
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("bash tool call missing non-empty command")
    return name, command


def usage_tokens(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def run_one(args: Any, inst: dict[str, Any]) -> dict[str, Any]:
    iid = iid_of(inst)
    repo = repo_of(inst)
    task_dir = Path(args.outdir) / iid
    task_dir.mkdir(parents=True, exist_ok=True)
    debug_log = task_dir / f"{iid}.debug.log"
    patch_path = task_dir / "patch.diff"
    traj_path = task_dir / f"{iid}.traj.json"

    started = time.time()
    session: DockerSession | None = None
    messages = build_messages(text_of(inst))
    trajectory: list[dict[str, Any]] = []
    actions: list[str] = []
    empty_responses = 0
    malformed = 0
    consecutive_format_errors = 0
    model_calls = 0
    request_retries = 0
    tokens_sent = 0
    tokens_received = 0
    length_retries = 0
    exit_status = ""
    error = ""
    patch = ""

    with debug_log.open("a", encoding="utf-8") as dbg:
        def log(msg: str) -> None:
            print(msg, file=dbg, flush=True)

        log(f"instance={iid} repo={repo} image={args.image}")
        log(f"command_timeout={args.command_timeout} output_truncation_limit={OUTPUT_TRUNCATION_LIMIT}")
        log(
            "request_timeout="
            f"{args.request_timeout} request_retries={args.request_retries} "
            f"retry_initial_sleep={args.request_retry_initial_sleep} "
            f"retry_max_sleep={args.request_retry_max_sleep}"
        )
        try:
            env = default_container_env(args.env)
            session = DockerSession(
                image=args.image,
                repo=repo,
                env=env,
                docker_args=args.docker_arg or [],
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
                output_limit=OUTPUT_TRUNCATION_LIMIT,
            )
            session.start()

            while model_calls < args.per_instance_call_limit:
                if args.max_input_tokens:
                    used = rough_tokens_from_messages(messages)
                    if used > args.max_input_tokens:
                        exit_status = "ContextLimitExceeded"
                        error = f"context estimate {used} > max_input_tokens {args.max_input_tokens}"
                        log(error)
                        break

                response = chat_completion(args, messages)
                model_calls += 1
                request_retries += int(response.get("_rb_retry_count") or 0)
                prompt_toks, completion_toks = usage_tokens(response)
                tokens_sent += prompt_toks
                tokens_received += completion_toks

                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                finish_reason = choice.get("finish_reason") or response.get("finish_reason") or ""
                content = message.get("content")
                tool_calls = message.get("tool_calls") or []
                if not str(content or "").strip():
                    empty_responses += 1
                trajectory.append({
                    "model_call": model_calls,
                    "finish_reason": finish_reason,
                    "assistant": message,
                    "usage": response.get("usage") or {},
                })

                if tool_calls:
                    messages.append(sanitize_assistant_message(message))
                    for call in tool_calls:
                        try:
                            name, command = parse_tool_args(call)
                            if name != "bash":
                                raise ValueError(f"unknown tool {name!r}")
                        except Exception as e:  # noqa: BLE001
                            malformed += 1
                            consecutive_format_errors += 1
                            err_payload = {
                                "returncode": -1,
                                "exception_info": f"ToolCallFormatError: {e}",
                                "output": "",
                            }
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call.get("id") or f"missing-{model_calls}",
                                "content": json.dumps(err_payload),
                            })
                            trajectory.append({"tool_call_error": str(e), "tool_call": call})
                            if consecutive_format_errors >= args.max_consecutive_format_errors:
                                exit_status = "format_error"
                                error = str(e)
                                break
                            continue

                        actions.append(command)
                        consecutive_format_errors = 0
                        if command.strip() == f"echo {SENTINEL}":
                            exit_status = "completed_marker"
                            trajectory.append({
                                "tool_call_id": call.get("id"),
                                "command": command,
                                "sentinel": True,
                            })
                            break

                        result = session.execute(command)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.get("id") or f"missing-{model_calls}",
                            "content": json.dumps(result),
                        })
                        trajectory.append({
                            "tool_call_id": call.get("id"),
                            "command": command,
                            "result": result,
                        })
                    if exit_status:
                        break
                    continue

                if finish_reason == "stop":
                    exit_status = "completed_no_tool"
                    break
                if finish_reason == "length":
                    length_retries += 1
                    trajectory.append({"length_truncated": True, "retry": length_retries})
                    if length_retries <= args.max_length_retries and model_calls < args.per_instance_call_limit:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your previous response ended due to the output token limit before "
                                "a complete tool call or final answer. Continue concisely. If work "
                                "is needed, call bash; if the task is complete, provide a final "
                                "answer with no tool call."
                            ),
                        })
                        continue
                    exit_status = "length_truncated"
                    error = "assistant response ended with finish_reason=length"
                    break

                malformed += 1
                consecutive_format_errors += 1
                exit_status = "unknown_finish_reason"
                error = f"no tool_calls with finish_reason={finish_reason!r}"
                break
            else:
                exit_status = "LimitsExceeded"

            if session is not None:
                patch = session.collect_patch()
                write_text_atomic(patch_path, patch)
                log(f"patch_bytes={len(patch)}")
        except Exception as e:  # noqa: BLE001
            error = str(e)
            log(f"ERROR: {error}")
            log(traceback.format_exc())
            if session is not None:
                with contextlib.suppress(Exception):
                    patch = session.collect_patch()
                    write_text_atomic(patch_path, patch)
                    log(f"patch_bytes_after_error={len(patch)}")
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    session.cleanup()

    control = control_metrics_from_actions(
        actions,
        empty_responses=empty_responses,
        model_calls=model_calls,
        tokens_sent=tokens_sent,
        tokens_received=tokens_received,
        malformed_actions=malformed,
    )
    control.update({
        "trajectory_steps": len(trajectory),
        "command_timeout": args.command_timeout,
        "output_truncation_limit": OUTPUT_TRUNCATION_LIMIT,
        "length_truncations": length_retries,
        "request_retries": request_retries,
    })
    if exit_status:
        control["exit_status"] = exit_status

    rec = {
        "model_name_or_path": args.model_name,
        "instance_id": iid,
        "model_patch": patch,
        "backend": "native-bash-toolcall",
        "repo": repo,
        "exit_status": exit_status,
        "stop_reason": stop_reason(exit_status, patch, error),
        "duration": round(time.time() - started, 2),
        "error": error,
        "control": control,
    }
    write_json_atomic(traj_path, {
        "instance_id": iid,
        "repo": repo,
        "messages": messages,
        "trajectory": trajectory,
        "info": rec,
    })
    write_json_atomic(task_dir / "result.json", rec)
    with PRED_LOCK:
        rebuild_preds(Path(args.outdir))
    return rec


def resolve_args(args: Any) -> None:
    resolve_common_args(args, force_stream_false=True)


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
    ap.add_argument("--startup-timeout", type=float, default=1200)
    ap.add_argument("--command-timeout", type=int, default=30)
    ap.add_argument("--request-timeout", type=int, default=300)
    ap.add_argument("--request-retries", type=int, default=20,
                    help="retry transient 429/5xx/timeout gateway failures this many times")
    ap.add_argument("--request-retry-initial-sleep", type=float, default=1.0)
    ap.add_argument("--request-retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--docker-arg", action="append", default=[],
                    help="extra arg passed to docker run (repeatable)")
    ap.add_argument("--env", action="append", default=[],
                    help="extra KEY=VALUE env var inside the task container")
    ap.add_argument("--per-instance-call-limit", type=int, default=100)
    ap.add_argument("--max-consecutive-format-errors", type=int, default=3)
    ap.add_argument("--max-length-retries", type=int, default=1)
    ap.add_argument("--max-input-tokens", type=int, default=0)
    ap.add_argument("--max-output-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--reasoning-effort")
    ap.add_argument("--stream", action="store_true", default=False,
                    help="accepted for sampling parity; native tool-call runner is non-streaming")
    ap.add_argument("--cost", type=float)
    ap.add_argument("--redo-existing", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--score-checkout", default="local", choices=["fork", "local"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--skip-base-tree-check", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    resolve_args(args)

    instances_path = Path(args.instances) if args.instances else SCRIPTS_DIR / f"{args.variant}_instances.yaml"
    if not instances_path.exists():
        sys.exit(f"instances file not found: {instances_path}")
    instances = load_instances(instances_path)

    write_json_atomic(Path(args.outdir) / "toolcall_run.config.json", {
        "model_name": args.model_name,
        "api_model_name": api_model_name(args.model_name),
        "variant": args.variant,
        "instances": str(instances_path),
        "image": args.image,
        "workers": args.workers,
        "startup_timeout": args.startup_timeout,
        "command_timeout": args.command_timeout,
        "request_timeout": args.request_timeout,
        "request_retries": args.request_retries,
        "request_retry_initial_sleep": args.request_retry_initial_sleep,
        "request_retry_max_sleep": args.request_retry_max_sleep,
        "output_truncation_limit": OUTPUT_TRUNCATION_LIMIT,
        "per_instance_call_limit": args.per_instance_call_limit,
        "max_consecutive_format_errors": args.max_consecutive_format_errors,
        "max_length_retries": args.max_length_retries,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": args.extra_body,
        "score_checkout": args.score_checkout,
        "tools": [BASH_TOOL],
    })

    print(f"# toolcall slug={Path(args.outdir).name} variant={args.variant} model={args.model_name}")
    print(f"# instances={len(instances)} image={args.image} workers={args.workers}")
    print(f"# command_timeout={args.command_timeout}s request_timeout={args.request_timeout}s")
    if args.dry_run:
        print("# dry-run: not starting Docker or model calls")
        return 0

    if not docker_image_present(args.image):
        sys.exit(f"Docker image not found locally: {args.image}\n"
                 f"Build it first with setup.sh. The tool-call runner will not pull images.")

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
            futs = {pool.submit(run_one, args, inst): iid_of(inst) for inst in todo}
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
