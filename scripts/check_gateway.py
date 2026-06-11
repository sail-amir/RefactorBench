#!/usr/bin/env python3
"""Quick health check for the model gateway, per model.

Resolves the same config run_model.py uses (scripts/models.env: RB_API_BASE /
RB_API_KEY / <PREFIX>_MODEL / optional per-model overrides) and probes the
gateway with three small requests:

  1. chat      — endpoint reachable + auth + model name valid (returns content)
  2. tools     — does the model return native tool_calls (function_calling works)
  3. thinking  — does reasoning_effort=high yield a reasoning_content channel
                 (only run with --thinking)

Exit code 0 if the basic chat works, non-zero otherwise.

Usage:
    python scripts/check_gateway.py --model deepseek
    python scripts/check_gateway.py --model glm --thinking
    python scripts/check_gateway.py --model-name openai/foo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV_FILE = os.path.join(SCRIPTS_DIR, "models.env")

# env var prefix per preset (mirrors run_model.py)
PRESET_PREFIX = {"claude": "CLAUDE", "deepseek": "DEEPSEEK", "glm": "GLM", "pangu": "PANGU"}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), _envval(v))


def _envval(v: str) -> str:
    """Parse an env-file value: honor quotes, else strip an inline ` # comment`."""
    v = v.strip()
    if v[:1] in ("'", '"'):
        q = v[0]
        end = v.find(q, 1)
        return v[1:end] if end != -1 else v[1:]
    for i, ch in enumerate(v):  # comment must follow whitespace (so URLs with # survive)
        if ch == "#" and i > 0 and v[i - 1] in " \t":
            v = v[:i]
            break
    return v.strip()


def resolve(args):
    pfx = PRESET_PREFIX.get(args.model)

    def env(suffix):
        return os.environ.get(f"{pfx}_{suffix}") if pfx else None

    model = args.model_name or env("MODEL") or ""
    base = args.api_base or env("API_BASE") or os.environ.get("RB_API_BASE") or ""
    key = args.api_key or env("API_KEY") or os.environ.get("RB_API_KEY") or ""
    return model, base, key


def post(url, key, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, time.time() - t0


def _dump(tag, d, raw):
    if raw:
        print(f"{DIM}--- raw {tag} response ---{RESET}")
        print(json.dumps(d, indent=2, ensure_ascii=False))
        print(f"{DIM}--- end {tag} ---{RESET}")


def probe_chat(url, key, model, timeout, raw=False):
    # Budget must be generous: thinking models spend tokens on reasoning before
    # emitting content, so a tiny max_tokens yields empty content (finish=length).
    body = {"model": model, "messages": [{"role": "user", "content": "Reply with the word: ok"}],
            "max_tokens": 128}
    try:
        d, dt = post(url, key, body, timeout)
        _dump("chat", d, raw)
        ch = d.get("choices", [{}])[0]
        msg = ch.get("message", {})
        content = (msg.get("content") or "").strip()
        rc = (msg.get("reasoning_content") or "").strip()
        fr = ch.get("finish_reason")
        if content:
            return True, f"{dt*1000:.0f}ms, content={content[:40]!r}"
        if rc:
            return True, (f"{dt*1000:.0f}ms, content empty but reasoning_content len={len(rc)} "
                          f"(thinking model; finish_reason={fr})")
        return False, f"200 but empty content & no reasoning (finish_reason={fr}): {json.dumps(d)[:200]}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode()[:160]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:140]}"


def probe_tools(url, key, model, timeout, raw=False):
    body = {"model": model,
            "messages": [{"role": "user", "content": "List files here. Call run_bash."}],
            "tools": [{"type": "function", "function": {
                "name": "run_bash", "description": "Run a bash command",
                "parameters": {"type": "object", "properties": {
                    "command": {"type": "string"}}, "required": ["command"]}}}],
            "tool_choice": "auto", "max_tokens": 256}
    try:
        d, dt = post(url, key, body, timeout)
        _dump("tools", d, raw)
        tc = d.get("choices", [{}])[0].get("message", {}).get("tool_calls")
        if tc:
            return True, f"{dt*1000:.0f}ms, tool_call={tc[0]['function']['name']}"
        return False, "no tool_calls (use --parse thought_action for this model)"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode()[:120]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def probe_thinking(url, key, model, timeout, raw=False):
    body = {"model": model,
            "messages": [{"role": "user", "content": "What is 17*23? Think step by step."}],
            "reasoning_effort": "high", "max_tokens": 600}
    try:
        d, dt = post(url, key, body, timeout)
        _dump("thinking", d, raw)
        msg = d.get("choices", [{}])[0].get("message", {})
        rc = msg.get("reasoning_content")
        u = d.get("usage", {}) or {}
        rtok = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if rc:
            return True, f"{dt*1000:.0f}ms, reasoning_content len={len(rc)} reasoning_tokens={rtok}"
        return False, "no reasoning_content (thinking not enabled by reasoning_effort)"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode()[:120]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def line(label, ok, detail):
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label:9} {DIM}{detail}{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESET_PREFIX), help="preset")
    ap.add_argument("--model-name", help="raw litellm model name (e.g. openai/foo)")
    ap.add_argument("--api-base")
    ap.add_argument("--api-key")
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    ap.add_argument("--thinking", action="store_true", help="also probe reasoning_effort")
    ap.add_argument("--raw", action="store_true",
                    help="print the full untrimmed JSON response from each probe")
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()

    load_env_file(a.env_file)
    model_full, base, key = resolve(a)
    if not model_full or model_full.endswith("/"):
        sys.exit(f"{RED}No model name set.{RESET} Fill <PREFIX>_MODEL in {a.env_file} "
                 f"or pass --model-name (got {model_full!r}).")
    if not base:
        sys.exit(f"{RED}No api_base{RESET} (set RB_API_BASE in {a.env_file}).")
    # The gateway expects the bare model name; strip the litellm provider prefix.
    model = model_full.split("/", 1)[1] if "/" in model_full else model_full
    url = base.rstrip("/") + "/chat/completions"

    print(f"gateway : {url}")
    print(f"model   : {model}  {DIM}(from {model_full}){RESET}")
    print(f"api_key : {key[:6]}…{key[-2:] if len(key) > 8 else ''}  ({len(key)} chars)")
    print("probes  :")

    ok_chat, d = probe_chat(url, key, model, a.timeout, a.raw); line("chat", ok_chat, d)
    # With --raw, run the other probes even if chat looked empty, so you see everything.
    if ok_chat or a.raw:
        ok_tools, d = probe_tools(url, key, model, a.timeout, a.raw); line("tools", ok_tools, d)
        if a.thinking:
            ok_t, d = probe_thinking(url, key, model, a.timeout, a.raw); line("thinking", ok_t, d)

    if ok_chat:
        print(f"\n{GREEN}endpoint healthy for {model}{RESET}")
        return 0
    print(f"\n{RED}endpoint NOT healthy for {model}{RESET}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
