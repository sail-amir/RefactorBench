#!/usr/bin/env python3
"""Compare pass rate and loop/control metrics across RefactorBench runs.

Accepts run directories, scores.json paths, or run-name substrings under runs/.
Works with Mini-SWE-Agent runs that store control metrics in preds.json and
best-effort SWE-agent trajectories under *.traj.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


def resolve_run(name: str) -> Path:
    p = Path(name)
    if p.name == "scores.json":
        return p.resolve().parent
    if p.is_dir():
        return p.resolve()
    direct = RUNS_DIR / name
    if direct.is_dir():
        return direct
    hits = sorted(d for d in RUNS_DIR.glob(f"*{name}*") if d.is_dir())
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"no run found for {name!r}")
    sys.exit(f"{name!r} is ambiguous:\n  " + "\n  ".join(h.name for h in hits))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def norm_action(action: str) -> str:
    return "\n".join(line.rstrip() for line in action.strip().splitlines()).strip()


def metrics_from_actions(actions: list[str], empty: int, calls: int,
                         sent: int = 0, recv: int = 0) -> dict[str, Any]:
    vals = [norm_action(a) for a in actions if norm_action(a)]
    hashes = [hashlib.sha256(v.encode("utf-8", "replace")).hexdigest() for v in vals]
    seen: set[str] = set()
    dup = 0
    last = None
    streak = max_streak = 0
    for h in hashes:
        if h in seen:
            dup += 1
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
        "duplicate_actions": dup,
        "duplicate_action_rate": dup / n if n else 0.0,
        "max_repeated_action_streak": max_streak if n else 0,
        "empty_responses": empty,
        "empty_response_rate": empty / calls if calls else 0.0,
        "model_calls": calls,
        "tokens_sent": sent,
        "tokens_received": recv,
    }


def extract_action(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("command", "cmd", "action", "arguments"):
            if key in value:
                return extract_action(value[key])
        return json.dumps(value, sort_keys=True)
    return str(value)


def parse_swe_traj(path: Path) -> dict[str, Any]:
    data = load_json(path) or {}
    actions: list[str] = []
    empty = 0
    for step in data.get("trajectory", []) or []:
        if not isinstance(step, dict):
            continue
        response = step.get("response")
        if isinstance(response, str) and not response.strip():
            empty += 1
        for key in ("action", "tool_call", "command"):
            if key in step:
                actions.append(extract_action(step[key]))
                break
        else:
            resp = step.get("response")
            if isinstance(resp, str):
                m = re.search(r"```(?:bash|sh|mswea_bash_command)?\s*\n(.*?)\n```", resp, re.S)
                if m:
                    actions.append(m.group(1))
    ms = (data.get("info") or {}).get("model_stats") or {}
    metrics = metrics_from_actions(
        actions, empty, int(ms.get("api_calls") or len(data.get("trajectory", []) or [])),
        int(ms.get("tokens_sent") or 0), int(ms.get("tokens_received") or 0),
    )
    metrics["stop_reason"] = (data.get("info") or {}).get("exit_status") or "unknown"
    return metrics


def parse_mini_traj(path: Path) -> dict[str, Any]:
    data = load_json(path) or {}
    actions: list[str] = []
    empty = 0
    calls = 0
    sent = recv = 0
    for msg in data.get("messages", []) or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        calls += 1
        if not str(msg.get("content") or "").strip():
            empty += 1
        extra = msg.get("extra") or {}
        for action in extra.get("actions") or []:
            actions.append(extract_action(action))
        usage = ((extra.get("response") or {}).get("usage") or {})
        sent += int(usage.get("prompt_tokens") or 0)
        recv += int(usage.get("completion_tokens") or 0)
    info = data.get("info") or {}
    ms = info.get("model_stats") or {}
    metrics = metrics_from_actions(
        actions, empty, int(ms.get("api_calls") or calls),
        int(ms.get("tokens_sent") or sent), int(ms.get("tokens_received") or recv),
    )
    metrics["stop_reason"] = info.get("exit_status") or "unknown"
    return metrics


def controls_from_preds(run: Path) -> dict[str, dict[str, Any]]:
    preds = load_json(run / "preds.json") or {}
    out = {}
    for iid, rec in preds.items():
        if isinstance(rec, dict) and isinstance(rec.get("control"), dict):
            c = dict(rec["control"])
            c["stop_reason"] = rec.get("stop_reason") or rec.get("exit_status") or "unknown"
            out[iid] = c
    return out


def controls_from_trajs(run: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in run.glob("*/*.traj.json"):
        out[p.parent.name] = parse_mini_traj(p)
    for p in run.glob("*/*.traj"):
        out[p.parent.name] = parse_swe_traj(p)
    return out


def summarize(run: Path) -> dict[str, Any]:
    scores = load_json(run / "scores.json") or {}
    total = scores.get("total") or {}
    controls = controls_from_preds(run)
    controls.update({k: v for k, v in controls_from_trajs(run).items() if k not in controls})
    actions = sum(int(c.get("actions") or 0) for c in controls.values())
    dup = sum(int(c.get("duplicate_actions") or 0) for c in controls.values())
    calls = sum(int(c.get("model_calls") or 0) for c in controls.values())
    empty = sum(int(c.get("empty_responses") or 0) for c in controls.values())
    sent = sum(int(c.get("tokens_sent") or 0) for c in controls.values())
    recv = sum(int(c.get("tokens_received") or 0) for c in controls.values())
    streaks = [int(c.get("max_repeated_action_streak") or 0) for c in controls.values()]
    stops = Counter(str(c.get("stop_reason") or "unknown") for c in controls.values())
    return {
        "run": run.name,
        "passed": int(total.get("passed") or 0),
        "n": int(total.get("n") or len(controls) or 0),
        "controls": len(controls),
        "actions": actions,
        "calls": calls,
        "duplicate_rate": dup / actions if actions else 0.0,
        "max_streak": max(streaks) if streaks else 0,
        "empty_rate": empty / calls if calls else 0.0,
        "tokens_sent": sent,
        "tokens_received": recv,
        "top_stop": stops.most_common(1)[0][0] if stops else "unknown",
        "stop_counts": dict(stops),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()

    rows = [summarize(resolve_run(r)) for r in args.runs]
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{'run':<34} {'pass':>9} {'ctrl':>5} {'calls':>6} "
          f"{'dup%':>7} {'streak':>6} {'empty%':>7} {'tok_in':>10} {'tok_out':>10} {'top_stop':<18}")
    print("-" * 120)
    for r in rows:
        pass_s = f"{r['passed']}/{r['n']}"
        print(f"{r['run'][:34]:<34} {pass_s:>9} {r['controls']:>5} {r['calls']:>6} "
              f"{100*r['duplicate_rate']:>6.1f}% {r['max_streak']:>6} "
              f"{100*r['empty_rate']:>6.1f}% {r['tokens_sent']:>10,} "
              f"{r['tokens_received']:>10,} {r['top_stop'][:18]:<18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
