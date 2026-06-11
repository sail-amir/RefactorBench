#!/usr/bin/env python3
"""Post-run diagnostic: status + health of a SWE-agent run.

Reads the artifacts a run produces under runs/<slug>__<variant>/ (the per-task
<id>.traj, plus preds.json / scores.json / *.debug.log / run_batch.config.yaml)
and reports, per task: parser used, exit status, step count, patch size, score,
tokens, wall-clock, and HEALTH flags.

Health is evidence-based, not string-grep guesswork:
  - no_trajectory   : agent never ran (no .traj) -> likely killed before start
  - runtime_failed  : container/swe-rex never started AND no steps recovered
  - empty_patch     : a trajectory exists but produced a 0-byte diff
  - patch_apply_failed / timeout / no_prediction : from the scorer
A first-attempt container timeout that later retried OK is reported as the
informational note `slow_start`, not a failure. The "does not support function
calling" warning is advisory (function_calling still works), so it is NOT a
health flag — the actual parser is read from the run config instead.

Usage:
    python scripts/run_status.py runs/smoke_pangu35b__descriptive
    python scripts/run_status.py smoke_pangu35b      # slug (searches runs/)
    python scripts/run_status.py                     # most recent run
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "runs")
G, R, Y, DIM, RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
TOK = re.compile(r"total_tokens_sent=([\d,]+).*?total_tokens_received=([\d,]+)"
                 r".*?total_api_calls=(\d+)")
PARSER = re.compile(r'"parse_function":\s*\{.*?"type":\s*"([a-z_]+)"', re.DOTALL)


def find_run(arg):
    if arg and os.path.isdir(arg):
        return arg.rstrip("/")
    if arg:
        hits = sorted(glob.glob(os.path.join(RUNS_DIR, f"{arg}*")))
        if hits:
            return hits[-1]
        sys.exit(f"no run dir matching {arg!r} under {RUNS_DIR}")
    runs = [d for d in glob.glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(d)]
    if not runs:
        sys.exit(f"no runs under {RUNS_DIR}")
    return max(runs, key=os.path.getmtime)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def read_text(path):
    try:
        return open(path, encoding="utf-8").read()
    except OSError:
        return ""


def parser_of(run_dir):
    m = PARSER.search(read_text(os.path.join(run_dir, "run_batch.config.yaml")))
    return m.group(1) if m else "?"


def wallclock(text):
    first = last = None
    for ln in text.splitlines():
        m = TS.match(ln)
        if m:
            last = m.group(1)
            first = first or m.group(1)
    if not (first and last):
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    return (_dt.datetime.strptime(last, fmt) - _dt.datetime.strptime(first, fmt)).total_seconds()


def scan_log(text):
    sig = set()
    if "Runtime did not start within timeout" in text:
        sig.add("runtime_retry")     # at least one start attempt timed out
    if "</think>" in text:
        sig.add("think_leak")        # reasoning leaked into parsed content
    tok, m = None, None
    for m in TOK.finditer(text):
        pass
    if m:
        tok = {"sent": int(m.group(1).replace(",", "")),
               "recv": int(m.group(2).replace(",", "")),
               "calls": int(m.group(3))}
    return tok, sig


def analyze(run_dir):
    preds = load_json(os.path.join(run_dir, "preds.json")) or {}
    scores = load_json(os.path.join(run_dir, "scores.json")) or {}
    score_by_id = {i["id"]: i for i in scores.get("instances", [])}
    parser = parser_of(run_dir)

    rows = []
    for d in sorted(g for g in glob.glob(os.path.join(run_dir, "*")) if os.path.isdir(g)):
        iid = os.path.basename(d)
        traj = load_json(next(iter(glob.glob(os.path.join(d, "*.traj"))), ""))
        dbg = read_text(next(iter(glob.glob(os.path.join(d, "*.debug.log"))), ""))
        tok, sig = scan_log(dbg)
        secs = wallclock(dbg)
        steps = len(traj.get("trajectory", [])) if traj else None
        exit_status = (traj.get("info") or {}).get("exit_status") if traj else None
        plen = len((preds.get(iid) or {}).get("model_patch") or "")
        sc = score_by_id.get(iid)

        bad, info = set(), set()
        if steps is None:
            bad.add("no_trajectory")
        else:
            if "runtime_retry" in sig and steps == 0:
                bad.add("runtime_failed")
            elif "runtime_retry" in sig:
                info.add("slow_start")          # timed out once, retried OK
            if plen == 0:
                bad.add("empty_patch")
        if sc:
            r0 = sc.get("reason", "").split(":")[0]
            if r0 in ("patch_apply_failed", "timeout", "no_prediction"):
                bad.add(r0)
        if "think_leak" in sig:
            info.add("think_leak")

        rows.append({"id": iid, "steps": steps, "exit": exit_status, "plen": plen,
                     "score": sc, "tok": tok, "secs": secs, "bad": bad, "info": info})
    model = (scores.get("model")
             or next((p.get("model_name_or_path") for p in preds.values()
                      if isinstance(p, dict)), "?"))
    return model, scores.get("variant", "?"), parser, rows


def fmt_secs(s):
    if s is None:
        return "?"
    return f"{int(s//60)}m{int(s%60):02d}s" if s >= 60 else f"{int(s)}s"


def main():
    run_dir = find_run(sys.argv[1] if len(sys.argv) > 1 else None)
    model, variant, parser, rows = analyze(run_dir)
    print(f"run     : {run_dir}")
    print(f"model   : {model}   variant: {variant}   parser: {parser}")
    print(f"tasks   : {len(rows)}")

    n_pass = n_fail = n_unhealthy = 0
    for r in rows:
        sc, bad, info = r["score"], r["bad"], r["info"]
        passed = bool(sc and sc.get("passed"))
        healthy = not bad
        n_pass += passed
        n_fail += bool(sc and not passed)
        n_unhealthy += (not healthy)

        score_str = (f"{G}PASS{RST}" if passed else
                     f"{R}fail{RST} {DIM}{(sc or {}).get('reason','?')[:64]}{RST}" if sc else
                     f"{DIM}unscored{RST}")
        health_str = f"{G}healthy{RST}" if healthy else f"{R}{','.join(sorted(bad))}{RST}"
        if info:
            health_str += f"  {Y}({','.join(sorted(info))}){RST}"
        tok = r["tok"]
        tok_s = f"{tok['sent']//1000}k↑/{tok['recv']//1000}k↓/{tok['calls']}calls" if tok else "?"
        steps = r["steps"] if r["steps"] is not None else "?"
        print(f"\n─ {r['id']}")
        print(f"    exit   : {r['exit'] or DIM+'(none)'+RST}    steps: {steps}    wall: {fmt_secs(r['secs'])}")
        print(f"    patch  : {r['plen']} bytes    tokens: {tok_s}")
        print(f"    score  : {score_str}")
        print(f"    health : {health_str}")

    print(f"\n{'='*52}")
    print(f"summary : {len(rows)} tasks | {G}{n_pass} pass{RST} | {n_fail} fail | "
          f"{(R+str(n_unhealthy)+' unhealthy'+RST) if n_unhealthy else G+'0 unhealthy'+RST}")
    hints = {
        "runtime_failed": "container/swe-rex never started — raise --startup-timeout or check Docker/load",
        "empty_patch":    "agent produced no diff — model struggled or hit a step/cost cap",
        "no_trajectory":  "no trajectory — run likely killed before the agent started",
        "patch_apply_failed": "scorer couldn't apply the diff — check the patch",
        "timeout":        "checker timed out",
        "no_prediction":  "no prediction for a mapped task",
    }
    allbad = set().union(*[r["bad"] for r in rows]) if rows else set()
    for s in sorted(allbad):
        if s in hints:
            print(f"  {Y}!{RST} {s}: {hints[s]}")
    return 1 if n_unhealthy else 0


if __name__ == "__main__":
    raise SystemExit(main())
