#!/usr/bin/env python3
"""Token-consumption report for one or more RefactorBench runs.

Per run (given an experiment name, run dir, or scores.json) it reports:
  - TOTAL input / output tokens across all trajectory steps,
  - mean and median input / output tokens PER trajectory step.

Two sources are shown because they measure different things:
  1. Gateway-reported (info.model_stats: tokens_sent / tokens_received / api_calls)
     — the provider's own usage. NOTE: under streaming many gateways don't return
     `usage`, so tokens_received is often badly under-reported (this is why we also
     tokenize). These give per-task totals only, not per-step.
  2. Tokenized per step (tiktoken `cl100k_base`) — we re-tokenize each step's
     `query` (the full prompt sent that call) and `response` (the text returned),
     which yields true per-step distributions (mean/median). It's an approximation
     (the gateway's tokenizer may differ; reasoning_content is not in `response`).

Usage:
    python scripts/token_usage.py glm-5.1            # name -> searches runs/
    python scripts/token_usage.py runs/glm-5.1__descriptive --per-task
    python scripts/token_usage.py glm-5.1 pangu35b   # several -> comparison table
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def ntok(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))
    TOKENIZER = "tiktoken cl100k_base"
except Exception:  # pragma: no cover - fallback if tiktoken missing
    def ntok(text: str) -> int:
        return max(0, round(len(text) / 4))   # ~4 chars/token heuristic
    TOKENIZER = "chars/4 heuristic (tiktoken unavailable)"


def resolve_run(name: str) -> str:
    """Accept a run dir, a scores.json, or an experiment name -> return run dir."""
    if name.endswith(".json"):
        return os.path.dirname(os.path.abspath(name))
    if os.path.isdir(name):
        return os.path.abspath(name)
    cand = os.path.join(RUNS_DIR, name)
    if os.path.isdir(cand):
        return cand
    matches = sorted(d for d in glob.glob(os.path.join(RUNS_DIR, f"*{name}*"))
                     if os.path.isdir(d))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(f"no run found for {name!r} (looked in {RUNS_DIR})")
    sys.exit(f"{name!r} is ambiguous; matches:\n  " +
             "\n  ".join(os.path.basename(m) for m in matches))


def _msg_text(m) -> str:
    parts = []
    c = m.get("content")
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        for p in c:
            parts.append(str(p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p))
    for tc in (m.get("tool_calls") or []):
        try:
            fn = tc["function"]
            parts.append(f"{fn.get('name', '')} {fn.get('arguments', '')}")
        except (KeyError, TypeError):
            pass
    return " ".join(parts)


def _query_tokens(query) -> int:
    if isinstance(query, list):
        return ntok(" ".join(_msg_text(m) for m in query if isinstance(m, dict)))
    if isinstance(query, str):
        return ntok(query)
    return 0


def analyze_run(rundir: str) -> dict:
    trajs = sorted(glob.glob(os.path.join(rundir, "**", "*.traj"), recursive=True))
    if not trajs:
        sys.exit(f"no .traj files under {rundir} (trajectories are gitignored — "
                 f"run must exist locally)")
    in_steps, out_steps = [], []          # tokenized per-step counts
    sent = recv = calls = 0               # gateway-reported totals
    n_tasks = 0
    per_task = []
    for tp in trajs:
        try:
            d = json.load(open(tp, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        n_tasks += 1
        ms = d.get("info", {}).get("model_stats", {}) or {}
        sent += ms.get("tokens_sent", 0) or 0
        recv += ms.get("tokens_received", 0) or 0
        calls += ms.get("api_calls", 0) or 0
        t_in, t_out, steps = 0, 0, 0
        for s in d.get("trajectory", []):
            qi = _query_tokens(s.get("query"))
            ro = ntok(s.get("response") or "") if isinstance(s.get("response"), str) else 0
            in_steps.append(qi)
            out_steps.append(ro)
            t_in += qi
            t_out += ro
            steps += 1
        per_task.append({"id": os.path.basename(tp)[:-5], "steps": steps,
                         "tok_in": t_in, "tok_out": t_out,
                         "sent": ms.get("tokens_sent", 0), "recv": ms.get("tokens_received", 0)})
    return {"slug": os.path.basename(rundir), "n_tasks": n_tasks,
            "n_steps": len(in_steps), "in_steps": in_steps, "out_steps": out_steps,
            "sent": sent, "recv": recv, "calls": calls, "per_task": per_task}


def _stats(vals):
    if not vals:
        return (0, 0.0, 0.0)
    return (sum(vals), statistics.mean(vals), statistics.median(vals))


def print_report(r: dict, per_task: bool):
    ti, mi, mdi = _stats(r["in_steps"])
    to, mo, mdo = _stats(r["out_steps"])
    print(f"\n=== {r['slug']} ===")
    print(f"tasks: {r['n_tasks']}   trajectory steps: {r['n_steps']}")
    print(f"\n  gateway-reported (model_stats, per-task totals summed):")
    print(f"    input  (tokens_sent)     : {r['sent']:>12,}")
    print(f"    output (tokens_received) : {r['recv']:>12,}   (unreliable under streaming)")
    print(f"    api_calls                : {r['calls']:>12,}")
    print(f"\n  tokenized per step ({TOKENIZER}):")
    print(f"    total input  : {ti:>12,}     total output : {to:>12,}")
    print(f"    input  / step: mean {mi:8.1f}   median {mdi:8.1f}")
    print(f"    output / step: mean {mo:8.1f}   median {mdo:8.1f}")
    if per_task:
        print(f"\n  per task (tokenized):")
        print(f"    {'id':<48} {'steps':>5} {'tok_in':>10} {'tok_out':>9}")
        for t in sorted(r["per_task"], key=lambda x: -x["tok_in"]):
            print(f"    {t['id'][:48]:<48} {t['steps']:>5} {t['tok_in']:>10,} {t['tok_out']:>9,}")


def print_compare(runs):
    print(f"\n{'run':<28} {'tasks':>5} {'steps':>6} {'tot_in':>12} {'tot_out':>11} "
          f"{'in/step(med)':>12} {'out/step(med)':>13}")
    print("-" * 92)
    for r in runs:
        ti, mi, mdi = _stats(r["in_steps"])
        to, mo, mdo = _stats(r["out_steps"])
        print(f"{r['slug'][:28]:<28} {r['n_tasks']:>5} {r['n_steps']:>6} {ti:>12,} {to:>11,} "
              f"{mdi:>12.0f} {mdo:>13.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="experiment name(s), run dir(s), or scores.json")
    ap.add_argument("--per-task", action="store_true", help="also print a per-task token table")
    ap.add_argument("--json", dest="as_json", action="store_true", help="emit machine-readable JSON")
    a = ap.parse_args()

    results = [analyze_run(resolve_run(n)) for n in a.runs]

    if a.as_json:
        out = []
        for r in results:
            ti, mi, mdi = _stats(r["in_steps"])
            to, mo, mdo = _stats(r["out_steps"])
            out.append({"run": r["slug"], "tasks": r["n_tasks"], "steps": r["n_steps"],
                        "gateway": {"tokens_sent": r["sent"], "tokens_received": r["recv"],
                                    "api_calls": r["calls"]},
                        "tokenized": {"total_input": ti, "total_output": to,
                                      "input_per_step_mean": mi, "input_per_step_median": mdi,
                                      "output_per_step_mean": mo, "output_per_step_median": mdo}})
        print(json.dumps(out, indent=2))
        return 0

    for r in results:
        print_report(r, a.per_task)
    if len(results) > 1:
        print_compare(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
