#!/usr/bin/env python3
"""Plot run health for RefactorBench: trajectory-step distribution + exit-status mix.

Reads the per-task trajectories (`<run_dir>/<task>/<task>.traj`) of one or two
runs and draws ONE figure with two panels:

  1. Distribution of the number of trajectory steps per task (box plot by
     default: box = quartiles, line = median, diamond = mean, with the raw
     points jittered over it; use --dist hist for overlaid histograms).
  2. Count of each exit status (submitted, submitted (exit_format), exit_format,
     exit_cost, exit_api, exit_context, ...) as grouped bars.

With `--compare RESULT2` the two runs are overlaid (panel 1) / grouped (panel 2)
so you can compare how far agents get and how they fail across experiments.

Usage:
    python scripts/plot_run_health.py runs/glm-5.1__descriptive
    python scripts/plot_run_health.py runs/glm-5.1__descriptive \
        --compare runs/pangu35b__descriptive --labels glm pangu

Notes:
- "steps" = len(trajectory) from the .traj (includes the autosubmit step).
- exit_status comes from each .traj's info.exit_status; an interrupted/empty
  trajectory shows as `unknown`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# statuses that mean "the agent ran but the harness/agent gave up" -> highlight
FAILURE_HINTS = ("exit_format", "exit_cost", "exit_api", "exit_context",
                 "exit_environment", "exit_error", "exit_forfeit", "unknown")
COLORS = ["#1f77b4", "#ff7f0e"]


def load_run(path: str) -> dict:
    """Accept a run dir or a scores.json path; scan its .traj files."""
    if path.endswith(".json"):
        rundir = os.path.dirname(os.path.abspath(path))
    elif os.path.isdir(path):
        rundir = os.path.abspath(path)
    else:
        sys.exit(f"not a run dir or scores.json: {path}")
    trajs = sorted(glob.glob(os.path.join(rundir, "**", "*.traj"), recursive=True))
    if not trajs:
        sys.exit(f"no .traj files found under {rundir} "
                 f"(trajectories are gitignored — run must exist locally)")
    steps, exits = [], collections.Counter()
    for tp in trajs:
        try:
            d = json.load(open(tp, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        steps.append(len(d.get("trajectory", [])))
        es = (d.get("info", {}) or {}).get("exit_status") or "unknown"
        exits[es] += 1
    return {"slug": os.path.basename(rundir), "dir": rundir,
            "steps": steps, "exits": exits, "n": len(steps)}


def _is_failure(status: str) -> bool:
    return any(h in status for h in FAILURE_HINTS)


def plot(runs, labels, out, bins=20, dist="box", title=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # ---- panel 1: step-count distribution ----
    if dist == "box":
        data = [r["steps"] or [0] for r in runs]
        pos = list(range(len(runs)))
        bp = ax1.boxplot(data, positions=pos, widths=0.5, patch_artist=True,
                         showmeans=True, meanprops=dict(marker="D", markersize=5,
                         markerfacecolor="white", markeredgecolor="black"),
                         medianprops=dict(color="black", lw=1.5),
                         flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        rng = random.Random(0)  # deterministic jitter for the overlaid points
        for i, r in enumerate(runs):
            xs = [i + (rng.random() - 0.5) * 0.28 for _ in r["steps"]]
            ax1.scatter(xs, r["steps"], s=12, color=COLORS[i], alpha=0.6,
                        edgecolor="white", linewidth=0.3, zorder=3)
            if r["steps"]:
                med = statistics.median(r["steps"])
                ax1.annotate(f" med {med:.0f}", (i + 0.28, med), fontsize=8,
                             va="center", color=COLORS[i])
        ax1.set_xticks(pos)
        ax1.set_xticklabels([f"{lab}\n(n={r['n']})" for lab, r in zip(labels, runs)],
                            fontsize=9)
        ax1.set_ylabel("trajectory steps")
        ax1.set_title("Step-count distribution  (box = quartiles, ♦ = mean)")
    else:
        allsteps = [s for r in runs for s in r["steps"]] or [0]
        hi = max(allsteps)
        edges = [i * hi / bins for i in range(bins + 1)] if hi else [0, 1]
        for r, lab, c in zip(runs, labels, COLORS):
            if not r["steps"]:
                continue
            med = statistics.median(r["steps"])
            mean = statistics.mean(r["steps"])
            ax1.hist(r["steps"], bins=edges, alpha=0.55, color=c, edgecolor="white",
                     label=f"{lab}  (n={r['n']}, med {med:.0f}, mean {mean:.0f})")
            ax1.axvline(med, color=c, ls="--", lw=1.3)
        ax1.set_xlabel("trajectory steps")
        ax1.set_ylabel("# tasks")
        ax1.set_title("Step-count distribution")
        ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(axis="y", ls=":", alpha=0.4)
    ax1.set_axisbelow(True)

    # ---- panel 2: exit-status counts (grouped) ----
    cats = set()
    for r in runs:
        cats |= set(r["exits"])
    # order: clean "submitted" first, then by total frequency desc
    def order_key(cat):
        tot = sum(r["exits"].get(cat, 0) for r in runs)
        return (cat == "submitted" and not _is_failure(cat), tot)
    cats = sorted(cats, key=lambda c: (-(c == "submitted"),
                                       -sum(r["exits"].get(c, 0) for r in runs), c))
    x = range(len(cats))
    grp = 0.8
    w = grp / len(runs)
    for i, (r, lab, c) in enumerate(zip(runs, labels, COLORS)):
        heights = [r["exits"].get(cat, 0) for cat in cats]
        xs = [xi - grp / 2 + w / 2 + i * w for xi in x]
        bars = ax2.bar(xs, heights, w, color=c, label=lab, edgecolor="white",
                       linewidth=0.5)
        for b, h in zip(bars, heights):
            if h:
                ax2.text(b.get_x() + b.get_width() / 2, h, str(h),
                         ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(list(x))
    # red tick labels for failure statuses
    ax2.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
    for lbl, cat in zip(ax2.get_xticklabels(), cats):
        if _is_failure(cat):
            lbl.set_color("#b00020")
    ax2.set_ylabel("# tasks")
    ax2.set_title("Exit-status mix  (red = failure/health)")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", ls=":", alpha=0.4)
    ax2.set_axisbelow(True)

    if title is None:
        title = "Run health — " + "  vs  ".join(
            f"{lab} ({r['n']} tasks)" for r, lab in zip(runs, labels))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def _print_summary(r, lab):
    s = r["steps"]
    if s:
        print(f"\n[{lab}]  n={r['n']}  steps: min {min(s)} / med "
              f"{statistics.median(s):.0f} / mean {statistics.mean(s):.0f} / max {max(s)}")
    print(f"  exit statuses:")
    for cat, cnt in sorted(r["exits"].items(), key=lambda kv: -kv[1]):
        mark = "  <-fail" if _is_failure(cat) else ""
        print(f"    {cnt:3d}  {cat}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="run dir (or scores.json) to scan for .traj files")
    ap.add_argument("--compare", metavar="RESULT2", help="second run for A-vs-B comparison")
    ap.add_argument("--labels", nargs=2, metavar=("A", "B"),
                    help="labels for the runs (default: their slugs)")
    ap.add_argument("--out", help="output png (default <result_dir>/run_health.png)")
    ap.add_argument("--dist", choices=["box", "hist"], default="box",
                    help="panel-1 style for the step distribution (default box)")
    ap.add_argument("--bins", type=int, default=20, help="hist bins (--dist hist; default 20)")
    ap.add_argument("--title", help="override figure title")
    a = ap.parse_args()

    runs = [load_run(a.result)]
    if a.compare:
        runs.append(load_run(a.compare))
    labels = a.labels or [r["slug"] for r in runs]

    for r, lab in zip(runs, labels):
        _print_summary(r, lab)

    out = a.out or os.path.join(runs[0]["dir"], "run_health.png")
    plot(runs, labels, out, bins=a.bins, dist=a.dist, title=a.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
