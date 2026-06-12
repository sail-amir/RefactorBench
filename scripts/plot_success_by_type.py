#!/usr/bin/env python3
"""Plot success rate per Fowler refactoring type for a RefactorBench run.

Given a result directory (one produced by run_model.py, i.e. containing a
`scores.json`) and the per-task type labels in
`analysis/descriptive_task_types.jsonl`, draw a stacked bar chart of pass/fail
per refactoring type and annotate each bar with its success rate and support
(n = number of scored tasks carrying that type).

Multi-label tasks contribute to EVERY type they carry, so the per-type supports
sum to more than the number of tasks. The dashed line marks the overall
(task-level) pass rate from scores.json.

Usage:
    python scripts/plot_success_by_type.py runs/glm-5.1__descriptive
    python scripts/plot_success_by_type.py runs/<slug>__descriptive/scores.json \
        --mode counts --sort support --out /tmp/by_type.png

    # compare two experiments: grouped A-vs-B stacked bars per type
    python scripts/plot_success_by_type.py runs/glm-5.1__descriptive \
        --compare runs/pangu35b__descriptive --labels glm pangu --sort delta

Notes:
- The type labels are derived from the *descriptive* instructions, but the
  underlying refactor is the same across base/lazy/descriptive, so the mapping
  applies to any variant's run (the task ids are shared).
- `--mode rate` (default): 100%-normalized stacks, so the green fraction *is* the
  success rate. `--mode counts`: raw pass/fail counts, so bar height is support.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_TAXONOMY = os.path.join(REPO_ROOT, "analysis", "descriptive_task_types.jsonl")

PASS_COLOR = "#2ca02c"
FAIL_COLOR = "#d62728"


def load_scores(path: str):
    """Accept a run dir or a scores.json path; return ({id: passed}, meta)."""
    if os.path.isdir(path):
        path = os.path.join(path, "scores.json")
    if not os.path.isfile(path):
        sys.exit(f"no scores.json found at: {path}")
    d = json.load(open(path, encoding="utf-8"))
    scored = {inst["id"]: bool(inst["passed"]) for inst in d.get("instances", [])}
    if not scored:
        sys.exit(f"{path} has no scored instances")
    total = d.get("total", {})
    meta = {
        "model": d.get("model", "?"),
        "variant": d.get("variant", "?"),
        "n": total.get("n", len(scored)),
        "pass_rate": total.get("pass_rate"),
        "slug": os.path.basename(os.path.dirname(os.path.abspath(path))),
        "path": path,
    }
    return scored, meta


def load_taxonomy(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"taxonomy file not found: {path}")
    m = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m[r["id"]] = r["types"]
    return m


def aggregate(scored: dict, taxonomy: dict):
    """type -> [passed, total]; plus list of scored ids missing from taxonomy."""
    agg = collections.defaultdict(lambda: [0, 0])
    missing = []
    for tid, passed in scored.items():
        types = taxonomy.get(tid)
        if not types:
            missing.append(tid)
            continue
        for t in types:
            agg[t][1] += 1
            if passed:
                agg[t][0] += 1
    return agg, missing


def plot(agg, meta, out, mode="rate", sort="rate", min_support=1, title=None):
    items = [(t, p, n) for t, (p, n) in agg.items() if n >= min_support]
    if not items:
        sys.exit("no types meet --min-support; nothing to plot")
    if sort == "rate":
        items.sort(key=lambda x: (-(x[1] / x[2]), -x[2], x[0]))
    elif sort == "support":
        items.sort(key=lambda x: (-x[2], x[0]))
    else:
        items.sort(key=lambda x: x[0])

    types = [i[0] for i in items]
    passed = [i[1] for i in items]
    total = [i[2] for i in items]
    failed = [n - p for p, n in zip(passed, total)]
    rates = [p / n for p, n in zip(passed, total)]
    x = range(len(types))

    if mode == "rate":
        pass_h = [r * 100 for r in rates]
        fail_h = [100 - h for h in pass_h]
        ylabel = "share of tasks (%)"
        ytop = 100
    else:
        pass_h = passed
        fail_h = failed
        ylabel = "# tasks"
        ytop = max(total)

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 0.92), 6))
    ax.bar(x, pass_h, color=PASS_COLOR, label="Passed", edgecolor="white", linewidth=0.5)
    ax.bar(x, fail_h, bottom=pass_h, color=FAIL_COLOR, alpha=0.55,
           label="Failed", edgecolor="white", linewidth=0.5)

    # annotate each bar with success rate + support
    pad = ytop * 0.02
    for xi, (p, n, r) in enumerate(zip(passed, total, rates)):
        top = (100 if mode == "rate" else n)
        ax.text(xi, top + pad, f"{r * 100:.0f}%\nn={n}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", linespacing=1.1)

    # overall (task-level) pass rate reference line (rate mode only)
    if mode == "rate" and meta.get("pass_rate") is not None:
        ovr = meta["pass_rate"] * 100
        ax.axhline(ovr, ls="--", lw=1.2, color="#333333",
                   label=f"overall {ovr:.0f}%")

    ax.set_xticks(list(x))
    ax.set_xticklabels(types, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ytop * 1.18)
    if title is None:
        title = (f"Success rate by refactoring type — {meta['slug']}\n"
                 f"{meta['model']} · {meta['variant']} · "
                 f"{meta['n']} tasks scored"
                 + (f" · overall {meta['pass_rate']*100:.0f}%"
                    if meta.get("pass_rate") is not None else ""))
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_compare(aggA, metaA, aggB, metaB, out, labels, mode="rate",
                 sort="rate", min_support=1, title=None):
    """Per type, two grouped stacked (pass/fail) bars: experiment A vs B.

    A is drawn solid, B hatched; pass=green, fail=red. Each bar is annotated
    with its own success rate and support, which may differ between runs if the
    two runs scored different task subsets.
    """
    labA, labB = labels
    # union of types present (>= min_support) in either experiment, keep stable
    types = sorted(set(aggA) | set(aggB))

    def rate(agg, t):
        p, n = agg.get(t, [0, 0])
        return (p / n) if n else 0.0, p, n

    rows = []
    for t in types:
        rA, pA, nA = rate(aggA, t)
        rB, pB, nB = rate(aggB, t)
        if max(nA, nB) < min_support:
            continue
        rows.append((t, rA, pA, nA, rB, pB, nB))
    if not rows:
        sys.exit("no types meet --min-support; nothing to plot")

    if sort == "rate":
        rows.sort(key=lambda r: (-r[1], -r[3], r[0]))      # by A's rate
    elif sort == "support":
        rows.sort(key=lambda r: (-(r[3] + r[6]), r[0]))    # by combined support
    elif sort == "delta":
        rows.sort(key=lambda r: (-(r[4] - r[1]), r[0]))    # biggest B-minus-A gain
    else:
        rows.sort(key=lambda r: r[0])

    types = [r[0] for r in rows]
    x = list(range(len(types)))
    w = 0.40
    ytop = 100 if mode == "rate" else max(max(r[3], r[6]) for r in rows)

    fig, ax = plt.subplots(figsize=(max(9, len(types) * 1.25), 6.2))

    def draw(offset, hatch, idx_rate, idx_p, idx_n):
        for xi, r in enumerate(rows):
            ra, p, n = r[idx_rate], r[idx_p], r[idx_n]
            if n == 0:
                continue
            ph = ra * 100 if mode == "rate" else p
            fh = (100 - ph) if mode == "rate" else (n - p)
            ax.bar(xi + offset, ph, w, color=PASS_COLOR, hatch=hatch,
                   edgecolor="white", linewidth=0.5)
            ax.bar(xi + offset, fh, w, bottom=ph, color=FAIL_COLOR, alpha=0.55,
                   hatch=hatch, edgecolor="white", linewidth=0.5)
            top = 100 if mode == "rate" else n
            ax.text(xi + offset, top + ytop * 0.015, f"{ra*100:.0f}%\nn={n}",
                    ha="center", va="bottom", fontsize=7, linespacing=1.05)

    draw(-w / 2 - 0.01, None, 1, 2, 3)   # A solid
    draw(+w / 2 + 0.01, "//", 4, 5, 6)   # B hatched

    if mode == "rate":
        if metaA.get("pass_rate") is not None:
            ax.axhline(metaA["pass_rate"] * 100, ls="--", lw=1.0, color="#1b5e20")
        if metaB.get("pass_rate") is not None:
            ax.axhline(metaB["pass_rate"] * 100, ls=":", lw=1.2, color="#0d3b66")

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("share of tasks (%)" if mode == "rate" else "# tasks")
    ax.set_ylim(0, ytop * 1.2)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=PASS_COLOR, label="Passed"),
        Patch(facecolor=FAIL_COLOR, alpha=0.55, label="Failed"),
        Patch(facecolor="#bbbbbb", label=f"A: {labA}"),
        Patch(facecolor="#bbbbbb", hatch="//", label=f"B: {labB}"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9, ncol=2)

    if title is None:
        title = (f"Success rate by refactoring type — A vs B\n"
                 f"A: {labA} ({metaA['n']} tasks"
                 + (f", {metaA['pass_rate']*100:.0f}%" if metaA.get("pass_rate") is not None else "")
                 + f")   vs   B: {labB} ({metaB['n']} tasks"
                 + (f", {metaB['pass_rate']*100:.0f}%" if metaB.get("pass_rate") is not None else "") + ")")
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="run dir (with scores.json) or a scores.json path")
    ap.add_argument("--compare", metavar="RESULT2",
                    help="second run dir/scores.json; draws grouped A-vs-B stacked bars per type")
    ap.add_argument("--labels", nargs=2, metavar=("A", "B"),
                    help="legend labels for the two experiments (default: their slugs)")
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY,
                    help="task->types jsonl (default analysis/descriptive_task_types.jsonl)")
    ap.add_argument("--out", help="output png (default <result_dir>/success_by_type[_compare].png)")
    ap.add_argument("--mode", choices=["rate", "counts"], default="rate",
                    help="rate: 100%% normalized stacks (default); counts: raw pass/fail counts")
    ap.add_argument("--sort", choices=["rate", "support", "name", "delta"], default="rate",
                    help="delta (compare only): sort by B-minus-A success-rate gain")
    ap.add_argument("--min-support", type=int, default=1,
                    help="drop types with fewer than N scored tasks")
    ap.add_argument("--title", help="override chart title")
    a = ap.parse_args()

    taxonomy = load_taxonomy(a.taxonomy)

    def _load(path):
        scored, meta = load_scores(path)
        agg, missing = aggregate(scored, taxonomy)
        if missing:
            print(f"WARNING [{meta['slug']}]: {len(missing)} scored task(s) absent "
                  f"from taxonomy (skipped): {', '.join(sorted(missing)[:6])}"
                  + (" ..." if len(missing) > 6 else ""), file=sys.stderr)
        return agg, meta

    aggA, metaA = _load(a.result)

    if not a.compare:
        out = a.out or os.path.join(os.path.dirname(metaA["path"]), "success_by_type.png")
        rows = sorted(((t, p, n) for t, (p, n) in aggA.items() if n >= a.min_support),
                      key=lambda x: (-(x[1] / x[2]), -x[2]))
        width = max((len(t) for t, _, _ in rows), default=10)
        print(f"\n{'type':<{width}}  rate    passed/support")
        print("-" * (width + 24))
        for t, p, n in rows:
            print(f"{t:<{width}}  {p/n*100:5.1f}%   {p}/{n}")
        if metaA.get("pass_rate") is not None:
            print(f"\noverall task-level pass rate: {metaA['pass_rate']*100:.1f}%  ({metaA['n']} tasks)")
        plot(aggA, metaA, out, mode=a.mode, sort=a.sort,
             min_support=a.min_support, title=a.title)
        return 0

    # --- comparison mode ---
    aggB, metaB = _load(a.compare)
    labels = a.labels or [metaA["slug"], metaB["slug"]]
    out = a.out or os.path.join(os.path.dirname(metaA["path"]), "success_by_type_compare.png")

    types = sorted(set(aggA) | set(aggB))
    width = max((len(t) for t in types), default=10)
    print(f"\n{'type':<{width}}   A rate (p/n)    B rate (p/n)    delta")
    print("-" * (width + 42))
    def fmt(agg, t):
        p, n = agg.get(t, [0, 0])
        return (p / n if n else 0.0, p, n)
    for t in sorted(types, key=lambda t: -(fmt(aggB, t)[0] - fmt(aggA, t)[0])):
        ra, pa, na = fmt(aggA, t); rb, pb, nb = fmt(aggB, t)
        if max(na, nb) < a.min_support:
            continue
        print(f"{t:<{width}}   {ra*100:5.1f}% ({pa}/{na})   {rb*100:5.1f}% ({pb}/{nb})   "
              f"{(rb-ra)*100:+5.1f}")
    print(f"\nA={labels[0]}  overall "
          + (f"{metaA['pass_rate']*100:.1f}%" if metaA.get('pass_rate') is not None else "?")
          + f" ({metaA['n']})   |   B={labels[1]}  overall "
          + (f"{metaB['pass_rate']*100:.1f}%" if metaB.get('pass_rate') is not None else "?")
          + f" ({metaB['n']})")

    plot_compare(aggA, metaA, aggB, metaB, out, labels, mode=a.mode,
                 sort=a.sort, min_support=a.min_support, title=a.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
