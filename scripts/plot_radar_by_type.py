#!/usr/bin/env python3
"""Plot a radar chart comparing RefactorBench runs by refactoring type.

Each spoke is a refactoring type from analysis/descriptive_task_types.jsonl.
Each run is drawn as a polygon whose radius is the success rate for that type.

By default, type axes where every run solved zero tasks are removed. This keeps
the radar chart focused on categories where at least one compared run succeeded.
Use --keep-zero-axes to show those axes anyway.

Usage:
    python scripts/plot_radar_by_type.py \
      runs/pangu-7b-code-nocode-10k-del-attempt1__descriptive \
      runs/pangu-7b-5k-refactoring-mini-full__descriptive \
      --labels swe-agent mini \
      --out runs/pangu-mini-radar.png

    python scripts/plot_radar_by_type.py runs/runA runs/runB runs/runC \
      --labels A B C --sort support --min-support 3 \
      --label-font-size 12 --label-pad 36 --no-shading
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_TAXONOMY = os.path.join(REPO_ROOT, "analysis", "descriptive_task_types.jsonl")
DEFAULT_LABEL_FONT_SIZE = 10
DEFAULT_RADIAL_FONT_SIZE = 9
DEFAULT_LEGEND_FONT_SIZE = 10
DEFAULT_TITLE_FONT_SIZE = 14
DEFAULT_LABEL_PAD = 28
DEFAULT_LABEL_WIDTH = 16


def load_scores(path: str) -> tuple[dict[str, bool], dict[str, Any]]:
    """Accept a run dir or scores.json path; return ({task_id: passed}, meta)."""
    if os.path.isdir(path):
        path = os.path.join(path, "scores.json")
    if not os.path.isfile(path):
        sys.exit(f"no scores.json found at: {path}")
    data = json.load(open(path, encoding="utf-8"))
    scored = {inst["id"]: bool(inst["passed"]) for inst in data.get("instances", [])}
    if not scored:
        sys.exit(f"{path} has no scored instances")
    total = data.get("total", {}) or {}
    meta = {
        "path": path,
        "slug": os.path.basename(os.path.dirname(os.path.abspath(path))),
        "model": data.get("model", "?"),
        "variant": data.get("variant", "?"),
        "n": total.get("n", len(scored)),
        "passed": total.get("passed", sum(scored.values())),
        "pass_rate": total.get("pass_rate"),
    }
    return scored, meta


def load_taxonomy(path: str) -> dict[str, list[str]]:
    if not os.path.isfile(path):
        sys.exit(f"taxonomy file not found: {path}")
    out: dict[str, list[str]] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["id"]] = list(row.get("types") or [])
    return out


def aggregate(scored: dict[str, bool], taxonomy: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Return type -> {'passed': int, 'total': int}."""
    agg: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"passed": 0, "total": 0})
    missing = []
    for task_id, passed in scored.items():
        types = taxonomy.get(task_id)
        if not types:
            missing.append(task_id)
            continue
        for typ in types:
            agg[typ]["total"] += 1
            if passed:
                agg[typ]["passed"] += 1
    if missing:
        print(
            f"WARNING: {len(missing)} scored task(s) absent from taxonomy; skipped: "
            + ", ".join(sorted(missing)[:8])
            + (" ..." if len(missing) > 8 else ""),
            file=sys.stderr,
        )
    return agg


def select_types(
    aggs: list[dict[str, dict[str, int]]],
    min_support: int,
    keep_zero_axes: bool,
    sort: str,
) -> list[str]:
    all_types = sorted(set().union(*(set(a) for a in aggs)))
    rows = []
    for typ in all_types:
        totals = [a.get(typ, {}).get("total", 0) for a in aggs]
        passes = [a.get(typ, {}).get("passed", 0) for a in aggs]
        max_total = max(totals or [0])
        total_passes = sum(passes)
        if max_total < min_support:
            continue
        if not keep_zero_axes and total_passes == 0:
            continue
        rates = [
            (a.get(typ, {}).get("passed", 0) / a.get(typ, {}).get("total", 1))
            if a.get(typ, {}).get("total", 0) else 0.0
            for a in aggs
        ]
        rows.append((typ, max(rates), sum(totals), max_total, total_passes))
    if not rows:
        sys.exit("no refactoring types remain after filtering")
    if sort == "support":
        rows.sort(key=lambda r: (-r[2], r[0]))
    elif sort == "name":
        rows.sort(key=lambda r: r[0])
    elif sort == "max-rate":
        rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    elif sort == "passes":
        rows.sort(key=lambda r: (-r[4], -r[2], r[0]))
    else:
        raise ValueError(sort)
    return [r[0] for r in rows]


def wrap_label(text: str, width: int = 18) -> str:
    words = text.replace("/", "/ ").split()
    lines: list[str] = []
    cur = ""
    for word in words:
        next_cur = word if not cur else f"{cur} {word}"
        if len(next_cur) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = next_cur
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def axis_labels(
    types: list[str],
    aggs: list[dict[str, dict[str, int]]],
    label_width: int,
) -> list[str]:
    labels = []
    for typ in types:
        supports = [a.get(typ, {}).get("total", 0) for a in aggs]
        nonzero = [n for n in supports if n]
        if not nonzero:
            support = "n=0"
        elif min(nonzero) == max(nonzero):
            support = f"n={nonzero[0]}"
        else:
            support = f"n={min(nonzero)}-{max(nonzero)}"
        labels.append(f"{wrap_label(typ, width=label_width)}\n{support}")
    return labels


def values_for(agg: dict[str, dict[str, int]], types: list[str]) -> list[float]:
    vals = []
    for typ in types:
        item = agg.get(typ, {})
        total = item.get("total", 0)
        vals.append((item.get("passed", 0) / total) if total else 0.0)
    return vals


def align_axis_labels(ax: Any, angles: list[float]) -> None:
    for label, theta in zip(ax.get_xticklabels(), angles):
        display_theta = math.pi / 2 - theta
        x = math.cos(display_theta)
        y = math.sin(display_theta)
        if x > 0.20:
            label.set_horizontalalignment("left")
        elif x < -0.20:
            label.set_horizontalalignment("right")
        else:
            label.set_horizontalalignment("center")
        if y > 0.70:
            label.set_verticalalignment("bottom")
        elif y < -0.70:
            label.set_verticalalignment("top")
        else:
            label.set_verticalalignment("center")


def print_table(types: list[str], labels: list[str], aggs: list[dict[str, dict[str, int]]]) -> None:
    width = max(len(t) for t in types)
    print(f"\n{'type':<{width}}  " + "  ".join(f"{lab:>18}" for lab in labels))
    print("-" * (width + 22 * len(labels) + 2))
    for typ in types:
        cells = []
        for agg in aggs:
            item = agg.get(typ, {})
            p = item.get("passed", 0)
            n = item.get("total", 0)
            rate = (p / n * 100) if n else 0.0
            cells.append(f"{rate:5.1f}% ({p}/{n})")
        print(f"{typ:<{width}}  " + "  ".join(f"{cell:>18}" for cell in cells))


def plot_radar(
    types: list[str],
    aggs: list[dict[str, dict[str, int]]],
    metas: list[dict[str, Any]],
    labels: list[str],
    out: str,
    title: str | None,
    fill_alpha: float,
    label_font_size: int,
    radial_font_size: int,
    legend_font_size: int,
    title_font_size: int,
    label_pad: int,
    label_width: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            sys.exit("matplotlib is required for radar plots; install it with: python -m pip install matplotlib")
        raise

    n_axes = len(types)
    angles = [2 * math.pi * i / n_axes for i in range(n_axes)]
    angles_closed = angles + [angles[0]]

    fig_size = max(
        9.0,
        min(16.0, 7.0 + n_axes * 0.35 + max(0, label_font_size - 10) * 0.2),
    )
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    colors = plt.cm.tab10.colors
    for idx, (agg, label, meta) in enumerate(zip(aggs, labels, metas)):
        vals = values_for(agg, types)
        vals_closed = vals + [vals[0]]
        color = colors[idx % len(colors)]
        overall = meta.get("pass_rate")
        legend_label = label
        if overall is not None:
            legend_label = f"{label} ({overall * 100:.0f}%)"
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2.0, label=legend_label)
        if fill_alpha > 0:
            ax.fill(angles_closed, vals_closed, color=color, alpha=fill_alpha)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=radial_font_size)
    ax.set_rlabel_position(90)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels(types, aggs, label_width=label_width), fontsize=label_font_size)
    ax.tick_params(axis="x", pad=label_pad)
    align_axis_labels(ax, angles)
    ax.grid(True, linestyle=":", alpha=0.6)

    if title is None:
        title = "RefactorBench Success Rate By Refactoring Type"
    ax.set_title(title, y=1.14, fontsize=title_font_size)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.27, 1.17),
        fontsize=legend_font_size,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def default_out(first_result: str) -> str:
    if os.path.isdir(first_result):
        rundir = first_result
    else:
        rundir = os.path.dirname(first_result)
    return os.path.join(rundir, "success_radar_by_type.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+",
                    help="run dirs or scores.json paths; pass two or more for comparison")
    ap.add_argument("--labels", nargs="+",
                    help="legend labels, one per result (default: run slugs)")
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY,
                    help="task->types jsonl (default analysis/descriptive_task_types.jsonl)")
    ap.add_argument("--out", help="output PNG (default <first_run>/success_radar_by_type.png)")
    ap.add_argument("--sort", choices=["support", "name", "max-rate", "passes"],
                    default="support",
                    help="spoke order (default support)")
    ap.add_argument("--min-support", type=int, default=1,
                    help="drop types whose maximum support across runs is below N")
    ap.add_argument("--keep-zero-axes", action="store_true",
                    help="keep type axes where every compared run solved zero tasks")
    ap.add_argument("--fill-alpha", type=float, default=0.10,
                    help="polygon fill alpha; use 0 for no fill")
    ap.add_argument("--no-shading", "--no-fill", action="store_true",
                    help="draw only radar outlines, with no filled polygon shading")
    ap.add_argument("--label-font-size", type=int, default=DEFAULT_LABEL_FONT_SIZE,
                    help=f"refactoring-type label font size (default {DEFAULT_LABEL_FONT_SIZE})")
    ap.add_argument("--radial-font-size", type=int, default=DEFAULT_RADIAL_FONT_SIZE,
                    help=f"radial percent label font size (default {DEFAULT_RADIAL_FONT_SIZE})")
    ap.add_argument("--legend-font-size", type=int, default=DEFAULT_LEGEND_FONT_SIZE,
                    help=f"legend font size (default {DEFAULT_LEGEND_FONT_SIZE})")
    ap.add_argument("--title-font-size", type=int, default=DEFAULT_TITLE_FONT_SIZE,
                    help=f"title font size (default {DEFAULT_TITLE_FONT_SIZE})")
    ap.add_argument("--label-pad", type=int, default=DEFAULT_LABEL_PAD,
                    help=f"outward padding for refactoring-type labels (default {DEFAULT_LABEL_PAD})")
    ap.add_argument("--label-width", type=int, default=DEFAULT_LABEL_WIDTH,
                    help=f"wrap refactoring-type labels after this many chars (default {DEFAULT_LABEL_WIDTH})")
    ap.add_argument("--title", help="override chart title")
    args = ap.parse_args()

    if len(args.results) < 2:
        print("WARNING: radar chart is most useful with two or more runs", file=sys.stderr)
    if args.labels and len(args.labels) != len(args.results):
        sys.exit("--labels must provide exactly one label per result")
    if min(
        args.label_font_size,
        args.radial_font_size,
        args.legend_font_size,
        args.title_font_size,
        args.label_width,
    ) <= 0:
        sys.exit("font sizes and --label-width must be positive integers")
    if args.label_pad < 0:
        sys.exit("--label-pad must be >= 0")
    if args.fill_alpha < 0:
        sys.exit("--fill-alpha must be >= 0")

    taxonomy = load_taxonomy(args.taxonomy)
    scored_and_meta = [load_scores(path) for path in args.results]
    scored = [x[0] for x in scored_and_meta]
    metas = [x[1] for x in scored_and_meta]
    labels = args.labels or [m["slug"] for m in metas]
    aggs = [aggregate(s, taxonomy) for s in scored]
    types = select_types(
        aggs,
        min_support=args.min_support,
        keep_zero_axes=args.keep_zero_axes,
        sort=args.sort,
    )

    print_table(types, labels, aggs)
    removed = sorted(
        t for t in set().union(*(set(a) for a in aggs))
        if t not in types and sum(a.get(t, {}).get("passed", 0) for a in aggs) == 0
    )
    if removed and not args.keep_zero_axes:
        print(f"\nremoved zero-pass axes ({len(removed)}): {', '.join(removed)}")

    out = args.out or default_out(args.results[0])
    fill_alpha = 0.0 if args.no_shading else args.fill_alpha
    plot_radar(
        types,
        aggs,
        metas,
        labels,
        out,
        title=args.title,
        fill_alpha=fill_alpha,
        label_font_size=args.label_font_size,
        radial_font_size=args.radial_font_size,
        legend_font_size=args.legend_font_size,
        title_font_size=args.title_font_size,
        label_pad=args.label_pad,
        label_width=args.label_width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
