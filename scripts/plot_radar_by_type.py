#!/usr/bin/env python3
"""Plot a radar chart comparing RefactorBench runs by refactoring type.

Each spoke is a refactoring type from analysis/descriptive_task_types.jsonl.
Each run is drawn as a polygon whose radius is the success rate for that type.

By default, axes use the canonical 13-category clockwise order used in the
RefactorBench summary figure. Use --drop-zero-axes to remove categories where
every compared run solved zero tasks.

Usage:
    python scripts/plot_radar_by_type.py \
      runs/pangu-7b-code-nocode-10k-del-attempt1__descriptive \
      runs/pangu-7b-5k-refactoring-mini-full__descriptive \
      --labels swe-agent mini \
      --out runs/pangu-mini-radar.png

    python scripts/plot_radar_by_type.py runs/runA runs/runB runs/runC \
      --labels GLM-5.1 pangu-7b-refactoring pangu-35b pangu-7b \
      --out runs/refactorbench_radar.png

    python scripts/plot_radar_by_type.py --config scripts/radar_config.example.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import textwrap
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_TAXONOMY = os.path.join(REPO_ROOT, "analysis", "descriptive_task_types.jsonl")
DEFAULT_LABEL_FONT_SIZE = 15
DEFAULT_RADIAL_FONT_SIZE = 10
DEFAULT_LEGEND_FONT_SIZE = 14
DEFAULT_TITLE_FONT_SIZE = 14
DEFAULT_LABEL_PAD = 20
DEFAULT_LABEL_WIDTH = 16
DEFAULT_LEGEND_LABEL_WIDTH = 16
DEFAULT_DPI = 200
DEFAULT_FIG_WIDTH = 13.0
DEFAULT_FIG_HEIGHT = 14.0
RADAR_AXES_RECT = [0.19, 0.235, 0.62, 0.575714]

CANONICAL_TYPES = [
    "Move Function",
    "Add Parameter (Change Fn Declaration)",
    "Rename Function/Method",
    "Inline / Merge Module",
    "Remove Dead Code",
    "Rename Variable/Constant/Attribute",
    "Rename File/Module",
    "Combine Functions into Class",
    "Introduce Parameter Object",
    "Rename Class",
    "Move Class",
    "Extract Class",
    "Split Function/Phase",
]

TYPE_LABELS = {
    "Move Function": "Move Function",
    "Add Parameter (Change Fn Declaration)": "Add Parameter/\nChange Value/\nDecoupling",
    "Rename Function/Method": "Rename Function/\nMethod",
    "Inline / Merge Module": "Inline / Merge\nModule",
    "Remove Dead Code": "Remove Dead Code",
    "Rename Variable/Constant/Attribute": "Rename Variable/\nConstant/\nAttribute",
    "Rename File/Module": "Rename File/\nModule",
    "Combine Functions into Class": "Combine\nFunctions into\nClass",
    "Introduce Parameter Object": "Introduce\nParameter Object",
    "Rename Class": "Rename Class",
    "Move Class": "Move Class",
    "Extract Class": "Extract Class",
    "Split Function/Phase": "Split Function/\nPhase",
}

ROLE_ORDER = ["frontier", "hero", "baseline35", "baseline7"]
ROLE_DISPLAY = {
    "frontier": "GLM-5.1 (frontier)",
    "hero": "pangu-7b-refactoring",
    "baseline35": "pangu-35b",
    "baseline7": "pangu-7b",
}
ROLE_STYLE = {
    "frontier": dict(color="#1A237E", lw=2.2, ls=(0, (6, 4)), fill=0.0, z=5),
    "hero": dict(color="#1B7F3B", lw=2.6, ls="-", fill=0.40, z=4),
    "baseline35": dict(color="#D98A29", lw=2.2, ls="-", fill=0.07, z=2),
    "baseline7": dict(color="#B22234", lw=2.2, ls=(0, (4, 3)), fill=0.07, z=2),
}
FALLBACK_COLORS = ["#4E79A7", "#59A14F", "#E15759", "#76B7B2", "#F28E2B", "#B07AA1"]
CONFIG_KEYS = {
    "taxonomy",
    "out",
    "sort",
    "min_support",
    "drop_zero_axes",
    "keep_zero_axes",
    "fill_alpha",
    "no_shading",
    "label_font_size",
    "radial_font_size",
    "legend_font_size",
    "no_legend",
    "title_font_size",
    "label_pad",
    "label_width",
    "legend_label_width",
    "fig_width",
    "fig_height",
    "dpi",
    "axes_rect",
    "title",
}


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


def load_yaml_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        sys.exit("PyYAML is required for --config; install it with: python -m pip install pyyaml")
    if not os.path.isfile(path):
        sys.exit(f"radar config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        sys.exit("--config YAML must contain a mapping at the top level")
    return data


def normalize_run_spec(row: Any, idx: int) -> dict[str, Any]:
    if isinstance(row, str):
        return {"path": row}
    if not isinstance(row, dict):
        sys.exit(f"runs[{idx}] must be a path string or mapping")
    spec = dict(row)
    if "path" not in spec:
        for key in ("run", "result", "scores"):
            if key in spec:
                spec["path"] = spec[key]
                break
    if not spec.get("path"):
        sys.exit(f"runs[{idx}] is missing path")
    return spec


def load_run_specs(config: dict[str, Any] | None, results: list[str], labels: list[str] | None) -> list[dict[str, Any]]:
    if config and config.get("runs") is not None:
        if results:
            sys.exit("use either --config runs or positional run paths, not both")
        rows = config["runs"]
        if not isinstance(rows, list) or not rows:
            sys.exit("--config runs must be a non-empty list")
        specs = [normalize_run_spec(row, i) for i, row in enumerate(rows)]
    else:
        specs = [{"path": p} for p in results]

    if labels:
        if len(labels) != len(specs):
            sys.exit("--labels must provide exactly one label per result")
        for spec, label in zip(specs, labels):
            spec["label"] = label
    return specs


def cli_option_was_set(argv: list[str], name: str) -> bool:
    option = "--" + name.replace("_", "-")
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def apply_config_defaults(args: argparse.Namespace, config: dict[str, Any] | None, argv: list[str]) -> None:
    if not config:
        return
    for key in CONFIG_KEYS:
        yaml_key = key if key in config else key.replace("_", "-")
        if yaml_key in config and not cli_option_was_set(argv, key):
            setattr(args, key, config[yaml_key])


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
    present_types = set().union(*(set(a) for a in aggs))
    if sort == "fixed":
        all_types = [t for t in CANONICAL_TYPES if t in present_types or min_support <= 0]
    else:
        all_types = sorted(present_types)
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
    if sort == "fixed":
        pass
    elif sort == "support":
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
    label_width: int,
) -> list[str]:
    return [TYPE_LABELS.get(typ, wrap_label(typ, width=label_width)) for typ in types]


def axis_label_alignment(theta: float) -> tuple[str, str]:
    display_theta = math.pi / 2 - theta
    x = math.cos(display_theta)
    y = math.sin(display_theta)
    if x > 0.20:
        ha = "left"
    elif x < -0.20:
        ha = "right"
    else:
        ha = "center"
    if y > 0.70:
        va = "bottom"
    elif y < -0.70:
        va = "top"
    else:
        va = "center"
    return ha, va


def draw_axis_labels(
    ax: Any,
    angles: list[float],
    labels: list[str],
    label_font_size: int,
    label_pad: int,
) -> None:
    # Keep labels outside the 100% boundary without pushing them past the canvas.
    label_radius = 100 + max(label_pad, int(round(label_font_size * 0.60)))
    for theta, label in zip(angles, labels):
        ha, va = axis_label_alignment(theta)
        ax.text(
            theta,
            label_radius,
            label,
            fontsize=label_font_size,
            ha=ha,
            va=va,
            clip_on=False,
        )


def values_for(agg: dict[str, dict[str, int]], types: list[str]) -> list[float]:
    vals = []
    for typ in types:
        item = agg.get(typ, {})
        total = item.get("total", 0)
        vals.append((item.get("passed", 0) / total * 100.0) if total else 0.0)
    return vals


def role_for_label(label: str) -> str | None:
    text = label.lower().replace("_", "-")
    if "glm" in text:
        return "frontier"
    if "pangu" in text and "refactoring" in text:
        return "hero"
    if "pangu" in text and "35b" in text:
        return "baseline35"
    if "pangu" in text and "7b" in text:
        return "baseline7"
    return None


def normalize_pass_rate(rate: Any) -> float | None:
    if rate is None:
        return None
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return None
    return value * 100.0 if value <= 1.0 else value


def parse_line_style(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        aliases = {
            "solid": "-",
            "-": "-",
            "dashed": "--",
            "dash": "--",
            "--": "--",
            "dotted": ":",
            ":": ":",
            "dashdot": "-.",
            "-.": "-.",
        }
        if value in aliases:
            return aliases[value]
        if "," in value:
            try:
                return (0, tuple(float(x.strip()) for x in value.split(",") if x.strip()))
            except ValueError:
                pass
        return value
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[1], list):
            return (value[0], tuple(value[1]))
        return (0, tuple(value))
    return value


def apply_style_overrides(style: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return style
    out = dict(style)
    mapping = {
        "color": "color",
        "lw": "lw",
        "line_width": "lw",
        "linewidth": "lw",
        "ls": "ls",
        "line_style": "ls",
        "linestyle": "ls",
        "fill": "fill",
        "fill_alpha": "fill",
        "z": "z",
        "zorder": "z",
    }
    for key, value in overrides.items():
        if key not in mapping:
            print(f"WARNING: ignoring unknown style key {key!r}", file=sys.stderr)
            continue
        target = mapping[key]
        out[target] = parse_line_style(value) if target == "ls" else value
    return out


def style_for_series(
    role: str | None,
    idx: int,
    fill_alpha: float | None,
    no_shading: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if role in ROLE_STYLE:
        style = dict(ROLE_STYLE[role])
    else:
        style = dict(
            color=FALLBACK_COLORS[idx % len(FALLBACK_COLORS)],
            lw=2.2,
            ls="-",
            fill=0.07,
            z=3,
        )
    style = apply_style_overrides(style, overrides)
    if no_shading:
        style["fill"] = 0.0
    elif fill_alpha is not None and style["fill"] > 0:
        style["fill"] = fill_alpha
    return style


def wrap_legend_name(name: str, width: int) -> str:
    if width <= 0:
        return name
    parts = textwrap.wrap(
        name,
        width=width,
        break_long_words=False,
        break_on_hyphens=True,
    )
    return "\n".join(parts) if parts else name


def legend_label(
    role: str | None,
    label: str,
    meta: dict[str, Any],
    legend_label_width: int,
    legend_name: str | None = None,
    legend_text: str | None = None,
    overall: float | None = None,
) -> str:
    if legend_text:
        return legend_text
    name = legend_name or ROLE_DISPLAY.get(role or "", label)
    name = wrap_legend_name(name, legend_label_width)
    percent = normalize_pass_rate(overall if overall is not None else meta.get("pass_rate"))
    if percent is None:
        return name
    return f"{name}\n{percent:.0f}%"


def legend_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    if item.get("legend_order") is not None:
        return (int(item["legend_order"]), item["idx"])
    role = item["role"]
    if role in ROLE_ORDER:
        return (ROLE_ORDER.index(role), item["idx"])
    return (len(ROLE_ORDER), item["idx"])


def parse_axes_rect(raw: Any) -> list[float] | None:
    if not raw:
        return None
    if isinstance(raw, list):
        if len(raw) != 4:
            sys.exit("axes_rect must contain four floats: left,bottom,width,height")
        try:
            rect = [float(p) for p in raw]
        except ValueError:
            sys.exit("axes_rect must contain four floats: left,bottom,width,height")
        left, bottom, width, height = rect
        if min(rect) < 0 or width <= 0 or height <= 0 or left + width > 1 or bottom + height > 1:
            sys.exit("axes_rect values must fit inside the figure: left,bottom,width,height")
        return rect
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        sys.exit("--axes-rect must be four comma-separated floats: left,bottom,width,height")
    try:
        rect = [float(p) for p in parts]
    except ValueError:
        sys.exit("--axes-rect must be four comma-separated floats: left,bottom,width,height")
    left, bottom, width, height = rect
    if min(rect) < 0 or width <= 0 or height <= 0 or left + width > 1 or bottom + height > 1:
        sys.exit("--axes-rect values must fit inside the figure: left,bottom,width,height")
    return rect


def default_axes_rect(fig_width: float, fig_height: float) -> list[float]:
    left = RADAR_AXES_RECT[0]
    bottom = RADAR_AXES_RECT[1]
    width = RADAR_AXES_RECT[2]
    height = width * fig_width / fig_height
    if bottom + height > 0.92:
        height = max(0.1, 0.92 - bottom)
        width = height * fig_height / fig_width
    if left + width > 0.92:
        width = max(0.1, 0.92 - left)
        height = width * fig_width / fig_height
    return [left, bottom, width, height]


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
    run_specs: list[dict[str, Any]],
    out: str,
    title: str | None,
    fill_alpha: float | None,
    no_shading: bool,
    label_font_size: int,
    radial_font_size: int,
    legend_font_size: int,
    no_legend: bool,
    title_font_size: int,
    label_pad: int,
    label_width: int,
    legend_label_width: int,
    fig_width: float,
    fig_height: float,
    dpi: int,
    axes_rect: list[float] | None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            sys.exit("matplotlib is required for radar plots; install it with: python -m pip install matplotlib")
        raise

    n_axes = len(types)
    angles = [2 * math.pi * i / n_axes for i in range(n_axes)]
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    rect = axes_rect or default_axes_rect(fig_width, fig_height)
    ax = fig.add_axes(rect, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    series = []
    for idx, (agg, label, meta, spec) in enumerate(zip(aggs, labels, metas, run_specs)):
        role = spec.get("role") or role_for_label(label)
        style = style_for_series(
            role,
            idx,
            fill_alpha=fill_alpha,
            no_shading=no_shading,
            overrides=spec.get("style"),
        )
        series.append(
            dict(
                idx=idx,
                agg=agg,
                label=label,
                meta=meta,
                role=role,
                style=style,
                legend=legend_label(
                    role,
                    label,
                    meta,
                    legend_label_width=legend_label_width,
                    legend_name=spec.get("legend_name"),
                    legend_text=spec.get("legend_label"),
                    overall=spec.get("overall"),
                ),
                legend_order=spec.get("legend_order"),
            )
        )

    for item in sorted(series, key=lambda x: (x["style"]["z"], x["idx"])):
        style = item["style"]
        vals = values_for(item["agg"], types)
        vals_closed = vals + [vals[0]]
        if style["fill"] > 0:
            ax.fill(
                angles_closed,
                vals_closed,
                color=style["color"],
                alpha=style["fill"],
                zorder=style["z"],
            )
        ax.plot(
            angles_closed,
            vals_closed,
            color=style["color"],
            linewidth=style["lw"],
            linestyle=style["ls"],
            zorder=style["z"] + 0.5,
            solid_capstyle="round",
            dash_capstyle="round",
        )

    ax.set_xticks(angles)
    ax.set_ylim(0, 100)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0, pad=0)
    draw_axis_labels(
        ax,
        angles,
        axis_labels(types, label_width=label_width),
        label_font_size=label_font_size,
        label_pad=label_pad,
    )
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=radial_font_size, color="#888")
    ax.set_rlabel_position(0)
    ax.grid(color="#DDDDDD", linewidth=0.8)
    ax.spines["polar"].set_color("#333333")

    if title:
        ax.set_title(title, y=1.14, fontsize=title_font_size)

    if not no_legend:
        handles = []
        legend_labels = []
        legend_items = sorted(series, key=legend_sort_key)
        for item in legend_items:
            style = item["style"]
            if style["fill"] >= 0.3:
                handle = Patch(
                    facecolor=style["color"],
                    alpha=0.55,
                    edgecolor=style["color"],
                    linewidth=style["lw"],
                )
            else:
                handle = Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=style["lw"] + 0.4,
                    linestyle=style["ls"],
                )
            handles.append(handle)
            legend_labels.append(item["legend"])

        legend = fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=len(handles),
            frameon=True,
            framealpha=0.95,
            fontsize=legend_font_size,
            handlelength=2.4,
            columnspacing=1.6,
            borderpad=0.9,
            handletextpad=0.7,
        )
        legend.get_frame().set_edgecolor("#CCCCCC")
        for text, item in zip(legend.get_texts(), legend_items):
            text.set_color(item["style"]["color"])
            text.set_fontweight("bold")

    fig.savefig(out, facecolor="white", dpi=dpi)
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
    ap.add_argument("results", nargs="*",
                    help="run dirs or scores.json paths; pass two or more for comparison")
    ap.add_argument("--config",
                    help="YAML file with chart settings and per-run style configuration")
    ap.add_argument("--labels", nargs="+",
                    help="legend labels, one per result (default: run slugs)")
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY,
                    help="task->types jsonl (default analysis/descriptive_task_types.jsonl)")
    ap.add_argument("--out", help="output PNG (default <first_run>/success_radar_by_type.png)")
    ap.add_argument("--sort", choices=["fixed", "support", "name", "max-rate", "passes"],
                    default="fixed",
                    help="spoke order (default fixed RefactorBench summary order)")
    ap.add_argument("--min-support", type=int, default=1,
                    help="drop types whose maximum support across runs is below N")
    ap.add_argument("--drop-zero-axes", action="store_true",
                    help="drop type axes where every compared run solved zero tasks")
    ap.add_argument("--keep-zero-axes", action="store_true",
                    help="keep zero-pass type axes; retained for compatibility and now the default")
    ap.add_argument("--fill-alpha", type=float,
                    help="override nonzero polygon fill alpha; use 0 or --no-shading for no fill")
    ap.add_argument("--no-shading", "--no-fill", action="store_true",
                    help="draw only radar outlines, with no filled polygon shading")
    ap.add_argument("--label-font-size", type=int, default=DEFAULT_LABEL_FONT_SIZE,
                    help=f"refactoring-type label font size (default {DEFAULT_LABEL_FONT_SIZE})")
    ap.add_argument("--radial-font-size", type=int, default=DEFAULT_RADIAL_FONT_SIZE,
                    help=f"radial percent label font size (default {DEFAULT_RADIAL_FONT_SIZE})")
    ap.add_argument("--legend-font-size", type=int, default=DEFAULT_LEGEND_FONT_SIZE,
                    help=f"legend font size (default {DEFAULT_LEGEND_FONT_SIZE})")
    ap.add_argument("--no-legend", action="store_true",
                    help="omit the bottom legend")
    ap.add_argument("--title-font-size", type=int, default=DEFAULT_TITLE_FONT_SIZE,
                    help=f"title font size (default {DEFAULT_TITLE_FONT_SIZE})")
    ap.add_argument("--label-pad", type=int, default=DEFAULT_LABEL_PAD,
                    help=f"minimum outward padding for refactoring-type labels (default {DEFAULT_LABEL_PAD})")
    ap.add_argument("--label-width", type=int, default=DEFAULT_LABEL_WIDTH,
                    help=f"wrap refactoring-type labels after this many chars (default {DEFAULT_LABEL_WIDTH})")
    ap.add_argument("--legend-label-width", type=int, default=DEFAULT_LEGEND_LABEL_WIDTH,
                    help=f"wrap legend names after this many chars (default {DEFAULT_LEGEND_LABEL_WIDTH})")
    ap.add_argument("--fig-width", type=float, default=DEFAULT_FIG_WIDTH,
                    help=f"figure width in inches (default {DEFAULT_FIG_WIDTH:g})")
    ap.add_argument("--fig-height", type=float, default=DEFAULT_FIG_HEIGHT,
                    help=f"figure height in inches (default {DEFAULT_FIG_HEIGHT:g})")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                    help=f"figure and saved PNG dpi (default {DEFAULT_DPI})")
    ap.add_argument("--axes-rect",
                    help="optional polar axes box as left,bottom,width,height figure fractions")
    ap.add_argument("--title", help="override chart title")
    args = ap.parse_args()
    config = load_yaml_config(args.config) if args.config else None
    apply_config_defaults(args, config, sys.argv[1:])
    run_specs = load_run_specs(config, args.results, args.labels)

    if not run_specs:
        sys.exit("provide run paths or --config with a non-empty runs list")
    if len(run_specs) < 2:
        print("WARNING: radar chart is most useful with two or more runs", file=sys.stderr)
    if min(
        args.label_font_size,
        args.radial_font_size,
        args.legend_font_size,
        args.title_font_size,
        args.label_width,
        args.legend_label_width,
    ) <= 0:
        sys.exit("font sizes, --label-width, and --legend-label-width must be positive integers")
    if args.label_pad < 0:
        sys.exit("--label-pad must be >= 0")
    if args.fill_alpha is not None and args.fill_alpha < 0:
        sys.exit("--fill-alpha must be >= 0")
    if args.fig_width <= 0 or args.fig_height <= 0:
        sys.exit("--fig-width and --fig-height must be positive")
    if args.dpi <= 0:
        sys.exit("--dpi must be positive")
    axes_rect = parse_axes_rect(args.axes_rect)

    taxonomy = load_taxonomy(args.taxonomy)
    scored_and_meta = [load_scores(str(spec["path"])) for spec in run_specs]
    scored = [x[0] for x in scored_and_meta]
    metas = [x[1] for x in scored_and_meta]
    labels = [str(spec.get("label") or meta["slug"]) for spec, meta in zip(run_specs, metas)]
    aggs = [aggregate(s, taxonomy) for s in scored]
    types = select_types(
        aggs,
        min_support=args.min_support,
        keep_zero_axes=args.keep_zero_axes or not args.drop_zero_axes,
        sort=args.sort,
    )

    print_table(types, labels, aggs)
    candidate_types = CANONICAL_TYPES if args.sort == "fixed" else sorted(set().union(*(set(a) for a in aggs)))
    removed = sorted(
        t for t in candidate_types
        if t not in types and sum(a.get(t, {}).get("passed", 0) for a in aggs) == 0
    )
    if removed and args.drop_zero_axes and not args.keep_zero_axes:
        print(f"\nremoved zero-pass axes ({len(removed)}): {', '.join(removed)}")

    out = args.out or default_out(str(run_specs[0]["path"]))
    plot_radar(
        types,
        aggs,
        metas,
        labels,
        run_specs,
        out,
        title=args.title,
        fill_alpha=args.fill_alpha,
        no_shading=args.no_shading,
        label_font_size=args.label_font_size,
        radial_font_size=args.radial_font_size,
        legend_font_size=args.legend_font_size,
        no_legend=args.no_legend,
        title_font_size=args.title_font_size,
        label_pad=args.label_pad,
        label_width=args.label_width,
        legend_label_width=args.legend_label_width,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        dpi=args.dpi,
        axes_rect=axes_rect,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
