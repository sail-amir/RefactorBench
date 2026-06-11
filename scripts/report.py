#!/usr/bin/env python3
"""Tabulate a model x variant comparison grid from many scores.json files.

Each ``scores.json`` (produced by score.py) carries ``model``, ``variant``,
``total`` and ``per_repo``. This builds a grid with rows = repos + TOTAL and
columns = "<model>/<variant>", each cell = "passed/n (pct)".

Usage:
    python3 scripts/report.py runs/*/scores.json [--format md|tsv] [--metric rate|frac]
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def col_label(score: dict) -> str:
    model = str(score.get("model", "?")).split("/")[-1]  # drop provider prefix
    return f"{model}/{score.get('variant', '?')}"


def cell(passed: int, n: int, metric: str) -> str:
    if n == 0:
        return "-"
    if metric == "frac":
        return f"{passed}/{n}"
    return f"{passed/n:.0%} ({passed}/{n})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scores", nargs="+", help="paths to scores.json files")
    ap.add_argument("--format", default="md", choices=["md", "tsv"])
    ap.add_argument("--metric", default="rate", choices=["rate", "frac"])
    args = ap.parse_args()

    cols = []          # (label, score_dict)
    repos = set()
    for p in args.scores:
        if not os.path.exists(p):
            print(f"skip (missing): {p}", file=sys.stderr)
            continue
        with open(p, "r", encoding="utf-8") as fh:
            s = json.load(fh)
        cols.append((col_label(s), s))
        repos.update(s.get("per_repo", {}).keys())

    if not cols:
        sys.exit("no scores.json files loaded")

    cols.sort(key=lambda c: c[0])
    rows = sorted(repos) + ["TOTAL"]
    header = ["repo"] + [c[0] for c in cols]

    def row_cells(repo: str) -> list[str]:
        out = [repo]
        for _, s in cols:
            if repo == "TOTAL":
                t = s.get("total", {})
                out.append(cell(t.get("passed", 0), t.get("n", 0), args.metric))
            else:
                d = s.get("per_repo", {}).get(repo)
                out.append(cell(d["passed"], d["n"], args.metric) if d else "-")
        return out

    if args.format == "tsv":
        print("\t".join(header))
        for r in rows:
            print("\t".join(row_cells(r)))
    else:
        widths = [len(h) for h in header]
        table = [row_cells(r) for r in rows]
        for r in table:
            for i, val in enumerate(r):
                widths[i] = max(widths[i], len(val))
        def fmt(cells):
            return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
        print(fmt(header))
        print("| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |")
        for r in table:
            print(fmt(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
