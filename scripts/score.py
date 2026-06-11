#!/usr/bin/env python3
"""Score a SWE-agent batch run against RefactorBench's AST checkers.

SWE-agent emits ``<outputs.dir>/preds.json`` keyed by ``instance_id`` (== the
yaml ``problem_statement.id`` == the problem-file stem), each value carrying a
``model_patch`` unified-diff string. RefactorBench ships no scorer, so we build
one: for each instance, apply the patch to a clean checkout of the matching
``dhruvji/<repo>`` fork (the tree SWE-agent ran against), run the mapped
``tests/<repo>/<file>.py`` AST checker from a directory one level under the repo
root, and record pass/fail.

The checkers are stdlib-only (no ``pip install`` of django/flask/etc.) but some
use ``ast.Str`` / ``ast.NameConstant``, which were REMOVED in Python 3.12 — so
they must run under Python <= 3.11. Default interpreter: ``python3.10``
(override with ``RB_TEST_PYTHON``).

Usage:
    python3 scripts/score.py --preds runs/<slug>/preds.json \
        --out runs/<slug>/scores.json --variant descriptive --checkout fork
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
CACHE_DIR = os.path.join(REPO_ROOT, ".rb_cache")
FORK_URL = "https://github.com/dhruvji/{repo}"  # repo already includes _refactor
PYTHON_FOR_TESTS = os.environ.get("RB_TEST_PYTHON", "python3.10")
RUN_DIR_NAME = "_rb_run"

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "rb",
    "GIT_AUTHOR_EMAIL": "rb@local",
    "GIT_COMMITTER_NAME": "rb",
    "GIT_COMMITTER_EMAIL": "rb@local",
    "GIT_TERMINAL_PROMPT": "0",
}


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def load_mapping(variant: str) -> dict:
    """instance_id -> {"repo": <repo>, "test": <abs test path>}.

    Parsed from ``scripts/<variant>_mapping.py`` without importing it: the file
    is a single ``file_mapping = { '<test>': '<problem>' }`` literal.
    """
    path = os.path.join(SCRIPTS_DIR, f"{variant}_mapping.py")
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    literal = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "file_mapping" for t in node.targets
        ):
            literal = ast.literal_eval(node.value)
            break
    if literal is None:
        raise ValueError(f"no file_mapping found in {path}")

    out = {}
    for test_rel, problem_rel in literal.items():
        # test_rel: '../tests/<repo>/<file>.py'
        rel = test_rel[3:] if test_rel.startswith("../") else test_rel.lstrip("./")
        parts = rel.split("/")  # ['tests', '<repo>', '<file>.py']
        repo = parts[1]
        abs_test = os.path.join(REPO_ROOT, rel)
        iid = os.path.splitext(os.path.basename(problem_rel))[0]  # problem stem
        if not os.path.exists(abs_test):
            raise FileNotFoundError(f"mapped test missing: {abs_test}")
        out[iid] = {"repo": repo, "test": abs_test}
    return out


# --------------------------------------------------------------------------- #
# Checkout management
# --------------------------------------------------------------------------- #
def _git(args, cwd, check=True, capture=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, env=GIT_ENV, check=check,
        capture_output=capture, text=True,
    )


def prepare_checkout(repo: str, strategy: str) -> tuple[str, str]:
    """Return (checkout_path, head_sha), creating/caching it once per repo."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{strategy}__{repo}")
    if not os.path.isdir(os.path.join(path, ".git")):
        if os.path.exists(path):
            shutil.rmtree(path)
        if strategy == "fork":
            url = FORK_URL.format(repo=repo)
            print(f"  cloning {url} ...", flush=True)
            _git(["clone", "--depth", "1", url, path], cwd=REPO_ROOT)
        elif strategy == "local":
            src = os.path.join(REPO_ROOT, "repositories", repo)
            if not os.path.isdir(src):
                raise FileNotFoundError(f"local repo not found: {src}")
            print(f"  seeding local checkout from {src} ...", flush=True)
            shutil.copytree(src, path, symlinks=True)
            # Drop any nested VCS metadata, then make it a real git repo so the
            # patch-apply path is identical to the fork strategy.
            shutil.rmtree(os.path.join(path, ".git"), ignore_errors=True)
            _git(["init", "-q"], cwd=path)
            _git(["add", "-A"], cwd=path)
            _git(["commit", "-q", "-m", "base", "--no-verify"], cwd=path)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
    head = _git(["rev-parse", "HEAD"], cwd=path).stdout.strip()
    return path, head


def reset_checkout(path: str) -> None:
    _git(["reset", "--hard", "-q", "HEAD"], cwd=path)
    _git(["clean", "-fdq"], cwd=path)


# --------------------------------------------------------------------------- #
# Patch application
# --------------------------------------------------------------------------- #
def apply_patch(path: str, patch_str: str) -> tuple[bool, str]:
    """Try a ladder of apply strategies. Return (ok, method_or_error)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".patch", delete=False, dir=path, encoding="utf-8"
    ) as fh:
        if not patch_str.endswith("\n"):
            patch_str += "\n"
        fh.write(patch_str)
        patch_file = fh.name
    try:
        attempts = [
            (["apply", "-p1", "--whitespace=nowarn", patch_file], "git-apply"),
            (["apply", "-p1", "--3way", "--whitespace=nowarn", patch_file], "git-apply-3way"),
        ]
        last_err = ""
        for git_args, label in attempts:
            r = _git(git_args, cwd=path, check=False)
            if r.returncode == 0:
                return True, label
            last_err = (r.stderr or r.stdout or "").strip()
        # Final fallback: GNU patch with fuzz.
        r = subprocess.run(
            ["patch", "-p1", "--fuzz=3", "--no-backup-if-mismatch", "-i", patch_file],
            cwd=path, capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True, "patch-fuzz"
        last_err = (r.stderr or r.stdout or last_err).strip()
        return False, last_err[-800:]
    finally:
        try:
            os.unlink(patch_file)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Test execution
# --------------------------------------------------------------------------- #
def run_test(path: str, abs_test: str, timeout: int) -> tuple[bool, str]:
    rundir = os.path.join(path, RUN_DIR_NAME)
    # The checker opens source via '../X' paths, so it MUST run from a directory
    # exactly one level under the repo root.
    assert os.path.dirname(rundir) == os.path.abspath(path), "run dir not depth-1"
    os.makedirs(rundir, exist_ok=True)
    dst = os.path.join(rundir, "rb_test.py")
    shutil.copyfile(abs_test, dst)
    try:
        r = subprocess.run(
            [PYTHON_FOR_TESTS, "rb_test.py"],
            cwd=rundir, capture_output=True, text=True,
            timeout=timeout, start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        sys.exit(f"interpreter {PYTHON_FOR_TESTS!r} not found (set RB_TEST_PYTHON)")
    if r.returncode == 0:
        return True, "ok"
    tail = ((r.stderr or "") + (r.stdout or "")).strip()[-1200:]
    return False, f"test_failed: {tail}"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_preds(preds_path: str) -> dict:
    with open(preds_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # SWE-agent preds.json is a dict keyed by instance_id.
    if isinstance(data, list):
        data = {d["instance_id"]: d for d in data}
    return data


def score(args) -> dict:
    mapping = load_mapping(args.variant)
    preds = load_preds(args.preds)

    only = set(args.only.split(",")) if args.only else None
    ids = [i for i in mapping if (only is None or i in only)]
    # Group by repo so each checkout is prepared/cloned once.
    ids.sort(key=lambda i: (mapping[i]["repo"], i))

    model = args.model or next(
        (p.get("model_name_or_path") for p in preds.values() if isinstance(p, dict)),
        "unknown",
    )

    instances, fork_shas, per_repo = [], {}, {}
    current_repo = None
    ckpt = None

    for iid in ids:
        repo = mapping[iid]["repo"]
        if repo != current_repo:
            ckpt, sha = prepare_checkout(repo, args.checkout)
            fork_shas[repo] = sha
            current_repo = repo
        per_repo.setdefault(repo, {"n": 0, "passed": 0})
        per_repo[repo]["n"] += 1

        rec = {"id": iid, "repo": repo, "passed": False,
               "patch_applied": False, "reason": "", "duration": 0.0}
        t0 = time.time()

        pred = preds.get(iid)
        patch = (pred or {}).get("model_patch") if isinstance(pred, dict) else None
        if pred is None:
            rec["reason"] = "no_prediction"
        elif not patch or not patch.strip():
            rec["reason"] = "empty_patch"
        else:
            reset_checkout(ckpt)
            ok, info = apply_patch(ckpt, patch)
            rec["patch_applied"] = ok
            if not ok:
                rec["reason"] = f"patch_apply_failed: {info}"
            else:
                passed, info = run_test(ckpt, mapping[iid]["test"], args.test_timeout)
                rec["passed"] = passed
                rec["reason"] = info
                if passed:
                    per_repo[repo]["passed"] += 1

        rec["duration"] = round(time.time() - t0, 2)
        instances.append(rec)
        flag = "PASS" if rec["passed"] else "fail"
        print(f"  [{flag}] {repo}/{iid}  ({rec['reason'][:80]})", flush=True)

    total_n = len(instances)
    total_p = sum(r["passed"] for r in instances)
    for repo, d in per_repo.items():
        d["pass_rate"] = round(d["passed"] / d["n"], 4) if d["n"] else 0.0

    return {
        "model": model,
        "variant": args.variant,
        "checkout": args.checkout,
        "fork_shas": fork_shas,
        "total": {
            "n": total_n,
            "passed": total_p,
            "pass_rate": round(total_p / total_n, 4) if total_n else 0.0,
        },
        "per_repo": dict(sorted(per_repo.items())),
        "instances": instances,
    }


def check_interpreter() -> None:
    try:
        r = subprocess.run(
            [PYTHON_FOR_TESTS, "-c",
             "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit(
            f"Checker interpreter {PYTHON_FOR_TESTS!r} not runnable. "
            "RefactorBench checkers use ast.Str/ast.NameConstant (removed in "
            "Python 3.12); set RB_TEST_PYTHON to a 3.8-3.11 interpreter."
        )
    major, minor = (int(x) for x in r.stdout.strip().split("."))
    if (major, minor) >= (3, 12):
        print(
            f"WARNING: RB_TEST_PYTHON={PYTHON_FOR_TESTS} is {major}.{minor} >= 3.12; "
            "checkers using ast.Str/ast.NameConstant will error. Use python<=3.11.",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="path to SWE-agent preds.json")
    ap.add_argument("--out", required=True, help="path to write scores.json")
    ap.add_argument("--variant", default="descriptive",
                    choices=["base", "descriptive", "lazy"])
    ap.add_argument("--checkout", default="fork", choices=["fork", "local"])
    ap.add_argument("--test-timeout", type=int, default=120)
    ap.add_argument("--only", default="", help="comma-separated subset of ids")
    ap.add_argument("--model", default="", help="override model label in output")
    args = ap.parse_args()

    check_interpreter()
    result = score(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    t = result["total"]
    print(f"\n== {result['model']} / {result['variant']} ==")
    print(f"TOTAL: {t['passed']}/{t['n']} = {t['pass_rate']:.1%}")
    for repo, d in result["per_repo"].items():
        print(f"  {repo:22} {d['passed']:>3}/{d['n']:<3} {d['pass_rate']:.1%}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
