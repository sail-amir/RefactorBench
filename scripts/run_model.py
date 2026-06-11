#!/usr/bin/env python3
"""Run one (model, variant) batch through SWE-agent, then score it.

Wraps:
    sweagent run-batch --instances.type expert_file \
        --instances.path scripts/<variant>_instances.yaml \
        --agent.model.name <name> [--agent.model.api_base <base>] \
        --agent.model.per_instance_cost_limit <c> [--agent.model.max_input_tokens <k>] \
        [--agent.tools.parse_function.type thought_action] \
        --num_workers <n> --outputs.dir runs/<slug>__<variant>
then:
    python3 scripts/score.py --preds runs/<slug>__<variant>/preds.json \
        --out runs/<slug>__<variant>/scores.json --variant <variant>

Presets (``--model claude|deepseek|glm|pangu``) fill in name/parser/cost/env.
Override anything explicitly. Use ``--dry-run`` to print the command only, and
``--instances`` to point at a custom yaml (e.g. a smoke subset).

API keys come from the environment (or ``--api-key``):
  claude -> ANTHROPIC_API_KEY   deepseek -> DEEPSEEK_API_KEY
  glm    -> OPENAI_API_KEY (Zhipu key)   pangu -> see notes below.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_ENV_FILE = os.path.join(SCRIPTS_DIR, "models.env")

# All models are served by ONE OpenAI-compatible gateway: a shared base URL
# (RB_API_BASE) + shared key (RB_API_KEY), differing only by model name. Each
# preset reads its model name from <PREFIX>_MODEL; per-model base/key overrides
# (<PREFIX>_API_BASE / <PREFIX>_API_KEY) are optional. Set these in
# scripts/models.env. The "openai/" prefix routes litellm to RB_API_BASE.
PRESETS = {
    "claude": {
        "slug": "claude-opus-4-8", "env_prefix": "CLAUDE",
        "model_name": "openai/claude-opus-4-8", "parse": "function_calling",
        "cost": 0.0, "max_input_tokens": 32000,
    },
    "deepseek": {
        "slug": "deepseek-chat", "env_prefix": "DEEPSEEK",
        "model_name": "openai/deepseek-chat", "parse": "function_calling",
        "cost": 0.0, "max_input_tokens": 32000,
    },
    "glm": {
        "slug": "glm-5.1", "env_prefix": "GLM",
        "model_name": "openai/glm-5.1", "parse": "thought_action",
        "cost": 0.0, "max_input_tokens": 32000,
    },
    "pangu": {
        "slug": "pangu", "env_prefix": "PANGU",
        "model_name": "openai/pangu_auto", "parse": "thought_action",
        "cost": 0.0, "max_input_tokens": 32000,
    },
}


def load_env_file(path: str) -> None:
    """Load KEY=VALUE lines from an env file into os.environ (no override of
    already-set vars). Lines may start with 'export '. Blank/comment lines ignored.
    """
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
            k, v = k.strip(), v.strip().strip("'\"")
            os.environ.setdefault(k, v)


def override_deployment(src_yaml: str, dst_yaml: str, image=None,
                        startup_timeout=None) -> str:
    """Write a copy of src_yaml with env.deployment.image / .startup_timeout set."""
    import yaml  # available in the SWE-agent (conda) env
    with open(src_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for inst in data:
        dep = inst.setdefault("env", {}).setdefault("deployment", {})
        if image is not None:
            dep["image"] = image
        if startup_timeout is not None:
            dep["startup_timeout"] = float(startup_timeout)
    with open(dst_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, width=10**9, allow_unicode=True)
    return dst_yaml


def build_cmd(a) -> list[str]:
    instances = a.instances or os.path.join(
        SCRIPTS_DIR, f"{a.variant}_instances.yaml"
    )
    if not os.path.exists(instances):
        sys.exit(f"instances file not found: {instances}\n"
                 f"(generate with: python scripts/gen_instances.py)")
    if a.image or a.startup_timeout:
        os.makedirs(a.outdir, exist_ok=True)
        instances = override_deployment(
            instances, os.path.join(a.outdir, "instances.yaml"),
            image=a.image, startup_timeout=a.startup_timeout,
        )

    cmd = [
        a.sweagent_bin, "run-batch",
        "--instances.type", "expert_file",
        "--instances.path", instances,
        "--agent.model.name", a.model_name,
        "--agent.model.per_instance_cost_limit", str(a.cost),
        "--num_workers", str(a.workers),
        "--output_dir", a.outdir,
    ]
    if a.api_base:
        cmd += ["--agent.model.api_base", a.api_base]
    if a.api_key:
        cmd += ["--agent.model.api_key", a.api_key]
    if a.registry and os.path.exists(a.registry):
        cmd += ["--agent.model.litellm_model_registry", a.registry]
    if a.max_input_tokens:
        cmd += ["--agent.model.max_input_tokens", str(a.max_input_tokens)]
    if a.parse and a.parse != "function_calling":
        cmd += ["--agent.tools.parse_function.type", a.parse]
    if a.per_instance_call_limit:
        cmd += ["--agent.model.per_instance_call_limit", str(a.per_instance_call_limit)]
    # Sampling: standard params as native flags, the rest via extra_body.
    if a.temperature is not None:
        cmd += ["--agent.model.temperature", str(a.temperature)]
    if a.top_p is not None:
        cmd += ["--agent.model.top_p", str(a.top_p)]
    if a.max_output_tokens:
        cmd += ["--agent.model.max_output_tokens", str(a.max_output_tokens)]
    if a.stream:
        cmd += ["--agent.model.stream", "true"]
    if a.extra_body:
        cmd += ["--agent.model.completion_kwargs",
                json.dumps({"extra_body": a.extra_body})]
    cmd += a.extra
    return cmd


def resolve_sampling(a) -> None:
    """Populate a.temperature/top_p/max_output_tokens/extra_body from the
    per-model sampling profile (scripts/sampling.yaml), with CLI overrides."""
    import yaml
    path = a.sampling_file or os.path.join(SCRIPTS_DIR, "sampling.yaml")
    allp = {}
    if os.path.exists(path):
        allp = yaml.safe_load(open(path)) or {}
    base = dict(allp.get("default") or {})
    mp = dict(allp.get(a.model) or {})
    extra = {**(base.get("extra_body") or {}), **(mp.get("extra_body") or {})}
    prof = {**base, **mp}
    for k in ("top_k", "min_p", "repetition_penalty",
              "frequency_penalty", "presence_penalty"):
        if k in prof:
            extra[k] = prof[k]
    # CLI overrides win over the profile.
    a.temperature = a.temperature if a.temperature is not None else prof.get("temperature")
    a.top_p = a.top_p if a.top_p is not None else prof.get("top_p")
    a.max_output_tokens = prof.get("max_output_tokens")
    a.stream = a.stream or bool(prof.get("stream"))
    if a.top_k is not None:
        extra["top_k"] = a.top_k
    if a.reasoning_effort:
        extra["reasoning_effort"] = a.reasoning_effort
    a.extra_body = extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), help="preset")
    ap.add_argument("--variant", default="descriptive",
                    choices=["base", "descriptive", "lazy"])
    ap.add_argument("--slug", help="output dir slug (default: preset slug)")
    ap.add_argument("--model-name", help="override litellm model name")
    ap.add_argument("--parse", help="function_calling | thought_action")
    ap.add_argument("--cost", type=float, help="per_instance_cost_limit (0 disables)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--api-base")
    ap.add_argument("--api-key", help="explicit key (else read key_env from env)")
    ap.add_argument("--max-input-tokens", type=int, default=0)
    ap.add_argument("--per-instance-call-limit", type=int, default=0)
    ap.add_argument("--instances", help="custom instances yaml (e.g. smoke subset)")
    ap.add_argument("--image", help="override env.deployment.image for all instances "
                    "(e.g. rb-swerex:py311 with swe-rex preinstalled)")
    ap.add_argument("--startup-timeout", type=float,
                    help="override env.deployment.startup_timeout seconds (default 180; "
                    "raise it when the host is heavily loaded)")
    ap.add_argument("--reasoning-effort",
                    help="enable model thinking via extra_body.reasoning_effort "
                    "(e.g. high) — for deepseek-v3.2 and similar reasoning models")
    ap.add_argument("--temperature", type=float, help="override profile temperature")
    ap.add_argument("--top-p", type=float, help="override profile top_p")
    ap.add_argument("--top-k", type=int, help="override profile top_k (via extra_body)")
    ap.add_argument("--sampling-file",
                    help="per-model sampling profile yaml (default scripts/sampling.yaml)")
    ap.add_argument("--stream", action="store_true", default=False,
                    help="stream completions (also settable per-model via sampling.yaml "
                    "'stream: true'); for gateways that require/prefer streaming")
    ap.add_argument("--score-checkout", default="fork", choices=["fork", "local"])
    ap.add_argument("--sweagent-bin", default="sweagent")
    ap.add_argument("--registry", default=os.path.join(SCRIPTS_DIR, "litellm_registry.json"),
                    help="litellm model-registry json (capabilities/context); set '' to disable")
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE,
                    help="KEY=VALUE file with endpoints/tokens (default scripts/models.env)")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print command, don't run")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args appended verbatim to sweagent")
    a = ap.parse_args()

    load_env_file(a.env_file)
    preset = PRESETS.get(a.model, {})
    pfx = preset.get("env_prefix")

    def env(suffix):  # per-model var, e.g. CLAUDE_MODEL
        return os.environ.get(f"{pfx}_{suffix}") if pfx else None

    # Precedence: explicit flag > per-model env > shared gateway env > preset default.
    a.model_name = a.model_name or env("MODEL") or preset.get("model_name")
    if not a.model_name:
        sys.exit("need --model <preset> or --model-name")
    a.parse = a.parse or os.environ.get("RB_PARSE") or preset.get("parse", "function_calling")
    a.cost = a.cost if a.cost is not None else preset.get("cost", 0.0)
    a.api_base = a.api_base or env("API_BASE") or os.environ.get("RB_API_BASE")
    a.api_key = a.api_key or env("API_KEY") or os.environ.get("RB_API_KEY")
    if a.max_input_tokens == 0 and preset.get("max_input_tokens"):
        a.max_input_tokens = preset["max_input_tokens"]
    resolve_sampling(a)
    slug = a.slug or preset.get("slug") or a.model_name.split("/")[-1]
    a.outdir = os.path.join(REPO_ROOT, "runs", f"{slug}__{a.variant}")

    # Auth sanity for the shared gateway.
    if not a.api_base:
        print(f"WARNING: no api_base (set RB_API_BASE in {a.env_file}).", file=sys.stderr)
    if not a.api_key:
        print(f"WARNING: no api_key (set RB_API_KEY in {a.env_file}).", file=sys.stderr)

    cmd = build_cmd(a)
    printable = " ".join(
        ("'%s'" % c if (" " in c or "" == c) else c) for c in cmd
    )
    print(f"# slug={slug} variant={a.variant} parse={a.parse} cost={a.cost}")
    print(f"# sampling: temperature={a.temperature} top_p={a.top_p} "
          f"extra_body={a.extra_body}")
    print(printable)
    if a.dry_run:
        return 0

    os.makedirs(a.outdir, exist_ok=True)
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        print(f"sweagent run-batch exited {rc}", file=sys.stderr)
        return rc

    preds = os.path.join(a.outdir, "preds.json")
    if a.no_score:
        print(f"skipping scoring (--no-score). preds at {preds}")
        return 0
    if not os.path.exists(preds):
        print(f"no preds.json at {preds}; cannot score", file=sys.stderr)
        return 1

    # Scope scoring to exactly the instances that were run (so a smoke subset
    # doesn't clone every repo / emit phantom no_prediction rows).
    import yaml
    src_inst = a.instances or os.path.join(SCRIPTS_DIR, f"{a.variant}_instances.yaml")
    run_ids = [i["problem_statement"]["id"] for i in yaml.safe_load(open(src_inst))]

    score_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, "score.py"),
        "--preds", preds,
        "--out", os.path.join(a.outdir, "scores.json"),
        "--variant", a.variant,
        "--checkout", a.score_checkout,
        "--model", a.model_name,
        "--only", ",".join(run_ids),
    ]
    return subprocess.run(score_cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
