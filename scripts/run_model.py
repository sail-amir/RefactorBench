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
        "cost": 0.0, "max_input_tokens": 110000,
    },
    "deepseek": {
        "slug": "deepseek-chat", "env_prefix": "DEEPSEEK",
        "model_name": "openai/deepseek-chat", "parse": "function_calling",
        "cost": 0.0, "max_input_tokens": 110000,
    },
    "glm": {
        "slug": "glm-5.1", "env_prefix": "GLM",
        "model_name": "openai/glm-5.1", "parse": "thought_action",
        "cost": 0.0, "max_input_tokens": 110000,
    },
    "pangu": {
        "slug": "pangu", "env_prefix": "PANGU",
        # pangu35b verified to return native tool_calls via the gateway, and the
        # registry marks it function-calling capable -> use the native loop.
        "model_name": "openai/pangu_auto", "parse": "function_calling",
        "cost": 0.0, "max_input_tokens": 110000,
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
            os.environ.setdefault(k.strip(), _envval(v))


def _envval(v: str) -> str:
    """Parse an env-file value: honor quotes, else strip an inline ` # comment`."""
    v = v.strip()
    if v[:1] in ("'", '"'):
        q = v[0]
        end = v.find(q, 1)
        return v[1:end] if end != -1 else v[1:]
    for i, ch in enumerate(v):  # comment must follow whitespace (so URLs with # survive)
        if ch == "#" and i > 0 and v[i - 1] in " \t":
            v = v[:i]
            break
    return v.strip()


def override_deployment(src_yaml: str, dst_yaml: str, image=None,
                        startup_timeout=None, docker_args=None) -> str:
    """Write a copy of src_yaml with env.deployment.image/.startup_timeout/.docker_args set."""
    import yaml  # available in the SWE-agent (conda) env
    with open(src_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for inst in data:
        dep = inst.setdefault("env", {}).setdefault("deployment", {})
        if image is not None:
            dep["image"] = image
        if startup_timeout is not None:
            dep["startup_timeout"] = float(startup_timeout)
        if docker_args:
            dep["docker_args"] = (dep.get("docker_args") or []) + list(docker_args)
    with open(dst_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, width=10**9, allow_unicode=True)
    return dst_yaml


def ensure_registry(a) -> str:
    """Return a litellm model-registry json that includes a.model_name.

    SWE-agent looks up litellm's registry by the EXACT model name to decide
    `supports_function_calling`. Gateway model names are deployment-specific
    (e.g. openai/pangu35b vs the committed openai/pangu_auto), so we start from
    the committed registry (--registry) and ensure an entry for the model being
    run. Use --parse thought_action to override if a model isn't tool-capable.
    """
    reg = {}
    if a.registry and os.path.exists(a.registry):
        try:
            with open(a.registry, "r", encoding="utf-8") as fh:
                reg = json.load(fh)
        except (json.JSONDecodeError, OSError):
            reg = {}
    if a.model_name not in reg:
        reg[a.model_name] = {
            "max_input_tokens": a.max_input_tokens or 131072,
            "max_output_tokens": a.max_output_tokens or 20000,
            "input_cost_per_token": 0, "output_cost_per_token": 0,
            "litellm_provider": "openai", "mode": "chat",
            "supports_function_calling": True, "supports_tool_choice": True,
        }
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "registry.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2)
    return path


def unsupported_model_flags(sweagent_bin, flags):
    """Return the subset of `flags` [(flag, value), ...] this SWE-agent build
    rejects. Probes by running a minimal run-batch that fails AFTER arg parsing
    (bogus instances path); flags listed in 'unrecognized arguments' are absent.
    SWE-agent builds drift (e.g. some lack litellm_model_registry; streaming
    needs the patch), so we skip flags this build doesn't understand."""
    if not flags:
        return set()
    cmd = [sweagent_bin, "run-batch", "--instances.type", "expert_file",
           "--instances.path", "/nonexistent_rb_flag_probe.yaml",
           "--agent.model.name", "probe", "--output_dir", "/tmp/rb_flag_probe"]
    for f, v in flags:
        cmd += [f, v]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return set()  # can't probe -> assume supported, let the real run surface it
    text = (r.stdout or "") + (r.stderr or "")
    if "unrecognized arguments" not in text:
        return set()
    return {f for f, _ in flags if f in text}


def build_cmd(a) -> list[str]:
    instances = a.instances or os.path.join(
        SCRIPTS_DIR, f"{a.variant}_instances.yaml"
    )
    if not os.path.exists(instances):
        sys.exit(f"instances file not found: {instances}\n"
                 f"(generate with: python scripts/gen_instances.py)")
    docker_args = list(a.docker_arg or [])
    if a.insecure_git:
        # Container's git skips TLS verification (restricted/MITM networks where
        # github's cert chain isn't trusted inside the container).
        docker_args += ["-e", "GIT_SSL_NO_VERIFY=1"]
    if a.image or a.startup_timeout or docker_args:
        os.makedirs(a.outdir, exist_ok=True)
        instances = override_deployment(
            instances, os.path.join(a.outdir, "instances.yaml"),
            image=a.image, startup_timeout=a.startup_timeout,
            docker_args=docker_args or None,
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
    # Skip optional flags this SWE-agent build doesn't understand (version drift).
    probe = []
    if a.registry != "":
        probe.append(("--agent.model.litellm_model_registry", "/tmp/_rbprobe"))
    if a.stream:
        probe.append(("--agent.model.stream", "true"))
    bad = unsupported_model_flags(a.sweagent_bin, probe)
    for f in sorted(bad):
        print(f"# note: this SWE-agent build lacks {f} — skipping it", file=sys.stderr)

    if a.api_base:
        cmd += ["--agent.model.api_base", a.api_base]
    if a.api_key:
        cmd += ["--agent.model.api_key", a.api_key]
    if a.registry != "" and "--agent.model.litellm_model_registry" not in bad:
        cmd += ["--agent.model.litellm_model_registry", ensure_registry(a)]
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
    if a.stream and "--agent.model.stream" not in bad:
        cmd += ["--agent.model.stream", "true"]
    # Request-body kwargs (spread into litellm.completion). The per-request output
    # cap MUST be max_tokens here: SWE-agent only maps max_output_tokens->max_tokens
    # for anthropic providers, so for an openai-compatible gateway it'd be ignored.
    ck = {}
    if a.extra_body:
        ck["extra_body"] = a.extra_body
    if a.max_output_tokens:
        ck["max_tokens"] = a.max_output_tokens
    if ck:
        cmd += ["--agent.model.completion_kwargs", json.dumps(ck)]
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
    # Per-request output cap: CLI > sampling profile > 20000 default.
    a.max_output_tokens = a.max_output_tokens or prof.get("max_output_tokens") or 20000
    a.stream = a.stream or bool(prof.get("stream"))
    if a.top_k is not None:
        extra["top_k"] = a.top_k
    if a.reasoning_effort:
        extra["reasoning_effort"] = a.reasoning_effort
    a.extra_body = extra


def _auto_parse(model_name, api_base, api_key) -> str:
    """Resolve ``--parse auto`` by probing the gateway once for native tool_calls.

    Enable ``function_calling`` ONLY when the gateway positively returns a
    tool_call; any failure (probe unavailable, no creds, HTTP/transport error, or
    a response without tool_calls) falls back to ``thought_action``, which works
    regardless of FC support. NOTE: a single probe confirms FC is *possible*, not
    *reliable* — a gateway that returns tool_calls only intermittently still
    passes here (use ``--parse thought_action`` explicitly if FC is flaky).
    """
    try:
        from check_gateway import probe_tools
    except Exception as e:  # pragma: no cover - import guard
        print(f"# auto-parse: probe unavailable ({e}); falling back to thought_action")
        return "thought_action"
    if not api_base or not api_key:
        print("# auto-parse: no api_base/api_key to probe; falling back to thought_action")
        return "thought_action"
    # The gateway expects the bare model name; strip the litellm provider prefix.
    model = model_name.split("/", 1)[1] if "/" in model_name else model_name
    url = api_base.rstrip("/") + "/chat/completions"
    ok, detail = probe_tools(url, api_key, model, timeout=30)
    decision = "function_calling" if ok else "thought_action"
    print(f"# auto-parse: native tool_calls {'OK' if ok else 'absent'} "
          f"-> parse={decision} ({detail})")
    return decision


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), help="preset")
    ap.add_argument("--variant", default="descriptive",
                    choices=["base", "descriptive", "lazy"])
    ap.add_argument("--slug", help="output dir slug (default: preset slug)")
    ap.add_argument("--model-name", help="override litellm model name")
    ap.add_argument("--parse", help="function_calling | thought_action | auto "
                    "(auto: probe the gateway once for native tool_calls; enable "
                    "function_calling only if confirmed, else thought_action)")
    ap.add_argument("--cost", type=float, help="per_instance_cost_limit (0 disables)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--api-base")
    ap.add_argument("--api-key", help="explicit key (else read key_env from env)")
    ap.add_argument("--max-input-tokens", type=int, default=0,
                    help="context budget: SWE-agent guardrail on the accumulating "
                    "conversation (default 110000 via preset). Not an API param.")
    ap.add_argument("--max-output-tokens", type=int, default=0,
                    help="per-request output cap (default 20000). Sent to the gateway "
                    "as max_tokens via completion_kwargs (works for openai-compatible).")
    ap.add_argument("--per-instance-call-limit", type=int, default=0)
    ap.add_argument("--instances", help="custom instances yaml (e.g. smoke subset)")
    ap.add_argument("--image", help="override env.deployment.image for all instances "
                    "(e.g. rb-swerex:py311 with swe-rex preinstalled)")
    ap.add_argument("--startup-timeout", type=float,
                    help="override env.deployment.startup_timeout seconds (default 180; "
                    "raise it when the host is heavily loaded)")
    ap.add_argument("--insecure-git", action="store_true",
                    help="set GIT_SSL_NO_VERIFY=1 in the container so the repo clone "
                    "works behind a TLS-inspecting proxy / untrusted-CA network")
    ap.add_argument("--docker-arg", action="append", default=[],
                    help="extra arg passed to `docker run` (repeatable), e.g. --docker-arg -e --docker-arg K=V")
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
    if a.parse == "auto":
        a.parse = _auto_parse(a.model_name, a.api_base, a.api_key)
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
