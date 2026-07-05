#!/usr/bin/env bash
# RefactorBench eval-harness setup.
#
# Bootstraps everything needed to run agents on RefactorBench on a fresh machine:
#   - a Python venv with SWE-agent (the streaming patch applied)
#   - Mini-SWE-Agent for the bash-only backend
#   - the rb-swerex Docker image (swe-rex preinstalled, so containers start fast)
#   - the base/lazy SWE-agent batch yamls
#   - scripts/models.env (from the template) + scripts/env.sh helper
#
# Prerequisites: git, Docker (daemon running), network access, and a
# Python >=3.11 interpreter. The AST checkers additionally need a Python <=3.11
# (ast.Str/ast.NameConstant were removed in 3.12); if your SWE-agent venv is
# 3.11 it is reused for both, otherwise a system python3.{11,10,9,8} is used.
#
# Usage:   bash setup.sh
# Re-runnable (idempotent). Override defaults via env vars, e.g.:
#   SWE_AGENT_COMMIT=<sha> IMAGE=rb-swerex:py311-tree-sitter bash setup.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SWE_VENV="${SWE_VENV:-$REPO_ROOT/.venv}"
SWE_SRC="${SWE_SRC:-$REPO_ROOT/.swe-agent-src}"
# Pin SWE-agent to an exact commit so every machine is identical. This commit
# has --agent.model.litellm_model_registry and accepts the streaming patch (the
# v1.1.0 release dropped both). Override with SWE_AGENT_COMMIT=... if needed.
SWE_AGENT_COMMIT="${SWE_AGENT_COMMIT:-a3d018f345241f5a3e1c4c3168289e6a3f81acad}"
IMAGE="${IMAGE:-rb-swerex:py311-tree-sitter}"

log(){ printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git not found"

# --- 1. Python >=3.11 for SWE-agent -----------------------------------------
log "Locating Python >=3.11 for SWE-agent"
SWE_PY=""
for p in python3.11 python3.12 python3.13 python3; do
  command -v "$p" >/dev/null 2>&1 || continue
  v=$("$p" -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])' 2>/dev/null) || continue
  if [ "${v:-0}" -ge 311 ]; then SWE_PY="$p"; break; fi
done
[ -n "$SWE_PY" ] || die "need Python >=3.11 (for SWE-agent); none found"
echo "using $SWE_PY ($($SWE_PY --version 2>&1))"

# --- 2. venv (+ ensure pip) --------------------------------------------------
log "Creating venv at $SWE_VENV"
[ -x "$SWE_VENV/bin/python" ] || "$SWE_PY" -m venv "$SWE_VENV" || die "venv creation failed"
VPY="$SWE_VENV/bin/python"
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || \
  { curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VPY" >/dev/null 2>&1; }
"$VPY" -m pip install -q --upgrade pip wheel setuptools || die "pip bootstrap failed"
"$VPY" -m pip install -q pyyaml || true   # for gen_instances.py / run_model.py

# --- 3. SWE-agent (pinned commit) + streaming patch -------------------------
log "Installing SWE-agent (pinned ${SWE_AGENT_COMMIT:0:12})"
[ -d "$SWE_SRC/.git" ] || git init -q "$SWE_SRC" || die "git init failed"
git -C "$SWE_SRC" remote get-url origin >/dev/null 2>&1 || \
  git -C "$SWE_SRC" remote add origin https://github.com/SWE-agent/SWE-agent.git
# Re-pin if missing or on the wrong commit (also fixes a box stuck on v1.1.0).
if [ "$(git -C "$SWE_SRC" rev-parse HEAD 2>/dev/null)" != "$SWE_AGENT_COMMIT" ]; then
  echo "fetching pinned commit ${SWE_AGENT_COMMIT:0:12} ..."
  git -C "$SWE_SRC" fetch --depth 1 origin "$SWE_AGENT_COMMIT" \
    || die "could not fetch SWE-agent commit (network/proxy/cert?): $SWE_AGENT_COMMIT"
  git -C "$SWE_SRC" checkout -q -f FETCH_HEAD || die "could not checkout pinned commit"
fi
"$VPY" -m pip install -q -e "$SWE_SRC" || die "SWE-agent install failed"

log "Installing Mini-SWE-Agent"
"$VPY" -m pip install -q mini-swe-agent==2.4.4 || die "Mini-SWE-Agent install failed"

log "Applying streaming + reasoning-capture patch"
if grep -q "stream_chunk_builder" "$SWE_SRC/sweagent/agent/models.py" 2>/dev/null; then
  echo "already patched (skipping)"
elif git -C "$SWE_SRC" apply "$REPO_ROOT/scripts/sweagent-streaming.patch" 2>/dev/null; then
  echo "patch applied"
else
  echo "WARNING: streaming patch did not apply cleanly (SWE-agent version drift)."
  echo "         Streaming is optional; apply scripts/sweagent-streaming.patch by hand if wanted."
fi

# --- 4. Checker interpreter (needs <=3.11) ----------------------------------
log "Selecting checker interpreter (RB_TEST_PYTHON)"
CHECKER_PY=""
vv=$("$VPY" -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])')
if [ "$vv" -le 311 ]; then CHECKER_PY="$VPY"; fi
if [ -z "$CHECKER_PY" ]; then
  for p in python3.11 python3.10 python3.9 python3.8; do
    command -v "$p" >/dev/null 2>&1 && { CHECKER_PY="$(command -v "$p")"; break; }
  done
fi
if [ -n "$CHECKER_PY" ]; then
  echo "RB_TEST_PYTHON=$CHECKER_PY"
else
  echo "WARNING: no Python<=3.11 found. The AST checkers use ast.Str/ast.NameConstant"
  echo "         (removed in 3.12). Install python3.11 and set RB_TEST_PYTHON before scoring."
  CHECKER_PY="python3.11"
fi

# --- 5. Docker image with swe-rex preinstalled ------------------------------
log "Building Docker image $IMAGE"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "$IMAGE already present (docker rmi $IMAGE to force a rebuild)"
  else
    # Bake the vendored benchmark repos into the image so tasks need NO GitHub
    # access — each repo is COPYed to /<name> and made a fresh git repo with a
    # base commit. Combined with the 'preexisting' repo config + reset:false in
    # the instance yamls, this removes the per-task clone/fetch (offline mode).
    echo "baking $(ls -d "$REPO_ROOT"/repositories/*_refactor 2>/dev/null | wc -l) repos into $IMAGE (offline mode) ..."
    _dockerfile="$(mktemp)"
    cat > "$_dockerfile" <<'DOCKERFILE'
FROM python:3.11
ENV PYTHONSAFEPATH=1
RUN pip install --no-cache-dir swe-rex tree-sitter==0.21.3 tree-sitter-languages==1.10.2
# Build context is repositories/ : COPY the *_refactor repos to the image root.
COPY . /
RUN set -e; \
    git config --global user.email rb@local; \
    git config --global user.name RefactorBench; \
    git config --global init.defaultBranch main; \
    for d in /*_refactor; do \
      rm -rf "$d/.git"; \
      git -C "$d" init -q; \
      git -C "$d" add -A; \
      git -C "$d" commit -q -m base --no-verify; \
    done
DOCKERFILE
    docker build -t "$IMAGE" -f "$_dockerfile" "$REPO_ROOT/repositories" \
      || echo "WARNING: docker build failed"
    rm -f "$_dockerfile"
  fi
else
  echo "WARNING: Docker not available/running — required to run SWE-agent."
fi

# --- 6. base/lazy instance yamls --------------------------------------------
log "Generating base/lazy instance yamls"
"$VPY" scripts/gen_instances.py || die "gen_instances.py failed"

# 1-instance smoke yaml (gitignored) for a quick end-to-end test
"$VPY" - <<'PY' || true
import yaml
d=yaml.safe_load(open("scripts/descriptive_instances.yaml"))
one=[next(x for x in d if x["problem_statement"]["id"]=="add-log-parameter-get-debug-flag-task")]
yaml.safe_dump(one, open("scripts/smoke_instances.yaml","w"), sort_keys=False, width=10**9)
print("wrote scripts/smoke_instances.yaml (add-log-parameter-get-debug-flag-task)")
PY

# --- 7. model config ---------------------------------------------------------
log "Model config"
if [ ! -f scripts/models.env ]; then
  cp scripts/models.env.example scripts/models.env
  echo "created scripts/models.env -> FILL IN RB_API_BASE / RB_API_KEY / *_MODEL"
else
  echo "scripts/models.env already exists (left untouched)"
fi

# --- 8. convenience env file -------------------------------------------------
cat > scripts/env.sh <<EOF
# source before running:  source scripts/env.sh
export PATH="$SWE_VENV/bin:\$PATH"
export RB_TEST_PYTHON="$CHECKER_PY"
EOF
echo "wrote scripts/env.sh"

# --- done --------------------------------------------------------------------
log "Setup complete"
cat <<EOF
Next steps:
  1) Edit scripts/models.env   (gateway URL, tokens, per-model names)
  2) source scripts/env.sh
  3) Smoke (1 task):
       python scripts/run_model.py --model deepseek --variant descriptive \\
         --instances scripts/smoke_instances.yaml --slug smoke \\
         --image $IMAGE --startup-timeout 1200 --parse thought_action --workers 1
  4) Full run (100 tasks):
       python scripts/run_model.py --model deepseek --variant descriptive \\
         --image $IMAGE --startup-timeout 1200 --parse thought_action --workers 4
  5) Compare:  python scripts/report.py runs/*/scores.json
  6) Mini backend smoke (same scorer, bash-only agent):
       python scripts/run_mini_model.py --model deepseek --variant descriptive \\
         --instances scripts/smoke_instances.yaml --slug mini-smoke \\
         --image $IMAGE --startup-timeout 1200 --command-timeout 30 \\
         --workers 1
  7) Control metrics:  python scripts/compare_agent_control.py runs/*/scores.json

Notes:
  - --startup-timeout high helps on loaded hosts (container start can be slow).
  - Add --reasoning-effort high for thinking models (slower).
  - score.py defaults to --checkout fork (clones dhruvji/* ); use --checkout local
    to score against the bundled repositories/ copies offline.
EOF
