#!/usr/bin/env bash
# RefactorBench eval-harness setup.
#
# Bootstraps everything needed to run agents on RefactorBench on a fresh machine:
#   - a Python venv with SWE-agent (the streaming patch applied)
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
#   SWE_AGENT_REF=main IMAGE=rb-swerex:py311 bash setup.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SWE_VENV="${SWE_VENV:-$REPO_ROOT/.venv}"
SWE_SRC="${SWE_SRC:-$REPO_ROOT/.swe-agent-src}"
SWE_AGENT_REF="${SWE_AGENT_REF:-v1.1.0}"   # version the streaming patch targets
IMAGE="${IMAGE:-rb-swerex:py311}"

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

# --- 3. SWE-agent (pinned) + streaming patch --------------------------------
log "Installing SWE-agent ($SWE_AGENT_REF)"
if [ ! -d "$SWE_SRC/.git" ]; then
  git clone --depth 1 --branch "$SWE_AGENT_REF" \
      https://github.com/SWE-agent/SWE-agent.git "$SWE_SRC" 2>/dev/null || \
  git clone --depth 1 https://github.com/SWE-agent/SWE-agent.git "$SWE_SRC" || \
      die "could not clone SWE-agent"
fi
"$VPY" -m pip install -q -e "$SWE_SRC" || die "SWE-agent install failed"

log "Applying streaming patch"
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
    echo "$IMAGE already present"
  else
    printf 'FROM python:3.11\nRUN pip install --no-cache-dir swe-rex\n' \
      | docker build -t "$IMAGE" - || echo "WARNING: docker build failed"
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
print("wrote scripts/smoke_instances.yaml (1 flask task)")
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

Notes:
  - --startup-timeout high helps on loaded hosts (container start can be slow).
  - Add --reasoning-effort high for thinking models (slower).
  - score.py defaults to --checkout fork (clones dhruvji/* ); use --checkout local
    to score against the bundled repositories/ copies offline.
EOF
