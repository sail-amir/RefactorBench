#!/usr/bin/env bash
# Alternate-agent add-on setup for an already-bootstrapped RefactorBench host.
#
# Use this on the Linux eval host when setup.sh has already created .venv and
# the rb-swerex image, and you only want to add/verify the Mini/tool-call backends.
# For a fresh machine, prefer: bash setup.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SWE_VENV="${SWE_VENV:-$REPO_ROOT/.venv}"
IMAGE="${IMAGE:-rb-swerex:py311-tree-sitter}"
MINI_SWE_AGENT_VERSION="${MINI_SWE_AGENT_VERSION:-2.4.4}"

log(){ printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
warn(){ printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }

[ -x "$SWE_VENV/bin/python" ] || die "missing $SWE_VENV/bin/python. Run bash setup.sh first, or set SWE_VENV=/path/to/venv."
VPY="$SWE_VENV/bin/python"

log "Installing Mini-SWE-Agent in $SWE_VENV"
"$VPY" -m pip install -q "mini-swe-agent==$MINI_SWE_AGENT_VERSION" pyyaml \
  || die "Mini-SWE-Agent install failed. Check host pip/proxy settings."

log "Verifying Mini-SWE-Agent imports"
"$VPY" - <<'PY' || die "Mini-SWE-Agent import/API check failed"
import minisweagent
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import LimitsExceeded, Submitted
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
print("minisweagent", getattr(minisweagent, "__version__", "unknown"))
print("imports ok")
PY

log "Checking runner syntax"
"$VPY" -m py_compile scripts/run_common.py scripts/run_toolcall_model.py \
  scripts/run_mini_model.py scripts/compare_agent_control.py \
  || die "alternate runner syntax check failed"

log "Checking Docker image $IMAGE"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "Docker image $IMAGE not found. Run bash setup.sh or rebuild with IMAGE=$IMAGE bash setup.sh."
  docker run --rm "$IMAGE" python3 -m pip show tree-sitter tree-sitter-languages >/dev/null \
    || die "$IMAGE is missing tree-sitter packages. Rebuild with: docker rmi $IMAGE && bash setup.sh"
else
  die "Docker is not available/running. Required for Mini eval runs."
fi

log "Ensuring smoke_instances.yaml exists"
if [ ! -f scripts/smoke_instances.yaml ]; then
  "$VPY" - <<'PY' || warn "could not create scripts/smoke_instances.yaml"
import yaml
d = yaml.safe_load(open("scripts/descriptive_instances.yaml", encoding="utf-8"))
one = [next(x for x in d if x["problem_statement"]["id"] == "add-log-parameter-get-debug-flag-task")]
yaml.safe_dump(one, open("scripts/smoke_instances.yaml", "w", encoding="utf-8"), sort_keys=False, width=10**9)
print("wrote scripts/smoke_instances.yaml")
PY
else
  echo "scripts/smoke_instances.yaml already exists"
fi

log "Alternate-agent setup complete"
cat <<EOF
Try the native bash tool-call smoke run:
  source scripts/env.sh
  python scripts/run_toolcall_model.py --model pangu --variant descriptive \\
    --instances scripts/smoke_instances.yaml --slug toolcall-smoke \\
    --image $IMAGE --startup-timeout 1800 --command-timeout 30 --workers 1

Try the Mini smoke run:
  source scripts/env.sh
  python scripts/run_mini_model.py --model pangu --variant descriptive \\
    --instances scripts/smoke_instances.yaml --slug mini-smoke \\
    --image $IMAGE --startup-timeout 1800 --command-timeout 30 --workers 1

After it finishes:
  python scripts/compare_agent_control.py runs/mini-smoke__descriptive
EOF
