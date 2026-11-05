#!/usr/bin/env bash
# Activate the torch-compile-guardrails project venv (local editable PyTorch).
#
# Usage (from anywhere):
#   source /workspaces/pytorch-devcontainers/torch-compile-guardrails/scripts/activate_guardrails_env.sh
#
# Optional MIG pin:
#   GUARDRAILS_MIG_UUID=MIG-... source .../activate_guardrails_env.sh

_GUARDRAILS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_PYTORCH_ROOT="${GUARDRAILS_PYTORCH_ROOT:-/workspaces/pytorch-devcontainers/pytorch}"

if [[ ! -x "${_GUARDRAILS_ROOT}/.venv/bin/python" ]]; then
  echo "guardrails .venv missing. From ${_GUARDRAILS_ROOT} run:"
  echo "  uv sync --extra dev --python 3.12 --no-install-package torch --no-install-package triton"
  echo "  bash scripts/link_local_torch.sh"
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${_GUARDRAILS_ROOT}/.venv/lib/python3.12/site-packages/_editable_skbc_torch.pth" ]] \
  && [[ ! -f "${_GUARDRAILS_ROOT}/.venv/lib64/python3.12/site-packages/_editable_skbc_torch.pth" ]]; then
  echo "Local torch hooks missing. From ${_GUARDRAILS_ROOT} run:"
  echo "  bash scripts/link_local_torch.sh"
  return 1 2>/dev/null || exit 1
fi

# Leave civic-evals / other venvs. Do NOT put the pytorch source tree on
# PYTHONPATH — that shadows the skbuild editable install and breaks CUDA libs.
unset VIRTUAL_ENV
unset PYTHONPATH

# shellcheck disable=SC1091
source "${_GUARDRAILS_ROOT}/.venv/bin/activate"

# Optional OpenUCX path used by some local pytorch builds (ignore if absent).
for _ucx in \
  "${HOME}/.local/openucx/ucx/lib" \
  "${HOME}/.local/openucx/ucc/lib"
do
  if [[ -d "${_ucx}" ]]; then
    export LD_LIBRARY_PATH="${_ucx}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done
unset _ucx

# Prefer an explicit MIG UUID; otherwise keep whatever the container exposes.
export CUDA_VISIBLE_DEVICES="${GUARDRAILS_MIG_UUID:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"
export GUARDRAILS_PYTORCH_ROOT="${_PYTORCH_ROOT}"

echo "torch-compile-guardrails env active"
echo "  python:  $(command -v python)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -c "import sys; print('  venv ok:', 'torch-compile-guardrails' in sys.executable, sys.executable)"
python -c "import torch; print('  torch:', torch.__version__, 'cuda=' + str(torch.cuda.is_available())); print('  device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" \
  || echo "  WARNING: torch import failed"
python -c "import triton; print('  triton:', getattr(triton, '__version__', 'unknown'))" \
  || echo "  WARNING: triton import failed"
python -c "import transformers; print('  transformers:', transformers.__version__)" \
  || echo "  WARNING: transformers import failed"
