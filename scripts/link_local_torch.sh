#!/usr/bin/env bash
# Wire the local editable PyTorch + Triton builds into this project's .venv
# without installing torch/triton from PyPI.
#
# Important: do NOT add the pytorch source tree to PYTHONPATH. That shadows the
# skbuild editable install and loads source-tree .so files that miss UCX/RUNPATH
# wiring. Instead copy the skbuild editable hooks from pytorch/.venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTORCH_ROOT="${GUARDRAILS_PYTORCH_ROOT:-/workspaces/pytorch-devcontainers/pytorch}"
PYTORCH_VENV="${PYTORCH_ROOT}/.venv"

SITE="$(echo "${ROOT}"/.venv/lib/python*/site-packages)"
if [[ ! -d "${SITE}" ]]; then
  SITE="$(echo "${ROOT}"/.venv/lib64/python*/site-packages)"
fi

if [[ ! -d "${SITE}" ]]; then
  echo "No site-packages under ${ROOT}/.venv — run:" >&2
  echo "  uv sync --extra dev --python 3.12 --no-install-package torch --no-install-package triton" >&2
  exit 1
fi

PYTORCH_SITE="${PYTORCH_VENV}/lib64/python3.12/site-packages"
if [[ ! -d "${PYTORCH_SITE}" ]]; then
  PYTORCH_SITE="${PYTORCH_VENV}/lib/python3.12/site-packages"
fi

if [[ ! -f "${PYTORCH_SITE}/_editable_skbc_torch.pth" ]]; then
  echo "Editable torch hooks missing in ${PYTORCH_SITE}" >&2
  echo "Build/install pytorch into ${PYTORCH_VENV} first." >&2
  exit 1
fi

# Drop any previous source-tree .pth that would break imports.
rm -f "${SITE}/_local_pytorch.pth"

cp -a "${PYTORCH_SITE}/_editable_skbc_torch.pth" "${SITE}/_editable_skbc_torch.pth"
cp -a "${PYTORCH_SITE}/_editable_skbc_torch.py" "${SITE}/_editable_skbc_torch.py"

# Version metadata for tooling.
shopt -s nullglob
for meta in "${PYTORCH_SITE}"/torch-*.dist-info; do
  ln -sfn "${meta}" "${SITE}/$(basename "${meta}")"
done
shopt -u nullglob

# Triton ships in the pytorch .venv; expose just that package.
link_pkg() {
  local name="$1"
  local src="${PYTORCH_SITE}/${name}"
  local dst="${SITE}/${name}"
  if [[ ! -e "${src}" ]]; then
    echo "WARNING: missing ${src}" >&2
    return 0
  fi
  rm -rf "${dst}"
  ln -sfn "${src}" "${dst}"
}

link_pkg "triton"
shopt -s nullglob
for meta in "${PYTORCH_SITE}"/triton-*.dist-info; do
  ln -sfn "${meta}" "${SITE}/$(basename "${meta}")"
done
shopt -u nullglob

echo "Linked local editable pytorch + triton into ${SITE}"
echo "  pytorch hooks: ${PYTORCH_SITE}/_editable_skbc_torch.*"
echo "  triton:        ${PYTORCH_SITE}/triton"
