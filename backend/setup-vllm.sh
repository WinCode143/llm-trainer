#!/usr/bin/env bash
# Creates backend/.venv-vllm: vLLM generates GRPO completions much faster than transformers.
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv-vllm --python 3.12 --allow-existing
uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt
.venv-vllm/bin/python -c "import vllm, trl; print('vLLM', vllm.__version__, '· TRL', trl.__version__)"
