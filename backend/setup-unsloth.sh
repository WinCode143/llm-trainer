#!/usr/bin/env bash
# Creates backend/.venv-unsloth, used only by the training worker when the "unsloth" engine is selected.
# It's a separate environment because Unsloth pins its own transformers/TRL versions.
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv-unsloth --python 3.12 --allow-existing
uv pip install --python .venv-unsloth/bin/python -r requirements-unsloth.txt --index-strategy unsafe-best-match
.venv-unsloth/bin/python -c "import unsloth; print('Unsloth', unsloth.__version__, 'OK')"
