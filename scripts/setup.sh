#!/usr/bin/env bash
# Fresh-machine setup (macOS arm64 / Linux x86_64). ~5-10 min, dominated by the torch download.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv (Python package manager)"; curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[mono,dev]"
.venv/bin/python scripts/fetch_weights.py          # pretrained weights -> ~/.cache/huggingface
echo "setup done. Try:  .venv/bin/scan2plan run data/c00a170fe1"
