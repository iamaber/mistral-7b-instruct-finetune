#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv sync
python scripts/prepare_assets.py

cat <<'EOF'
Bootstrap complete.
Base model:   models/base_model/
Dataset:      data/pubmedqa/
Run root:     models/pubmedqa-loraplus/
Local helper: python scripts/train.py --config configs/loraplus_train.json
Evaluate:     python scripts/evaluate.py --config configs/loraplus_train.json --artifact adapter
Export:       python scripts/export_model.py --config configs/loraplus_train.json
Primary path: run notebooks/pubmedqa_loraplus_notebook.ipynb inside Modal
EOF
