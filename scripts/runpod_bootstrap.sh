#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv sync
python scripts/prepare_assets.py

cat <<'EOF'
RunPod bootstrap complete.
Base model:   models/base/Mistral-7B-Instruct-v0.3/
Dataset:      data/raw/pubmedqa/pqa_labeled/
Run root:     models/runs/pubmedqa-loraplus/
Train:        accelerate launch scripts/train.py --config configs/loraplus_train.json
Evaluate:     python scripts/evaluate.py --config configs/loraplus_train.json --artifact adapter
Export:       python scripts/export_model.py --config configs/loraplus_train.json
EOF
