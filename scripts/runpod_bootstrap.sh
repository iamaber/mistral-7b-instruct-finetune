#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv sync
python scripts/download_assets.py

cat <<'EOF'
RunPod bootstrap complete.
Base model:   models/base/Mistral-7B-Instruct-v0.3/
Dataset:      data/raw/pubmedqa/pqa_labeled/
Baseline out: models/runs/baseline_current_lora/
SFT out:      models/runs/sft_masked_lora/
EOF
