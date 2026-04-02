#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set"
  exit 1
fi

echo "Hugging Face token configured"
echo "Modal volume: pubmedqa-loraplus"
echo "Modal GPU: A100"

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "Weights & Biases enabled"
else
  echo "Weights & Biases not configured (optional)"
fi

echo "Bootstrap complete"
