# PubMedQA Mistral Fine-Tuning Workspace

This repo is organized as a small experiment workspace for comparing two LoRA-based runs on PubMedQA:

- `baseline_current_lora`: keeps the existing full-prompt training behavior
- `sft_masked_lora`: adds an explicit masked-answer SFT run for cleaner supervision

Both runs save models, tokenizers, checkpoints, and evaluation outputs into separate folders so you can compare them directly.

## Repository Layout

```text
.
├── configs/                           # Experiment configs
├── data/
│   └── raw/                           # Downloaded PubMedQA dataset (gitignored)
├── docs/                              # Reference docs and papers
├── models/
│   ├── base/                          # Downloaded base model weights (gitignored)
│   ├── baselines/current_lora/        # Preserved existing artifacts
│   └── runs/                          # New training outputs (gitignored)
├── notebooks/                         # Original exploration notebook
├── reports/evaluations/               # Metrics JSON and plots
├── scripts/                           # Download, train, compare entrypoints
├── src/mistral_pubmedqa/              # Shared code
├── .env                               # Local credentials
└── main.py                            # Baseline convenience entrypoint
```

## Preserved Existing Artifacts

- Existing adapter-only weights: `models/baselines/current_lora/adapter_only/`
- Existing exported bundle: `models/baselines/current_lora/exported_adapter_bundle/`
- Existing metrics and plot:
  - `reports/evaluations/baseline_current_lora_metrics.json`
  - `reports/evaluations/baseline_current_lora_confusion_matrix.png`

## New Output Locations

### Downloaded inputs

- Base model weights: `models/base/Mistral-7B-Instruct-v0.3/`
- Saved PubMedQA dataset: `data/raw/pubmedqa/pqa_labeled/`

### Fresh training outputs

- Baseline checkpoints: `models/runs/baseline_current_lora/checkpoints/`
- Baseline adapter: `models/runs/baseline_current_lora/adapter/`
- Baseline tokenizer: `models/runs/baseline_current_lora/tokenizer/`
- Baseline metrics: `reports/evaluations/baseline_current_lora.json`

- SFT checkpoints: `models/runs/sft_masked_lora/checkpoints/`
- SFT adapter: `models/runs/sft_masked_lora/adapter/`
- SFT tokenizer: `models/runs/sft_masked_lora/tokenizer/`
- SFT metrics: `reports/evaluations/sft_masked_lora.json`

These paths make the before/after comparison explicit:

- before: `models/baselines/current_lora/...`
- after baseline rerun: `models/runs/baseline_current_lora/...`
- after SFT rerun: `models/runs/sft_masked_lora/...`

## Credentials

Set credentials in `.env`:

```dotenv
HF_TOKEN=...
WANDB_API_KEY=...
WANDB_PROJECT=pubmedqa-mistral-comparison
WANDB_ENTITY=...
WANDB_MODE=offline
RUNPOD_API_KEY=...
RUNPOD_POD_ID=...
```

Switch `WANDB_MODE` to `online` once the API key is set on the machine or RunPod pod.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Download Assets

```bash
. .venv/bin/activate
python scripts/download_assets.py
```

If you only want one asset:

```bash
python scripts/download_assets.py --dataset-only
python scripts/download_assets.py --model-only
```

Local storage note: the base `Mistral-7B-Instruct-v0.3` checkpoint is larger than the remaining free space on this machine after dependency installation, so the full model download should be done on the RunPod volume instead of this local workstation.

## Train

Baseline rerun:

```bash
. .venv/bin/activate
python scripts/train_baseline.py
```

Masked-answer SFT run:

```bash
. .venv/bin/activate
python scripts/train_sft.py
```

## Compare Results

```bash
. .venv/bin/activate
python scripts/compare_runs.py
```

## RunPod Notes

- Put the repo on a persistent RunPod volume before downloading the base model.
- Keep `models/base/`, `models/runs/`, and `data/raw/` on the mounted volume so training can resume without re-downloading.
- Fill `.env` on the pod before launching training so W&B logging is enabled automatically.
- Bootstrap the pod with:

```bash
bash scripts/runpod_bootstrap.sh
```
