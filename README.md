# PubMedQA LoRA+ Fine-Tuning on Modal

This repository is a single-purpose LoRA+ fine-tuning workspace for `qiaojin/PubMedQA`, with a Modal-hosted notebook as the primary training interface. The notebook owns the main training, evaluation, export, and publishing flow. The Python scripts remain available as secondary helpers for automation and local fallback.

The training path is explicitly non-quantized. The base model is loaded in full precision based on `model_dtype` from [`configs/loraplus_train.json`](configs/loraplus_train.json): `float32`, `float16`, `bfloat16`, or `auto`.

## Primary Modal Workflow

1. Start a Modal notebook on one strong GPU such as `A100`, `H100`, or `L40S`.
2. Mount a persistent Modal Volume and expose it as `PUBMEDQA_MODAL_ROOT`.
3. Provide `HF_TOKEN` and optional `WANDB_API_KEY`, `ADAPTER_HUB_REPO_ID`, and `MERGED_HUB_REPO_ID`.
4. Upload and run [`notebooks/pubmedqa_loraplus_notebook.ipynb`](notebooks/pubmedqa_loraplus_notebook.ipynb).

The notebook:
- downloads or reuses the cached dataset and base model from the Modal Volume
- trains the LoRA+ adapter directly in the notebook
- evaluates PubMedQA and medical MMLU
- exports adapter and merged model artifacts
- optionally pushes artifacts to Hugging Face

## Volume Layout

Inside the Modal Volume, the notebook uses a flat layout:

- `base_model/`
- `dataset/`
- `runs/<run_name>/`
- `reports/`

This keeps the persistent training state easy to inspect and resume.

## Repository Layout

```text
.
├── configs/
│   └── loraplus_train.json
├── data/
│   └── pubmedqa/                       # Local fallback dataset cache
├── models/
│   ├── base_model/                     # Local fallback base model cache
│   └── pubmedqa-loraplus/              # Local fallback run outputs
├── notebooks/
│   └── pubmedqa_loraplus_notebook.ipynb
├── reports/                            # Generated reports, gitignored
├── scripts/                            # Secondary helpers
├── src/
│   └── pubmedqa_loraplus/
├── .env.example
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Shared Config

[`configs/loraplus_train.json`](configs/loraplus_train.json) remains the shared default config. The Modal notebook reads it for default hyperparameters, then rewrites runtime paths to the mounted Modal Volume.

Modal-specific fields in the config:

- `modal_volume_name`
- `modal_mount_root`
- `modal_gpu`
- `modal_timeout_minutes`
- `modal_secret_name`
- `modal_image_name`

Common environment overrides the notebook accepts:

- `RUN_NAME`
- `BASE_MODEL_ID`
- `MODEL_DTYPE`
- `TRAIN_BATCH_SIZE`
- `EVAL_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `LEARNING_RATE`
- `NUM_EPOCHS`
- `PUBMEDQA_MODAL_ROOT`

## Training Objective

The project trains with masked-answer supervision:

- prompt tokens are masked in the labels
- only answer tokens contribute to the loss
- PubMedQA task metrics and a medical MMLU slice are reported for each run

## Local Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Environment variables live in `.env`:

```dotenv
HF_TOKEN=
WANDB_API_KEY=
WANDB_PROJECT=pubmedqa-loraplus
WANDB_ENTITY=
WANDB_MODE=offline
BASE_MODEL_ID=
PUBMEDQA_MODAL_ROOT=/vol/pubmedqa-loraplus
```

## Notebook

Use [notebooks/pubmedqa_loraplus_notebook.ipynb](notebooks/pubmedqa_loraplus_notebook.ipynb). It is Modal-first:

- no Kaggle assumptions
- no subprocess call into `scripts/train.py`
- no fixed two-GPU requirement
- direct in-notebook LoRA+ training on a single strong GPU
- persistence through the mounted Modal Volume

## Secondary Helpers

The scripts remain available as secondary helpers for automation and local fallback:

```bash
. .venv/bin/activate
python scripts/prepare_assets.py
python scripts/train.py --config configs/loraplus_train.json
python scripts/evaluate.py --config configs/loraplus_train.json --artifact adapter
python scripts/export_model.py --config configs/loraplus_train.json
python scripts/push_to_hub.py --config configs/loraplus_train.json --artifact adapter
```
