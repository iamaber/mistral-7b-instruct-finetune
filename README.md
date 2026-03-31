# PubMedQA LoRA+ Fine-Tuning

This repository is a single-purpose LoRA+ training workspace for fine-tuning instruct models on `qiaojin/PubMedQA`. The codebase is organized around one masked-answer training flow, one evaluation flow, one export flow, and one Hugging Face publishing flow.

The model-loading path is explicitly non-quantized. The base model is always loaded as a full-precision model dtype (`float32`, `float16`, or `bfloat16`), controlled by `model_dtype` in [`configs/loraplus_train.json`](/home/abir/Repos/mistral-7b-instruct-finetune/configs/loraplus_train.json).

## What This Repo Contains

- `configs/loraplus_train.json`: the single training config
- `src/pubmedqa_loraplus/`: shared LoRA+, evaluation, export, and publishing logic
- `scripts/prepare_assets.py`: downloads the PubMedQA dataset and the base model
- `scripts/train.py`: runs the LoRA+ training job
- `scripts/evaluate.py`: evaluates an adapter or merged artifact
- `scripts/export_model.py`: exports adapter and merged model artifacts
- `scripts/push_to_hub.py`: pushes the adapter or merged artifact to Hugging Face
- `notebooks/pubmedqa_loraplus_kaggle.ipynb`: a self-contained Kaggle notebook designed for two T4 GPUs

## Training Objective

The project trains with masked-answer supervision:

- prompt tokens are masked in the labels
- only the answer tokens contribute to the loss
- PubMedQA task metrics and a medical MMLU slice are reported for each run

## Setup

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
RUNPOD_API_KEY=
RUNPOD_POD_ID=
```

## Asset Preparation

```bash
. .venv/bin/activate
python scripts/prepare_assets.py
```

Optional:

```bash
python scripts/prepare_assets.py --dataset-only
python scripts/prepare_assets.py --model-only
```

## Cloud Training

Run the main training flow with `accelerate`:

```bash
. .venv/bin/activate
accelerate launch scripts/train.py --config configs/loraplus_train.json
```

Outputs are written under:

- checkpoints: `models/runs/pubmedqa-loraplus/checkpoints/`
- adapter: `models/runs/pubmedqa-loraplus/adapter/`
- tokenizer: `models/runs/pubmedqa-loraplus/tokenizer/`
- merged model: `models/runs/pubmedqa-loraplus/merged/`
- reports: `reports/runs/`

## Evaluation

Evaluate the saved adapter:

```bash
. .venv/bin/activate
python scripts/evaluate.py --config configs/loraplus_train.json --artifact adapter
```

Evaluate the merged model:

```bash
python scripts/evaluate.py --config configs/loraplus_train.json --artifact merged
```

## Export

Export the adapter and merged model artifacts:

```bash
. .venv/bin/activate
python scripts/export_model.py --config configs/loraplus_train.json
```

To export from a specific adapter checkpoint directory:

```bash
python scripts/export_model.py \
  --config configs/loraplus_train.json \
  --checkpoint models/runs/pubmedqa-loraplus/checkpoints/checkpoint-200
```

## Publish To Hugging Face

Set `adapter_hub_repo_id` and `merged_hub_repo_id` in `configs/loraplus_train.json`, then push the desired artifact:

```bash
. .venv/bin/activate
python scripts/push_to_hub.py --config configs/loraplus_train.json --artifact adapter
python scripts/push_to_hub.py --config configs/loraplus_train.json --artifact merged
```

## Kaggle Notebook

Use [notebooks/pubmedqa_loraplus_kaggle.ipynb](/home/abir/Repos/mistral-7b-instruct-finetune/notebooks/pubmedqa_loraplus_kaggle.ipynb) for Kaggle. The notebook is self-contained, targets both T4 GPUs, and defaults to a smaller full-precision instruct model until the final base model details are fixed.
