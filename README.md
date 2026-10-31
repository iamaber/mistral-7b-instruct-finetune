# PubMedQA LoRA+ Fine-Tuning on Modal

This repository is a single-purpose LoRA+ fine-tuning workspace for `qiaojin/PubMedQA`, with a Modal-hosted notebook as the primary training interface. The notebook owns the main training, evaluation, export, and publishing flow. The Python scripts remain available as secondary helpers for automation and local fallback.

The training path is explicitly non-quantized. The base model is loaded in full precision based on `model_dtype` from [`configs/loraplus_train.json`](configs/loraplus_train.json): `float32`, `float16`, `bfloat16`, or `auto`.

**Project links:** [source](https://github.com/iamaber/mistral-7b-instruct-finetune) · [portfolio case study](https://iamaber.github.io/portfolio-website/#projects)

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

## Evaluation Notes

The published artifacts from this project came from a single evaluation run that
was not fit to publish. Three things must be read carefully:

- **PubMedQA collapsed.** Every one of the 100 predictions was `yes`, so the
  reported accuracy of 0.4500 equals the majority-class rate of that split
  (45 `yes` / 40 `no` / 15 `maybe`), and macro F1 (0.2069) is what an all-`yes`
  predictor scores. The cause of the collapse is **not established**: it was
  not reproduced here (no model was loaded) and it cannot be attributed to the
  scoring defects fixed below. The evaluator now reports the predicted-label
  distribution, the majority-class baseline and a `collapsed` flag, so a
  degenerate predictor is visible instead of being published as a score.
- **The medical MMLU figure covered one subject.** The sample budget took the
  first 50 rows of a subject-filtered `cais/mmlu` split. Those rows are all
  `anatomy` (measured against the current dataset revision), which is why the
  other five medical subjects show 0 samples. The budget is now spread across
  the requested subjects and zero-sample subjects are recorded in the report.
- **The metrics were computed once and copied to both cards.** They describe
  the adapter run, not each repository separately.

The card generator in [`hub.py`](src/pubmedqa_loraplus/hub.py) renders these
warnings into the model card, so a future run cannot publish a collapsed
predictor or a single-subject MMLU figure without saying so.

### Defects fixed in the scoring path

These are real defects in `src/pubmedqa_loraplus/evaluation.py`, fixed with
CPU-only tests. None of them is shown to have produced the published numbers:

- **Causal offset.** `_score_candidate` indexed logits from `prompt_length`,
  but a causal LM's logits at position `i` predict token `i + 1`. The position
  that predicts the first answer token was therefore never read. The Modal
  notebook, which produced the published run, already used
  `prompt_length - 1`, so this bug lived only in the `src/` path.
- **Candidate length normalisation.** Summed log-probabilities favour shorter
  candidates. All published PubMedQA candidates (`" yes</s>"`, `" no</s>"`,
  `" maybe</s>"`) are two tokens, so there was no length bias to exploit in
  that run; averaging is applied anyway because the tokenizer decides this.
- **Prompt budgeting.** The joined prompt used to be sliced from the left,
  which can remove `Question: ... [/INST]`. On the published split the longest
  of 100 prompts was 688 tokens against a 1024 budget, so the slice never
  truncated; the head/context/tail split keeps the suffix for longer inputs.


## Tests

The scoring, aggregation, prompt-budgeting and card-generation logic is covered
by CPU-only unit tests that do not load a model:

```bash
uv run --extra dev pytest
```

`pytest` alone also works inside a virtualenv that has the `dev` extra
installed. The suite is stdlib + pytest only, so it runs on CPU in a second.

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

The scripts are still available for automation and local fallback, but they are no longer the primary documented path:

```bash
. .venv/bin/activate
python scripts/prepare_assets.py
python scripts/train.py --config configs/loraplus_train.json
python scripts/evaluate.py --config configs/loraplus_train.json --artifact adapter
python scripts/export_model.py --config configs/loraplus_train.json
python scripts/push_to_hub.py --config configs/loraplus_train.json --artifact adapter
```

## License

Copyright (c) 2025 iamaber

This project is released under the Apache License 2.0; see [LICENSE](LICENSE)
for the full terms. This matches the two published PubMedQA model cards and
the Apache-2.0 Mistral base model. Third-party models and datasets retain
their own terms.
