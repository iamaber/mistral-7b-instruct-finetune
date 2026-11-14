# PubMedQA LoRA+ Fine-Tuning on Modal

This repository is a single-purpose LoRA+ fine-tuning workspace for `qiaojin/PubMedQA`, with a Modal-hosted notebook as the primary training interface. The notebook owns the main training, evaluation, export, and publishing flow. The Python scripts remain available as secondary helpers for automation and local fallback.

The training path is explicitly non-quantized. The base model is loaded in full precision based on `model_dtype` from [`configs/loraplus_train.json`](configs/loraplus_train.json): `float32`, `float16`, `bfloat16`, or `auto`.

## Primary Modal Workflow

1. Start a Modal notebook on one strong GPU such as `A100`, `H100`, or `L40S`.
2. Mount a persistent Modal Volume and expose it as `PUBMEDQA_MODAL_ROOT`.
3. Provide `HF_TOKEN` and optional `WANDB_API_KEY`, `ADAPTER_HUB_REPO_ID`, and `MERGED_HUB_REPO_ID`.
4. Upload and run [`notebooks/pubmedqa_loraplus_notebook.ipynb`](notebooks/pubmedqa_loraplus_notebook.ipynb).

The notebook:
- installs the pinned versions from `requirements.lock.txt` (exported from `uv.lock`)
- downloads or reuses the cached dataset and base model from the Modal Volume
- trains the LoRA+ adapter directly in the notebook, selecting the checkpoint on
  a validation split that is disjoint from the reported test split
- evaluates PubMedQA and medical MMLU
- exports adapter and merged model artifacts
- optionally pushes artifacts to Hugging Face, but only when
  `ADAPTER_HUB_REPO_ID` / `MERGED_HUB_REPO_ID` are set; the repos are created
  private unless `hub_private` is set to `false`

## Volume Layout

Inside the Modal Volume, the notebook uses a flat layout:

- `base_model/`
- `dataset/`
- `runs/<run_name>/`
- `reports/`

This keeps the persistent training state easy to inspect and resume.

## Repository Layout

Tracked files (`git ls-files`):

```text
.
├── configs/
│   └── loraplus_train.json
├── notebooks/
│   └── pubmedqa_loraplus_notebook.ipynb
├── scripts/
│   ├── bootstrap.sh
│   ├── evaluate.py
│   ├── export_model.py
│   ├── prepare_assets.py
│   ├── push_to_hub.py
│   └── train.py
├── src/
│   └── pubmedqa_loraplus/
│       ├── __init__.py
│       ├── artifact.py
│       ├── config.py
│       ├── data.py
│       ├── env.py
│       ├── evaluation.py
│       ├── hub.py
│       ├── paths.py
│       ├── prompts.py
│       ├── report.py
│       ├── scoring.py
│       └── workflow.py
├── tests/
│   ├── conftest.py
│   ├── test_aggregation.py
│   ├── test_artifact.py
│   ├── test_data.py
│   ├── test_model_card.py
│   ├── test_prompts.py
│   ├── test_report.py
│   ├── test_scoring.py
│   └── test_splits.py
├── .env.example
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.lock.txt
├── requirements.txt
└── uv.lock
```

Generated at runtime and gitignored: `data/` (dataset cache), `models/`
(base model and run output) and `reports/` (run reports).

## Shared Config

[`configs/loraplus_train.json`](configs/loraplus_train.json) remains the shared default config. The Modal notebook reads it for default hyperparameters, then rewrites runtime paths to the mounted Modal Volume.

The only Modal-specific field in the config is `modal_mount_root`: the
notebook reads it as the default mount point for the volume. GPU type,
timeout, secret name and image name belong to the Modal app definition that
launches the notebook, so they are not part of the training config.

Publishing is **private by default** (`hub_private: true`). Making a Hub repo
public is an explicit choice: pass `--public` to `scripts/push_to_hub.py`.

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

### Fixed: selection no longer uses the reported split

The published run selected its checkpoint with `load_best_model_at_end` on
`eval_loss` and then reported PubMedQA on those same 100 examples, so model
selection leaked into the headline number. `load_pubmedqa` now returns three
disjoint splits — `train`, `validation` (used for checkpoint selection) and
`test` (used only for the reported metrics) — configured by `validation_size`.
`tests/test_splits.py` pins that the splits are pairwise disjoint and cover every
example exactly once. A future run that reports `test` metrics is no longer
choosing its checkpoint on the same examples.

The caveats below describe the already-published run, which predates that split.

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

For a reproducible environment that matches `uv.lock`, use
`uv sync --extra dev` (or install the pinned export,
`pip install -r requirements.lock.txt`).

Environment variables live in `.env`; copy the tracked template first:

```bash
cp .env.example .env
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

`push_to_hub.py` creates the Hub repo **private** unless you pass `--public`.
The adapter and merged model directories must exist locally (the notebook
writes them under the Modal Volume's `runs/<run_name>/`).

## License

Copyright (c) 2025 iamaber

This project is released under the Apache License 2.0; see [LICENSE](LICENSE)
for the full terms. This matches the two published PubMedQA model cards and
the Apache-2.0 Mistral base model. Third-party models and datasets retain
their own terms.
