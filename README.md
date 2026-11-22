# PubMedQA LoRA+ Fine-Tuning on Modal

This repository is a single-purpose LoRA+ fine-tuning workspace for `qiaojin/PubMedQA`. Training and tests run on Modal through [`modal_app.py`](modal_app.py), which calls `train_model`, the same function `scripts/train.py` and the notebook run. The notebook and the Python scripts remain available as secondary paths.

The training path is explicitly non-quantized. The base model is loaded in full precision based on `model_dtype` from [`configs/loraplus_train.json`](configs/loraplus_train.json): `float32`, `float16`, `bfloat16`, or `auto`.

## Modal App (`modal run`)

One command per job. The image installs `requirements.lock.txt`, the volume
`pubmedqa-loraplus` caches the dataset, base models and runs, and the secret
`huggingface-secret` must provide `HF_TOKEN` (the Mistral weights are gated).

```bash
modal run modal_app.py --action tests                      # CPU unit tests
modal run modal_app.py --action smoke                      # L4, Qwen2.5-0.5B, 1 epoch
modal run modal_app.py --action train                      # H100, full config
modal run modal_app.py --action evaluate --artifact base   # zero-shot baseline only
modal run modal_app.py --action train --overrides '{"target_format": "label"}'
```

Each run prints a compact summary (test metrics with 95% intervals, baseline,
paired delta, best checkpoint) and writes the full report to
`reports/modal/<run_name>.json` locally and to `reports/` on the volume. The
Modal app never pushes to the Hub; publishing stays an explicit step
(`scripts/push_to_hub.py` or the notebook's last cell).

## Modal Notebook Workflow

1. Start a Modal notebook on one strong GPU such as `A100`, `H100`, or `L40S`.
2. Mount a persistent Modal Volume and expose it as `PUBMEDQA_MODAL_ROOT`.
3. Provide `HF_TOKEN` and optional `WANDB_API_KEY`, `ADAPTER_HUB_REPO_ID`, and `MERGED_HUB_REPO_ID`.
4. Upload and run [`notebooks/pubmedqa_loraplus_notebook.ipynb`](notebooks/pubmedqa_loraplus_notebook.ipynb).

The notebook:
- installs the pinned versions from `requirements.lock.txt` (exported from `uv.lock`)
- downloads or reuses the cached dataset and base model on the Modal Volume, at
  the revisions pinned in the config
- calls `train_model`, which scores the untuned base model, trains the adapter,
  and scores the adapter on PubMedQA's official 500-question test set and a
  medical MMLU sample
- logs the evaluation metrics to W&B when a run is active, and shows the adapter
  metrics, the baseline metrics and the difference between them
- exports adapter and merged model artifacts
- pushes each artifact to Hugging Face when its repo id is set. The config sets
  `adapter_hub_repo_id`, so the adapter push runs by default.
  `ADAPTER_HUB_REPO_ID` and `MERGED_HUB_REPO_ID` override the config. A new repo
  is created private unless `hub_private` is set to `false`.

## Volume Layout

Inside the Modal Volume, the notebook uses a flat layout:

- `base_model/`
- `dataset/`
- `runs/<run_name>/`
- `reports/`

`reports/` holds `<run_name>.json` (the run report the model card reads), the
per-example predictions for the adapter and the baseline, the config snapshot
and the training log history. The rest of the layout keeps the persistent
training state easy to inspect and resume.

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
│       ├── pqal_test_ground_truth.json
│       ├── prompts.py
│       ├── report.py
│       ├── scoring.py
│       ├── splits.py
│       └── workflow.py
├── tests/
│   ├── _stubs.py
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
├── modal_app.py
├── README.md
├── pyproject.toml
├── requirements.lock.txt
├── requirements.txt
└── uv.lock
```

Generated at runtime and gitignored: `data/` (dataset cache), `models/`
(base model and run output) and `reports/` (run reports).

## Shared Config

[`configs/loraplus_train.json`](configs/loraplus_train.json) is the single source of training defaults. The notebook stops with an error if the file is missing. It reads the file, applies environment overrides, then rewrites runtime paths to the mounted Modal Volume.

`base_model_revision`, `dataset_revision` and `mmlu_revision` pin the Hub
commits a run loads, so a rerun sees the same weights and data. A
`BASE_MODEL_ID` override in the notebook drops the pinned base revision unless
`BASE_MODEL_REVISION` is also set.

Prompts go through the tokenizer's own chat template, so no prompt string is
tied to Mistral. `chat_template_kwargs` passes options to that template, for
example `{"enable_thinking": false}` for a model whose template supports it.
`target_format` picks the training target, described under Training Objective.

`train_model` refuses a config whose run would end before its first evaluation,
because checkpoint selection would then never run.

The only Modal-specific field in the config is `modal_mount_root`: the
notebook reads it as the default mount point for the volume. GPU type,
timeout, secret name and image name belong to the Modal app definition that
launches the notebook, so they are not part of the training config.

Publishing is **private by default** (`hub_private: true`). Making a Hub repo
public is an explicit choice: pass `--public` to `scripts/push_to_hub.py`.

The notebook accepts these environment overrides. An empty variable counts as
unset, so the config value applies.

- `RUN_NAME`
- `BASE_MODEL_ID`, `BASE_MODEL_REVISION`
- `MODEL_DTYPE`
- `TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`
- `LEARNING_RATE`, `LORAPLUS_LR_RATIO`, `LORAPLUS_LR_EMBEDDING`
- `NUM_EPOCHS`, `LOGGING_STEPS`, `SAVE_STEPS`, `EVAL_STEPS`
- `MAX_LENGTH`, `MMLU_EVAL_SAMPLES`
- `WANDB_PROJECT`
- `ADAPTER_HUB_REPO_ID`, `MERGED_HUB_REPO_ID`
- `PUBMEDQA_MODAL_ROOT`

## Training Objective

The loss covers the assistant reply only. Prompt tokens are masked in the labels.
`target_format` sets what the reply is:

- `rationale` (the default) trains on the PubMedQA `long_answer` conclusion,
  followed by a line `Answer: <label>`.
- `label` trains on the decision word alone.

`balance_train_classes` oversamples each minority label up to the size of the
largest one. No original example is dropped.

`pqa_artificial_samples` adds a yes/no-balanced sample of PubMedQA's
`pqa_artificial` subset to the train split before balancing. It is 0 by
default. That subset has no `maybe` label, and any question in validation or
test is excluded from the sample.

## Evaluation Notes

### How a run is scored

- The test split is PubMedQA's official 500-question set. `load_pubmedqa`
  selects it by PMID from the ground truth vendored in
  `src/pubmedqa_loraplus/pqal_test_ground_truth.json` (from
  [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa), MIT). It fails if
  a PMID is missing or a label disagrees. The numbers are comparable with
  published PubMedQA results.
- Validation is a label-stratified `validation_size` (10%) of the other 500
  questions. Train is the rest, about 450 questions. No question is in two
  splits.
- The evaluator scores a decision as one token. It reads the log-probability
  of `yes`, `no` and `maybe` at the position right after the prompt and takes
  the highest. A `rationale` run first generates its conclusion greedily, up to
  `rationale_max_new_tokens`, then scores the token after `Answer:`. A
  prediction exists even when the model never writes `Answer:`. Medical MMLU
  uses the same scoring over `A` to `D`.
- Every `eval_steps`, the trainer scores the validation questions with the same
  predictor. It keeps the checkpoint with the best macro-F1.
- With `evaluate_base_model` on, `train_model` first scores the base model
  zero-shot, with the adapter disabled, on the same test questions and prompt.
  The report stores this under `evaluation.baseline`.
  `evaluation.delta_vs_baseline` holds the fine-tune minus base difference in
  accuracy and macro-F1, each with a paired bootstrap 95% interval.
- Accuracy and macro-F1 carry percentile bootstrap 95% intervals over
  `bootstrap_samples` resamples.
- The report lists the predicted-label counts, the majority-class rate and a
  `collapsed` flag, so a predictor that always answers the same label is
  visible.
- `reports/<run_name>_predictions.jsonl` and
  `reports/<run_name>_baseline_predictions.jsonl` hold one record per test
  question: PMID, gold label, prediction, the three log-probabilities, a tie
  flag and the generated rationale.
- The medical MMLU budget (`mmlu_eval_samples`) is split evenly across six
  medical subjects. The report warns about any subject with no samples.

The card generator in [`hub.py`](src/pubmedqa_loraplus/hub.py) renders the
baseline comparison and these warnings into the model card.

### The published run is stale

The adapter published at `iamaber/mistral-7b-pubmedqa-loraplus` predates
everything above. Its model card is stale until a new run is pushed. That run
used a hand-built Mistral prompt, label-only targets, no class balancing and
100 questions from a random split instead of the official test set. It planned
about 150 optimizer steps against `eval_steps=200`, so it never evaluated on
validation and kept its last step. Its reported numbers have these problems:

- **PubMedQA collapsed.** Every one of the 100 predictions was `yes`, so the
  reported accuracy of 0.4500 equals the majority-class rate of that split
  (45 `yes` / 40 `no` / 15 `maybe`), and macro F1 (0.2069) is what an all-`yes`
  predictor scores. The cause of the collapse is **not established**. It was
  not reproduced here, and no model was loaded to check it.
- **The medical MMLU figure covered one subject.** The sample budget took the
  first 50 rows of a subject-filtered `cais/mmlu` split. Those rows are all
  `anatomy` (measured against the dataset revision of the time), which is why
  the other medical subjects show 0 samples.
- **The metrics were computed once and copied to both cards.** They describe
  the adapter run, not each repository separately.

The prompt, split and scoring code behind those numbers are gone from this
repository, so a new run is not comparable with them.

## Tests

The scoring, splitting, prompt, aggregation and card-generation logic is
covered by CPU-only unit tests that do not load a model:

```bash
uv run --extra dev pytest
```

The suite needs the `dev` extra and the project dependencies. The tests import
the package, which imports `torch`, and `tests/test_splits.py` builds small
`datasets` objects. `pytest` alone also works inside a virtualenv with the
project and its `dev` extra installed.

## Local Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

For a reproducible environment that matches `uv.lock`, use
`uv sync --extra dev` (or install the pinned export,
`pip install -r requirements.lock.txt`).

`transformers` is capped below 5. Version 5 removed the `warmup_ratio` and
`safe_serialization` arguments this code relies on, and that migration belongs
with a future base-model swap.

Environment variables live in `.env`; copy the tracked template first:

```bash
cp .env.example .env
```

## Notebook

Use [notebooks/pubmedqa_loraplus_notebook.ipynb](notebooks/pubmedqa_loraplus_notebook.ipynb). It is Modal-first:

- no Kaggle assumptions
- no subprocess call into `scripts/train.py`
- no fixed two-GPU requirement
- LoRA+ training through `train_model` on a single strong GPU
- persistence through the mounted Modal Volume

## Secondary Helpers

The scripts are still available for automation and local fallback, but they are no longer the primary documented path:

```bash
. .venv/bin/activate
python scripts/prepare_assets.py
python scripts/train.py --config configs/loraplus_train.json
python scripts/evaluate.py --config configs/loraplus_train.json --artifact adapter
python scripts/evaluate.py --config configs/loraplus_train.json --artifact base
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
