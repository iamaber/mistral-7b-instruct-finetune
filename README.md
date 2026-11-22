# PubMedQA LoRA+ Fine-Tuning

LoRA+ fine-tuning and evaluation for biomedical yes/no/maybe question answering on
[PubMedQA](https://pubmedqa.github.io/). The model reads a PubMed abstract and a research
question, writes a short conclusion, and answers `yes`, `no` or `maybe`.

## Models

| Model | PubMedQA accuracy | Macro-F1 | Base model zero-shot (acc / F1) |
|---|---|---|---|
| [iamaber/mistral-7b-pubmedqa-loraplus](https://huggingface.co/iamaber/mistral-7b-pubmedqa-loraplus) | 0.728 (3 seeds: 0.749 ± 0.023) | 0.583 (3 seeds: 0.584 ± 0.013) | 0.660 / 0.530 |
| [iamaber/qwen3.8-27b-pubmedqa-loraplus](https://huggingface.co/iamaber/qwen3.8-27b-pubmedqa-loraplus) | 0.790 | 0.645 | 0.772 / 0.592 |

All numbers are on PubMedQA's official 500-question test set, the same questions for the
fine-tune and its base model. Each model card lists the paired 95% intervals, the per-seed
results and medical MMLU before and after fine-tuning. These are research artifacts, not
medical tools: do not use them for clinical decisions.

### Use a published adapter

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "iamaber/mistral-7b-pubmedqa-loraplus").eval()

prompt = (
    "Context: <abstract text>\n\nQuestion: <question>\n\n"
    "Write a short conclusion supported by the abstract, then give the final answer on its "
    "own line as 'Answer: yes', 'Answer: no', or 'Answer: maybe'."
)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True, return_tensors="pt", return_dict=True,
).to(model.device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

For the Qwen3.8 adapter, use `Qwen/Qwen3.8-27B` as the base and pass
`enable_thinking=False` to `apply_chat_template`.

## How a run is trained

- **Target.** With `target_format: "rationale"` (the default) the reply is PubMedQA's
  `long_answer` conclusion followed by `Answer: <label>`. `"label"` trains on the decision
  word alone; in the sweep it collapsed toward `yes`, as the first published run did.
- **Loss.** Prompt tokens are masked, so only the reply is supervised.
- **Class balance.** `balance_train_classes` oversamples `no` and `maybe` up to the `yes`
  count. No example is dropped.
- **Extra data.** `pqa_artificial_samples` adds a yes/no-balanced sample of PubMedQA's
  `pqa_artificial` subset (0 by default; it did not help in the sweep).
- **Adapter.** LoRA+ on every linear layer (`target_modules: "all-linear"`), rank 32,
  alpha 64, learning rate 1e-4 with a LoRA+ ratio of 4.
- **Checkpoint selection.** Every `eval_steps`, the validation questions are scored with the
  same predictor as the test set, and the checkpoint with the best macro-F1 is kept. A
  config whose run would end before its first evaluation is rejected.
- **Prompts** come from the tokenizer's own chat template, so the same code trains Mistral,
  Qwen2.5 and Qwen3.8. `chat_template_kwargs` passes template options such as
  `{"enable_thinking": false}`.

## How a run is evaluated

- **Test set.** PubMedQA's official 500 questions, selected by PMID from the ground truth
  vendored in `src/pubmedqa_loraplus/pqal_test_ground_truth.json` (from
  [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa), MIT). Validation is a
  label-stratified 10% of the other 500 questions; train is the rest. No question is in two
  splits.
- **Scoring.** A rationale run first writes its conclusion, then the log-probabilities of
  `yes`, `no` and `maybe` are read at the token after `Answer:`. A prediction exists even when
  the model never writes the answer line.
- **Baseline.** The untuned base model is scored zero-shot on the same questions and prompt.
  The report holds the fine-tune minus base difference with a paired bootstrap 95% interval.
  Baselines are cached by every setting they depend on, including library versions.
- **Uncertainty.** Accuracy and macro-F1 carry bootstrap 95% intervals.
- **Collapse check.** The report lists predicted-label counts, the majority-class rate and a
  `collapsed` flag.
- **Medical MMLU.** 300 questions over six medical subjects, scored twice: the option letter
  as the first reply token, and the letter after a short written rationale. A drop in both
  means lost knowledge, not only a changed answer format.
- **Per-question output.** `reports/<run_name>_predictions.jsonl` and
  `reports/<run_name>_baseline_predictions.jsonl` hold the gold label, prediction,
  log-probabilities and generated rationale for every test question.

## Reproduce

Needs one GPU with enough memory for the base model in bf16. Both published adapters were
trained on a single 80 GB H100; Mistral-7B should also fit on a 40 GB card.

```bash
uv sync --extra dev
cp .env.example .env            # HF_TOKEN, optional W&B settings

uv run python scripts/prepare_assets.py
uv run python scripts/evaluate.py --config configs/loraplus_train.json --artifact base
uv run python scripts/train.py --config configs/loraplus_train.json
uv run python scripts/evaluate.py --config configs/loraplus_train.json --artifact adapter
uv run python scripts/export_model.py --config configs/loraplus_train.json
uv run python scripts/push_to_hub.py --config configs/loraplus_train.json --artifact adapter
```

`train.py` runs the whole pipeline: baseline, training with checkpoint selection, test
evaluation, and the run report in `reports/<run_name>.json`. `push_to_hub.py` uploads only
model files and creates the repo private unless you pass `--public`.

For Qwen3.8-27B, merge [`configs/overrides/qwen3.8-27b.json`](configs/overrides/qwen3.8-27b.json)
into the config. [`configs/sweeps/phase2.json`](configs/sweeps/phase2.json) lists the
variants compared in the sweep.

[`notebooks/pubmedqa_loraplus_notebook.ipynb`](notebooks/pubmedqa_loraplus_notebook.ipynb)
runs the same pipeline from a hosted GPU notebook with a persistent storage mount.

## Configuration

[`configs/loraplus_train.json`](configs/loraplus_train.json) holds every default.
`base_model_revision`, `dataset_revision` and `mmlu_revision` pin the Hub commits a run loads,
and published adapters record the base revision they were trained on. Publishing is private
by default (`hub_private: true`).

## Repository layout

```text
.
├── configs/
│   ├── loraplus_train.json
│   ├── overrides/qwen3.8-27b.json
│   └── sweeps/phase2.json
├── notebooks/pubmedqa_loraplus_notebook.ipynb
├── scripts/                 # prepare_assets, train, evaluate, export_model, push_to_hub
├── src/pubmedqa_loraplus/
│   ├── config.py            # run config and schedule checks
│   ├── data.py              # official split, tokenisation, class balancing
│   ├── evaluation.py        # PubMedQA and medical MMLU scoring
│   ├── hub.py               # model card and Hub upload
│   ├── prompts.py           # chat-template prompts and decision tokens
│   ├── scoring.py           # metrics and bootstrap intervals
│   ├── splits.py            # split and sampling plans
│   ├── workflow.py          # training, evaluation and export
│   └── pqal_test_ground_truth.json
└── tests/                   # CPU-only unit tests
```

`data/`, `models/` and `reports/` are generated at runtime and gitignored.

## Tests

```bash
uv run --extra dev pytest
```

The tests run on CPU without downloading a model. CI also checks that `uv.lock` and
`requirements.lock.txt` agree and runs ruff.

## History

The first published adapter (Mistral-7B, label-only targets, a hand-built prompt and 100
questions from a random split) answered `yes` to every question: accuracy 0.45, the
majority-class rate of that split. Its checkpoint selection never ran because training ended
before the first evaluation. The current pipeline was rebuilt to make that failure
impossible to miss, and the published adapters were retrained with it.

## License

Copyright (c) 2025 iamaber. Apache License 2.0; see [LICENSE](LICENSE). Base models and
datasets keep their own licences: Mistral-7B-Instruct-v0.3 and Qwen3.8-27B are Apache-2.0,
PubMedQA is MIT.
