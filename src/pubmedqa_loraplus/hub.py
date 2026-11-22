import json
import os
from pathlib import Path
from typing import Any

from .artifact import prepare_adapter_for_publish
from .config import LoRAPlusConfig
from .prompts import build_user_message
from .splits import OFFICIAL_TEST_SPLIT

PUBMEDQA_CITATION = "\n".join(
    [
        "@inproceedings{jin2019pubmedqa,",
        "  title     = {{PubMedQA}: A Dataset for Biomedical Research Question Answering},",
        (
            "  author    = {Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping "
            "and Cohen, William W. and Lu, Xinghua},"
        ),
        (
            "  booktitle = {Proceedings of the 2019 Conference on Empirical Methods "
            "in Natural Language Processing (EMNLP-IJCNLP)},"
        ),
        "  year      = {2019},",
        "  pages     = {2567--2577},",
        "  url       = {https://aclanthology.org/D19-1259/}",
        "}",
    ]
)


def _load_report(config: LoRAPlusConfig) -> dict[str, Any] | None:
    if not config.report_path.exists():
        return None
    return json.loads(config.report_path.read_text(encoding="utf-8"))


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_interval(interval: list[float] | None) -> str:
    if not interval or len(interval) != 2:
        return "not available"
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def _baseline_section(evaluation: dict[str, Any]) -> list[str]:
    """Base-model zero-shot numbers and the paired delta, when the run recorded them."""
    baseline = evaluation.get("baseline", {}).get("pubmedqa")
    if not baseline:
        return []
    pubmedqa = evaluation.get("pubmedqa", {})
    delta = evaluation.get("delta_vs_baseline", {})
    rows = []
    for name, key in (("Accuracy", "accuracy"), ("Macro F1", "macro_f1")):
        change = delta.get(key, {})
        rows.append(
            f"| {name} | {_format_metric(baseline.get(key))} | "
            f"{_format_metric(pubmedqa.get(key))} | {_format_metric(change.get('delta'))} | "
            f"{_format_interval(change.get('interval'))} |"
        )
    lines = [
        "### Base model (zero-shot) vs this fine-tune",
        "",
        "Same questions, same prompt, adapter disabled for the baseline. The interval is a "
        "paired bootstrap over questions; a fine-tune has only shown an improvement when "
        "the interval excludes zero.",
        "",
        "| Metric | Base model | Fine-tune | Delta | 95% interval of delta |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    if baseline.get("collapsed"):
        lines.extend(["> The base model's predictions were also a single label.", ""])
    base_mmlu = evaluation.get("baseline", {}).get("mmlu_medical", {})
    tuned_mmlu = evaluation.get("mmlu_medical", {})
    if "answer_line_accuracy" in base_mmlu and "answer_line_accuracy" in tuned_mmlu:
        lines.extend(
            [
                "| Medical MMLU | Base model | Fine-tune |",
                "| --- | --- | --- |",
                "| Letter as first reply token | "
                f"{_format_metric(base_mmlu.get('overall_accuracy'))} | "
                f"{_format_metric(tuned_mmlu.get('overall_accuracy'))} |",
                "| Letter after a short rationale | "
                f"{_format_metric(base_mmlu['answer_line_accuracy'])} | "
                f"{_format_metric(tuned_mmlu['answer_line_accuracy'])} |",
                "",
                "The first row penalises a model that learned to open with a written "
                "conclusion; the second scores the letter after its own reasoning, so a drop "
                "there is the better forgetting signal.",
                "",
            ]
        )
    elif (
        base_mmlu.get("overall_accuracy") is not None
        and tuned_mmlu.get("overall_accuracy") is not None
    ):
        lines.extend(
            [
                f"Medical MMLU: base model {_format_metric(base_mmlu['overall_accuracy'])}, "
                f"fine-tune {_format_metric(tuned_mmlu['overall_accuracy'])}. MMLU is scored "
                "by the probability of the option letter as the first reply token. A fine-tune "
                "that learned to open with a written conclusion can lose that score from the "
                "format change alone, so read a drop here as unresolved, not as proven "
                "forgetting.",
                "",
            ]
        )
    return lines


def _seed_section(report: dict[str, Any]) -> list[str]:
    """Spread of the same configuration over training seeds, when recorded."""
    runs = report.get("seed_summary", {}).get("runs")
    if not runs:
        return []
    rows = [
        f"| {run['seed']} | {_format_metric(run['accuracy'])} | {_format_metric(run['macro_f1'])} |"
        for run in runs
    ]
    summary = report["seed_summary"]
    return [
        "### Seed variation",
        "",
        f"The same configuration trained with {len(runs)} seeds. This card's adapter is "
        f"seed {report['seed_summary']['published_seed']}, chosen before seeing test scores.",
        "",
        "| Seed | Accuracy | Macro F1 |",
        "| --- | --- | --- |",
        *rows,
        f"| mean ± sd | {summary['accuracy_mean']:.4f} ± {summary['accuracy_sd']:.4f} | "
        f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_sd']:.4f} |",
        "",
    ]


def _confusion_matrix_table(
    matrix: list[list[int]] | None,
    label_order: list[str],
) -> list[str]:
    if not matrix:
        return ["Confusion matrix not available."]
    header = "| Actual \\ Predicted | " + " | ".join(label_order) + " |"
    separator = "| " + " | ".join(["---"] * (len(label_order) + 1)) + " |"
    rows = [header, separator]
    for label, row in zip(label_order, matrix):
        rows.append(f"| {label} | " + " | ".join(str(cell) for cell in row) + " |")
    return rows


def _subject_table(subject_results: dict[str, dict[str, int]]) -> list[str]:
    if not subject_results:
        return ["Subject breakdown not available."]
    rows = [
        "| Subject | Accuracy | Correct | Total |",
        "| --- | --- | --- | --- |",
    ]
    for subject, result in subject_results.items():
        total = result.get("total", 0)
        correct = result.get("correct", 0)
        value = f"{correct / total:.4f}" if total else "not evaluated"
        rows.append(f"| {subject} | {value} | {correct} | {total} |")
    return rows


def _evaluation_section(report: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []

    if report is None:
        lines.extend(
            [
                "No evaluation report was found when this card was generated, so no "
                "metrics are reported here. Absence of numbers is intentional: this "
                "card does not borrow metrics from another run.",
            ]
        )
        return lines

    evaluation = report.get("evaluation", {})
    pubmedqa = evaluation.get("pubmedqa", {})
    mmlu = evaluation.get("mmlu_medical", {})
    label_order = pubmedqa.get("label_order") or ["yes", "no", "maybe"]

    lines.extend(
        [
            "These numbers come from a **single** evaluation run. The same figures are "
            "reported on both the adapter and the merged model card because the metrics "
            "were computed once, on the adapter, and then copied into both cards. They "
            "were not re-measured for each repository.",
            "",
            "### PubMedQA",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Accuracy | {_format_metric(pubmedqa.get('accuracy'))} |",
            f"| Macro F1 | {_format_metric(pubmedqa.get('macro_f1'))} |",
            f"| Weighted F1 | {_format_metric(pubmedqa.get('weighted_f1'))} |",
            (
                f"| Majority-class rate (reference) | "
                f"{_format_metric(pubmedqa.get('majority_class_rate'))} |"
            ),
            f"| Samples | {_format_metric(pubmedqa.get('num_samples'))} |",
            "",
        ]
    )

    intervals = pubmedqa.get("confidence_intervals")
    if intervals:
        lines.extend(
            [
                f"95% bootstrap intervals ({_format_metric(intervals.get('samples'))} "
                f"resamples): accuracy {_format_interval(intervals.get('accuracy'))}, "
                f"macro F1 {_format_interval(intervals.get('macro_f1'))}.",
                "",
            ]
        )

    lines.extend(_baseline_section(evaluation))
    lines.extend(_seed_section(report))

    prediction_counts = pubmedqa.get("prediction_counts")
    if prediction_counts:
        rendered = ", ".join(f"{label}: {prediction_counts.get(label, 0)}" for label in label_order)
        lines.append(f"Predicted-label distribution: {rendered}.")
        lines.append("")

    if pubmedqa.get("collapsed"):
        accuracy = pubmedqa.get("accuracy")
        baseline = pubmedqa.get("majority_class_rate")
        lines.extend(
            [
                "> **The reported evaluation is a collapsed predictor.** Every one of the "
                f"{_format_metric(pubmedqa.get('num_samples'))} PubMedQA predictions was "
                f"`{_dominant_label(prediction_counts, label_order)}`. Accuracy "
                f"({_format_metric(accuracy)}) is therefore equal to the rate you would get "
                f"by always answering the majority class "
                f"({_format_metric(baseline)}), and it does **not** show that the model "
                "learned to answer biomedical questions. Macro F1 is correspondingly low, "
                "which is the signature of this failure mode.",
                "",
            ]
        )

    lines.extend(
        [
            "### Confusion matrix",
            "",
            *_confusion_matrix_table(pubmedqa.get("confusion_matrix"), label_order),
            "",
        ]
    )

    covered = mmlu.get("covered_subjects")
    subject_results = mmlu.get("subject_results", {})
    evaluated_subjects = [
        subject for subject, result in subject_results.items() if result.get("total", 0) > 0
    ]
    lines.extend(
        [
            "### Medical MMLU",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            (
                f"| Medical MMLU accuracy (as reported) | "
                f"{_format_metric(mmlu.get('overall_accuracy'))} |"
            ),
            f"| Subjects with samples | {', '.join(covered or evaluated_subjects) or 'none'} |",
            f"| Samples | {_format_metric(mmlu.get('total_samples'))} |",
            "",
        ]
    )

    if len(covered or evaluated_subjects) == 1:
        subject = (covered or evaluated_subjects)[0]
        lines.extend(
            [
                f"> **The medical MMLU figure covers `{subject}` only.** The other five "
                "medical subjects were allocated no samples, so this is a single-subject "
                "accuracy and not a medical-MMLU average.",
                "",
            ]
        )

    lines.extend(_subject_table(subject_results))
    lines.append("")
    return lines


def _dominant_label(
    prediction_counts: dict[str, int] | None,
    label_order: list[str],
) -> str:
    if not prediction_counts:
        return "yes"
    return max(label_order, key=lambda label: prediction_counts.get(label, 0))


def _pubmedqa_collapsed(report: dict[str, Any] | None) -> bool | None:
    """Return the report's collapse verdict, or ``None`` when there is no report."""
    if report is None:
        return None
    collapsed = report.get("evaluation", {}).get("pubmedqa", {}).get("collapsed")
    return None if collapsed is None else bool(collapsed)


def _mmlu_covered_subjects(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return []
    mmlu = report.get("evaluation", {}).get("mmlu_medical", {})
    covered = mmlu.get("covered_subjects")
    if covered:
        return list(covered)
    return [
        subject
        for subject, result in mmlu.get("subject_results", {}).items()
        if result.get("total", 0) > 0
    ]


def _collapse_expectation(collapsed: bool | None) -> list[str]:
    if collapsed is True:
        return [
            "Expect a degenerate answer: in the recorded run this produced one label for "
            "every question. The snippet shows how to load the artifact, not that it works.",
        ]
    if collapsed is False:
        return [
            "The evaluation recorded in this card did not collapse: the predictions span "
            "more than one label. The snippet shows how to load the artifact, not that its "
            "answers are clinically valid.",
        ]
    return [
        "No evaluation report was available when this card was generated, so nothing is "
        "claimed here about the quality of the output. The snippet shows how to load the "
        "artifact, not that it works.",
    ]


def _limitations_section(report: dict[str, Any] | None) -> list[str]:
    lines: list[str] = ["## Limitations", ""]

    collapsed = _pubmedqa_collapsed(report)
    if collapsed is True:
        pubmedqa = (report or {}).get("evaluation", {}).get("pubmedqa", {})
        samples = pubmedqa.get("num_samples")
        label_order = pubmedqa.get("label_order") or ["yes", "no", "maybe"]
        dominant = _dominant_label(pubmedqa.get("prediction_counts"), label_order)
        scope = (
            f"All {_format_metric(samples)} held-out PubMedQA predictions"
            if samples is not None
            else "Every held-out PubMedQA prediction"
        )
        lines.append(
            f"- **The predictor collapsed.** {scope} were the same label (`{dominant}`), "
            "so the accuracy figure reflects the label distribution of the test split, not "
            "the model's ability. **The cause was not established**: the evaluation was not "
            "re-run, and the scoring defects since fixed in the evaluator are not shown to "
            "have produced it."
        )
    elif collapsed is False:
        lines.append(
            "- **No collapse in the recorded run, but no clinical validity either.** The "
            "predictions span more than one label, so the accuracy is not just the "
            "majority-class rate. The evaluation is still small and narrow (see below)."
        )
    else:
        lines.append(
            "- **No evaluation was recorded**, so this card makes no claim about prediction "
            "quality in either direction."
        )

    covered = _mmlu_covered_subjects(report)
    if len(covered) == 1:
        lines.append(
            f"- **The medical MMLU figure is one subject.** Only `{covered[0]}` received "
            "samples; the other medical subjects were allocated none."
        )
    elif len(covered) > 1:
        lines.append(
            f"- **The medical MMLU figure covers {len(covered)} subjects** "
            f"({', '.join(f'`{subject}`' for subject in covered)}); it is not the full "
            "medical-MMLU set."
        )

    lines.append(
        "- **The metrics were computed once** on the adapter run and copied to both this "
        "card and the sibling card, so they do not describe this artifact separately."
    )

    if report is not None:
        eval_examples = report.get("eval_examples")
        validation_examples = report.get("validation_examples")
        size = (
            f"{_format_metric(eval_examples)} PubMedQA examples"
            if eval_examples is not None
            else "the PubMedQA holdout"
        )
        split_note = (
            f" and {_format_metric(validation_examples)} validation examples"
            if validation_examples is not None
            else ""
        )
        if report.get("test_split") == OFFICIAL_TEST_SPLIT:
            lines.append(
                f"- **One benchmark, one run.** The reported metrics rest on {size} (PubMedQA's "
                f"official test set){split_note} from a single training seed. A 500-question "
                "test cannot resolve differences much smaller than ~9 points of accuracy."
            )
        else:
            lines.append(
                f"- **Small evaluation.** The reported metrics rest on {size}{split_note}, far "
                "too little to characterise a 7B model, and the holdout comes from PubMedQA's "
                "`pqa_labeled` training pool rather than the official test set."
            )
    else:
        lines.append(
            "- **No evaluation report.** Nothing about prediction quality on PubMedQA's "
            "official test set is recorded for this artifact."
        )

    lines.extend(
        [
            "- **English only**, and only on PubMed-style abstracts and multiple-choice questions.",
            "- **Training-data bias carries over** from the base model and from PubMedQA's "
            "published-literature selection.",
            "",
        ]
    )
    return lines


def _generation_lines(
    target_format: str, max_new_tokens: int, template_kwargs: dict[str, Any]
) -> list[str]:
    user_text = build_user_message("<question>", "<abstract text>", target_format)
    extra = "".join(f", {key}={value!r}" for key, value in template_kwargs.items())
    return [
        f'messages = [{{"role": "user", "content": {user_text!r}}}]',
        "inputs = tokenizer.apply_chat_template(",
        f'    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True{extra}',
        ").to(model.device)",
        "with torch.no_grad():",
        f"    output = model.generate(**inputs, max_new_tokens={max_new_tokens}, do_sample=False)",
        'new_tokens = output[0][inputs["input_ids"].shape[-1]:]',
        "print(tokenizer.decode(new_tokens, skip_special_tokens=True))",
    ]


def _usage_section(
    config: LoRAPlusConfig,
    artifact: str,
    repo_id: str | None,
    collapsed: bool | None,
) -> list[str]:
    target = repo_id or "<this-repo-id>"
    max_new_tokens = 8
    if config.target_format == "rationale":
        max_new_tokens = config.rationale_max_new_tokens + 8
    generation = _generation_lines(
        config.target_format, max_new_tokens, dict(config.chat_template_kwargs)
    )
    answer_note = (
        "The model writes a short conclusion and ends with `Answer: yes|no|maybe`."
        if config.target_format == "rationale"
        else "The model answers with one word: `yes`, `no` or `maybe`."
    )
    if artifact == "adapter":
        return [
            "## Usage",
            "",
            "This repository contains a PEFT LoRA adapter, not full model weights. Load the "
            "base model first and attach the adapter.",
            "",
            "```python",
            "import torch",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'base_model_id = "{config.base_model_id}"',
            "",
            "tokenizer = AutoTokenizer.from_pretrained(base_model_id)",
            "base_model = AutoModelForCausalLM.from_pretrained(",
            "    base_model_id,",
            "    dtype=torch.bfloat16,",
            '    device_map="auto",',
            ")",
            f'model = PeftModel.from_pretrained(base_model, "{target}")',
            "model.eval()",
            "",
            *generation,
            "```",
            "",
            answer_note,
            "",
            "The adapter config stores the base model as a Hub id, so the snippet above "
            "resolves the base without needing a local copy of the training files.",
            "",
            *_collapse_expectation(collapsed),
            "",
        ]

    return [
        "## Usage",
        "",
        "This repository contains merged weights, so the base model does not need to be "
        "downloaded separately.",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'tokenizer = AutoTokenizer.from_pretrained("{target}")',
        "model = AutoModelForCausalLM.from_pretrained(",
        f'    "{target}",',
        "    dtype=torch.bfloat16,",
        '    device_map="auto",',
        ")",
        "model.eval()",
        "",
        *generation,
        "```",
        "",
        answer_note,
        "",
        *_collapse_expectation(collapsed),
        "",
    ]


def build_model_card(config: LoRAPlusConfig, artifact: str) -> str:
    if artifact not in {"adapter", "merged"}:
        raise ValueError("artifact must be 'adapter' or 'merged'")

    report = _load_report(config)
    library_name = "peft" if artifact == "adapter" else "transformers"
    repo_id = config.adapter_hub_repo_id if artifact == "adapter" else config.merged_hub_repo_id
    name = repo_id.split("/")[-1] if repo_id else config.run_name
    title = f"{name} (LoRA+ adapter)" if artifact == "adapter" else f"{name} (merged)"
    purpose = (
        "It is published for reproducibility and as a record of a negative result, not "
        "as a model intended to be deployed."
        if _pubmedqa_collapsed(report)
        else "It is published for reproducibility with the evaluation below, not as a "
        "model intended to be deployed."
    )

    lines: list[str] = [
        "---",
        "license: apache-2.0",
        f"library_name: {library_name}",
        f"base_model: {config.base_model_id}",
        "pipeline_tag: text-generation",
        "language:",
        "  - en",
        "datasets:",
        f"  - {config.dataset_id}",
        "tags:",
        *(["  - peft"] if artifact == "adapter" else []),
        "  - lora",
        "  - loraplus",
        "  - pubmedqa",
        "  - medical",
        "  - experimental",
        "---",
        "",
        f"# {title}",
        "",
        "> **This is an experiment, not a usable medical model.** It is a small LoRA+ "
        "research run on PubMedQA. Do not use it for clinical decision-making, "
        "diagnosis, triage, or any other medical purpose. Read the evaluation and "
        "limitations sections before drawing any conclusion.",
        "",
        "## What this is",
        "",
        f"A LoRA+ fine-tune of `{config.base_model_id}` on "
        f"`{config.dataset_id}` / `{config.dataset_config}`, trained as a "
        "single-GPU research exercise to check the training and evaluation pipeline "
        f"end to end. {purpose}",
        "",
        "## Intended use",
        "",
        "- Inspecting what a small LoRA+ run on PubMedQA's labeled pool does to an "
        "instruction-tuned base model.",
        "- Reproducing the evaluation and extending it: whatever the run produced is "
        "recorded in the evaluation section, including an explicitly flagged collapse if "
        "the predictions were degenerate.",
        "",
        "## Out-of-scope use",
        "",
        "- Any clinical, diagnostic, triage, or treatment decision.",
        "- Any claim that this model answers biomedical questions correctly.",
        "- Any comparison against published PubMedQA results as if the reported accuracy "
        "were a real result.",
        "",
    ]

    lines.extend(_usage_section(config, artifact, repo_id, _pubmedqa_collapsed(report)))

    lines.extend(
        [
            "## Training setup",
            "",
            f"- Base model: `{config.base_model_id}`",
            f"- Dataset: `{config.dataset_id}` / `{config.dataset_config}`",
            f"- Artifact type: `{artifact}`",
            f"- Target: `{config.target_format}`",
            (
                f"- LoRA+: rank={config.lora_r}, alpha={config.lora_alpha}, "
                f"dropout={config.lora_dropout}, lr_ratio={config.loraplus_lr_ratio}"
            ),
            f"- Learning rate: {config.learning_rate}",
            f"- Epochs: {config.num_epochs}",
            (
                f"- Batch: per-device {config.train_batch_size} with "
                f"{config.gradient_accumulation_steps} gradient-accumulation steps"
            ),
            "",
        ]
    )

    if report is not None:
        training = report.get("training", {})
        training_metrics = training.get("train_metrics", {})
        summary = report.get("training_summary", {})
        rows = [
            ("Train examples", _format_metric(report.get("train_examples"))),
            ("Eval examples", _format_metric(report.get("eval_examples"))),
            ("Global step", _format_metric(training.get("global_step"))),
            (
                "Train loss (final)",
                _format_metric(training_metrics.get("train_loss")),
            ),
            (
                "Best validation macro F1 (selection metric)",
                _format_metric(summary.get("best_validation_macro_f1")),
            ),
            (
                "Latest eval loss",
                _format_metric(summary.get("latest_eval_loss")),
            ),
            (
                "Train runtime (s)",
                _format_metric(summary.get("train_runtime")),
            ),
        ]
        lines.extend(
            [
                "### Reported run details",
                "",
                "| Field | Value |",
                "| --- | --- |",
                *[f"| {name} | {value} |" for name, value in rows],
                "",
            ]
        )

    lines.extend(["## Evaluation", ""])
    lines.extend(_evaluation_section(report))

    lines.extend(
        [
            "## Training data",
            "",
            "`qiaojin/PubMedQA` (`pqa_labeled`), 1,000 expert-annotated biomedical "
            "research questions with yes/no/maybe decisions. Runs from this pipeline report "
            "on PubMedQA's official 500-question test set, selected by PMID, and train on the "
            "other 500 minus a stratified validation split used for checkpoint selection. "
            "PubMedQA is released under the MIT licence; its abstracts are derived from "
            "PubMed and remain subject to the original publishers' terms.",
            "",
        ]
    )
    lines.extend(_limitations_section(report))
    lines.extend(
        [
            "## Licence",
            "",
            "- This adapter/model is released under **Apache-2.0**.",
            f"- Using or redistributing the base model `{config.base_model_id}` is subject "
            "to the licence that accompanies it, not to this repository alone.",
            "- PubMedQA is MIT-licensed; see its dataset card for the abstract provenance terms.",
            "",
            "## Citation",
            "",
            "If you use PubMedQA, cite the dataset paper:",
            "",
            "```bibtex",
            PUBMEDQA_CITATION,
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def push_artifact_to_hub(
    config: LoRAPlusConfig,
    artifact: str,
    source_dir: str | Path | None = None,
) -> dict[str, str]:
    if artifact not in {"adapter", "merged"}:
        raise ValueError("artifact must be 'adapter' or 'merged'")

    repo_id = config.adapter_hub_repo_id if artifact == "adapter" else config.merged_hub_repo_id
    if not repo_id:
        raise ValueError(f"No Hugging Face repo configured for {artifact} artifact")

    artifact_dir = (
        Path(source_dir)
        if source_dir is not None
        else (config.adapter_output_dir if artifact == "adapter" else config.merged_output_dir)
    )
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")

    allow_patterns = ["README.md", "tokenizer*", "special_tokens_map.json", "chat_template.jinja"]
    if artifact == "adapter":
        prepare_adapter_for_publish(artifact_dir, config.base_model_id)
        allow_patterns.extend(["adapter_config.json", "adapter_model.safetensors"])
    else:
        allow_patterns.extend(
            [
                "config.json",
                "generation_config.json",
                "*.safetensors",
                "model.safetensors.index.json",
            ]
        )

    from huggingface_hub import HfApi

    token = os.getenv(config.hub_token_env) or None
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=config.hub_private,
        exist_ok=True,
    )

    readme_path = artifact_dir / "README.md"
    readme_path.write_text(build_model_card(config, artifact), encoding="utf-8")
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=allow_patterns,
    )

    return {
        "artifact": artifact,
        "repo_id": repo_id,
        "source_dir": str(artifact_dir),
    }
