import json
import os
from pathlib import Path
from typing import Any

from .artifact import prepare_adapter_for_publish
from .config import LoRAPlusConfig

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

    prediction_counts = pubmedqa.get("prediction_counts")
    if prediction_counts:
        rendered = ", ".join(
            f"{label}: {prediction_counts.get(label, 0)}" for label in label_order
        )
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
        subject
        for subject, result in subject_results.items()
        if result.get("total", 0) > 0
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


def _usage_section(artifact: str, base_model_id: str, repo_id: str | None) -> list[str]:
    target = repo_id or "<this-repo-id>"
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
            f'base_model_id = "{base_model_id}"',
            "",
            "tokenizer = AutoTokenizer.from_pretrained(base_model_id)",
            "base_model = AutoModelForCausalLM.from_pretrained(",
            "    base_model_id,",
            "    torch_dtype=torch.bfloat16,",
            "    device_map=\"auto\",",
            ")",
            f'model = PeftModel.from_pretrained(base_model, "{target}")',
            "model.eval()",
            "",
            "prompt = (",
            "    \"<s>[INST] Context: <abstract text>\\n\\n\"",
            "    \"Question: <question> [/INST]\"",
            ")",
            "inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)",
            "with torch.no_grad():",
            "    output = model.generate(**inputs, max_new_tokens=8, do_sample=False)",
            "new_tokens = output[0][inputs[\"input_ids\"].shape[-1]:]",
            "print(tokenizer.decode(new_tokens, skip_special_tokens=True))",
            "```",
            "",
            "The adapter config stores the base model as a Hub id, so the snippet above "
            "resolves the base without needing a local copy of the training files.",
            "",
            "Expect a degenerate answer: in the recorded run this produced one label for "
            "every question. The snippet shows how to load the artifact, not that it works.",
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
        "    torch_dtype=torch.bfloat16,",
        "    device_map=\"auto\",",
        ")",
        "model.eval()",
        "",
        "prompt = (",
        "    \"<s>[INST] Context: <abstract text>\\n\\n\"",
        "    \"Question: <question> [/INST]\"",
        ")",
        "inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)",
        "with torch.no_grad():",
        "    output = model.generate(**inputs, max_new_tokens=8, do_sample=False)",
        "new_tokens = output[0][inputs[\"input_ids\"].shape[-1]:]",
        "print(tokenizer.decode(new_tokens, skip_special_tokens=True))",
        "```",
        "",
        "Expect a degenerate answer: in the recorded run this produced one label for "
        "every question. The snippet shows how to load the artifact, not that it works.",
        "",
    ]


def build_model_card(config: LoRAPlusConfig, artifact: str) -> str:
    if artifact not in {"adapter", "merged"}:
        raise ValueError("artifact must be 'adapter' or 'merged'")

    report = _load_report(config)
    library_name = "peft" if artifact == "adapter" else "transformers"
    repo_id = (
        config.adapter_hub_repo_id if artifact == "adapter" else config.merged_hub_repo_id
    )
    title = (
        f"{config.run_name} (LoRA+ adapter)"
        if artifact == "adapter"
        else f"{config.run_name} (merged)"
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
        "> **This is an experiment, not a usable medical model.** The evaluation in this "
        "card shows a collapsed classifier that answers one label for every question. "
        "Do not use it for clinical decision-making, diagnosis, triage, or any other "
        "medical purpose.",
        "",
        "## What this is",
        "",
        f"A LoRA+ fine-tune of `{config.base_model_id}` on "
        f"`{config.dataset_id}` / `{config.dataset_config}`, trained as a "
        "single-GPU research exercise to check the training and evaluation pipeline "
        "end to end. It is published for reproducibility and as a record of a negative "
        "result, not as a model intended to be deployed.",
        "",
        "## Intended use",
        "",
        "- Inspecting what a small LoRA+ run on 900 PubMedQA examples does to a 7B "
        "instruction-tuned model.",
        "- Reproducing the evaluation and extending it: the interesting finding is the "
        "collapse, and the evaluation code now reports that collapse explicitly.",
        "",
        "## Out-of-scope use",
        "",
        "- Any clinical, diagnostic, triage, or treatment decision.",
        "- Any claim that this model answers biomedical questions correctly.",
        "- Any comparison against published PubMedQA results as if the reported accuracy "
        "were a real result.",
        "",
    ]

    lines.extend(_usage_section(artifact, config.base_model_id, repo_id))

    lines.extend(
        [
            "## Training setup",
            "",
            f"- Base model: `{config.base_model_id}`",
            f"- Dataset: `{config.dataset_id}` / `{config.dataset_config}`",
            f"- Artifact type: `{artifact}`",
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
                "Best eval loss",
                _format_metric(
                    summary.get("best_eval_loss", training.get("best_eval_loss"))
                ),
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
            "research questions with yes/no/maybe decisions. The pipeline splits it 90/10, "
            "so the reported evaluation uses 100 held-out examples. PubMedQA is released "
            "under the MIT licence; its abstracts are derived from PubMed and remain "
            "subject to the original publishers' terms.",
            "",
            "## Limitations",
            "",
            "- **The predictor collapsed.** All 100 held-out PubMedQA predictions were the "
            "same label, so the accuracy figure reflects the label distribution of the "
            "test split, not the model's ability. **The cause was not established**: the "
            "evaluation was not re-run, and the scoring defects since fixed in the "
            "evaluator are not shown to have produced it.",
            "- **The medical MMLU figure is one subject.** Only `anatomy` received samples; "
            "the other five medical subjects were allocated none.",
            "- **The metrics were computed once** on the adapter run and copied to both "
            "this card and the sibling card, so they do not describe this artifact "
            "separately.",
            "- **Small evaluation.** 100 PubMedQA examples and a single MMLU subject are "
            "far too little to characterise a 7B model, and the 90/10 split comes from "
            "PubMedQA's `pqa_labeled` training pool rather than the official test set.",
            "- **English only**, and only on PubMed-style abstracts and multiple-choice "
            "questions.",
            "- **Training-data bias carries over** from the base model and from PubMedQA's "
            "published-literature selection.",
            "",
            "## Licence",
            "",
            f"- This adapter/model is released under **Apache-2.0**, matching the "
            f"`{config.base_model_id}` base model, which is itself Apache-2.0.",
            "- Using or redistributing the base model is subject to the Mistral AI "
            "licence terms that accompany it, not to this repository alone.",
            "- PubMedQA is MIT-licensed; see its dataset card for the abstract "
            "provenance terms.",
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

    repo_id = (
        config.adapter_hub_repo_id if artifact == "adapter" else config.merged_hub_repo_id
    )
    if not repo_id:
        raise ValueError(f"No Hugging Face repo configured for {artifact} artifact")

    artifact_dir = Path(source_dir) if source_dir is not None else (
        config.adapter_output_dir if artifact == "adapter" else config.merged_output_dir
    )
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")

    if artifact == "adapter":
        prepare_adapter_for_publish(artifact_dir, config.base_model_id)

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
    )

    return {
        "artifact": artifact,
        "repo_id": repo_id,
        "source_dir": str(artifact_dir),
    }
