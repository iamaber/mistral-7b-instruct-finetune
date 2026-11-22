from typing import Any

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .config import LoRAPlusConfig
from .data import encode_prompt
from .prompts import ANSWER_LINE, ANSWER_MARKER, ChatFormat, choice_token_ids, extract_rationale
from .scoring import (
    PUBMEDQA_LABELS,
    accuracy,
    bootstrap_interval,
    choose_label,
    confusion_matrix_counts,
    is_collapsed,
    macro_f1,
    macro_f1_score,
    majority_class_rate,
    max_prediction_share,
    prediction_counts,
    reference_counts,
    select_subject_rows,
    subject_sample_plan,
    weighted_f1,
)

MMLU_CHOICES = ("A", "B", "C", "D")
MMLU_MEDICAL_TASKS = (
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "virology",
)
_ANSWER_RESERVE = 16


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _choice_log_probs(
    model: Any,
    input_ids: list[int],
    choice_ids: dict[str, int],
) -> dict[str, float]:
    """Log-probability of each choice's decision token at the next position."""
    tensor = torch.tensor([input_ids], dtype=torch.long, device=_model_device(model))
    with torch.no_grad():
        logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).logits
    log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
    return {choice: float(log_probs[token_id]) for choice, token_id in choice_ids.items()}


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> list[str]:
    """Greedy continuations of left-padded prompts, stopping at ``Answer:``."""
    pad_id = tokenizer.pad_token_id
    width = max(len(ids) for ids in prompts)
    input_ids = torch.tensor(
        [[pad_id] * (width - len(ids)) + ids for ids in prompts],
        dtype=torch.long,
        device=_model_device(model),
    )
    attention_mask = torch.tensor(
        [[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts],
        dtype=torch.long,
        device=input_ids.device,
    )
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stop_strings=[ANSWER_LINE],
            tokenizer=tokenizer,
            pad_token_id=pad_id,
        )
    return tokenizer.batch_decode(output[:, width:], skip_special_tokens=True)


class PubMedQAPredictor:
    """Turns one normalised PubMedQA example into a label decision.

    ``label`` runs scores the decision token right after the assistant prefix.
    ``rationale`` runs first generate the conclusion, then score the decision
    token after ``Answer:``, so a prediction always exists even when the model
    never writes the marker.
    """

    def __init__(self, model: Any, tokenizer: Any, config: LoRAPlusConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.chat_format = ChatFormat.from_tokenizer(tokenizer, config.chat_template_kwargs)
        reserve = _ANSWER_RESERVE
        if config.target_format == "rationale":
            reserve += config.rationale_max_new_tokens
        self.prompt_budget = config.max_length - reserve
        if self.prompt_budget < 1:
            raise ValueError("max_length is too small for the answer reserve")

    def _prompt_ids(self, example: dict[str, Any]) -> list[int]:
        return encode_prompt(
            self.tokenizer,
            self.chat_format,
            example["question"],
            example["context"],
            self.config.target_format,
            self.prompt_budget,
        )

    def _lead_text(self, rationale: str | None) -> str:
        if rationale is None:
            return self.chat_format.completion_prefix
        return f"{self.chat_format.completion_prefix}{rationale}\n{ANSWER_MARKER} "

    def predict_batch(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [self._prompt_ids(example) for example in examples]
        rationales: list[str | None] = [None] * len(examples)
        if self.config.target_format == "rationale":
            generated = _generate_batch(
                self.model, self.tokenizer, prompts, self.config.rationale_max_new_tokens
            )
            rationales = [extract_rationale(text) for text in generated]

        records = []
        for example, prompt_ids, rationale in zip(examples, prompts, rationales):
            lead_ids, choice_ids = choice_token_ids(
                self.tokenizer, self._lead_text(rationale), PUBMEDQA_LABELS
            )
            scores = _choice_log_probs(self.model, prompt_ids + lead_ids, choice_ids)
            label, tied = choose_label(scores, order=PUBMEDQA_LABELS)
            records.append(
                {
                    "pubid": example["pubid"],
                    "gold": example["final_decision"],
                    "prediction": label,
                    "tied": tied,
                    "log_probs": scores,
                    "rationale": rationale,
                }
            )
        return records


def summarise_pubmedqa(
    predictions: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    references = [record["gold"] for record in predictions]
    labels = [record["prediction"] for record in predictions]
    tied_predictions = sum(1 for record in predictions if record["tied"])
    matrix = confusion_matrix_counts(references, labels, PUBMEDQA_LABELS)
    collapsed = is_collapsed(labels)

    warnings: list[str] = []
    if collapsed:
        warnings.append(
            "All predictions are the same label; the classifier has collapsed and "
            "accuracy equals the majority-class rate."
        )
    if tied_predictions:
        warnings.append(
            f"{tied_predictions} prediction(s) were decided by a tie between candidates."
        )

    summary: dict[str, Any] = {
        "accuracy": accuracy(references, labels),
        "macro_f1": macro_f1(matrix, PUBMEDQA_LABELS),
        "weighted_f1": weighted_f1(matrix, PUBMEDQA_LABELS),
        "majority_class_rate": majority_class_rate(references),
        "max_prediction_share": max_prediction_share(labels),
        "collapsed": collapsed,
        "tied_predictions": tied_predictions,
        "label_order": list(PUBMEDQA_LABELS),
        "confusion_matrix": matrix,
        "reference_counts": reference_counts(references),
        "prediction_counts": prediction_counts(labels),
        "num_samples": len(labels),
        "warnings": warnings,
    }
    if bootstrap_samples and references:
        summary["confidence_intervals"] = {
            "level": 0.95,
            "samples": bootstrap_samples,
            "accuracy": bootstrap_interval(references, labels, accuracy, bootstrap_samples, seed),
            "macro_f1": bootstrap_interval(
                references, labels, macro_f1_score, bootstrap_samples, seed
            ),
        }
    return summary


def evaluate_pubmedqa(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    config: LoRAPlusConfig,
    bootstrap_samples: int | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(metrics, per-example predictions)`` for normalised examples."""
    was_training = model.training
    model.eval()
    predictor = PubMedQAPredictor(model, tokenizer, config)
    batch_size = max(1, config.eval_batch_size)
    predictions: list[dict[str, Any]] = []
    for start in tqdm(
        range(0, len(dataset), batch_size),
        desc="Evaluating PubMedQA",
        disable=not show_progress,
    ):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        predictions.extend(predictor.predict_batch(batch))
    if was_training:
        model.train()
    samples = config.bootstrap_samples if bootstrap_samples is None else bootstrap_samples
    return summarise_pubmedqa(predictions, samples, config.seed), predictions


def evaluate_mmlu_medical(
    model: Any,
    tokenizer: Any,
    config: LoRAPlusConfig,
) -> dict[str, Any]:
    mmlu_dataset = load_dataset(
        config.mmlu_dataset_id,
        config.mmlu_dataset_config,
        split=config.mmlu_split,
        revision=config.mmlu_revision,
    )
    medical_data = mmlu_dataset.filter(lambda row: row["subject"] in MMLU_MEDICAL_TASKS)

    subject_totals: dict[str, int] = {subject: 0 for subject in MMLU_MEDICAL_TASKS}
    for subject in medical_data["subject"]:
        if subject in subject_totals:
            subject_totals[subject] += 1

    plan = subject_sample_plan(subject_totals, config.mmlu_eval_samples, MMLU_MEDICAL_TASKS)
    selected_rows = select_subject_rows(list(medical_data["subject"]), plan)

    subject_results = {subject: {"correct": 0, "total": 0} for subject in MMLU_MEDICAL_TASKS}
    chat_format = ChatFormat.from_tokenizer(tokenizer, config.chat_template_kwargs)
    lead_ids, choice_ids = choice_token_ids(tokenizer, chat_format.completion_prefix, MMLU_CHOICES)
    model.eval()

    for row_index in tqdm(selected_rows, desc="Evaluating MMLU medical"):
        example = medical_data[row_index]
        options = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(MMLU_CHOICES, example["choices"])
        )
        user_text = (
            "The following is a multiple choice question about medical knowledge.\n\n"
            f"{example['question']}\n{options}\n\n"
            "Answer with the letter of the correct option only."
        )
        prompt = chat_format.render_prompt(tokenizer, user_text)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        scores = _choice_log_probs(model, prompt_ids + lead_ids, choice_ids)
        prediction, _ = choose_label(scores, order=MMLU_CHOICES)
        subject = example["subject"]
        subject_results[subject]["total"] += 1
        if MMLU_CHOICES.index(prediction) == example["answer"]:
            subject_results[subject]["correct"] += 1

    total_correct = sum(result["correct"] for result in subject_results.values())
    total_samples = sum(result["total"] for result in subject_results.values())
    subject_accuracies = {
        subject: (result["correct"] / result["total"] if result["total"] else None)
        for subject, result in subject_results.items()
    }

    warnings: list[str] = []
    uncovered = [
        subject
        for subject, count in plan.items()
        if count == 0 or subject_results[subject]["total"] == 0
    ]
    if uncovered:
        warnings.append("No MMLU samples were evaluated for: " + ", ".join(sorted(uncovered)))
    covered_subjects = [
        subject for subject in MMLU_MEDICAL_TASKS if subject_results[subject]["total"] > 0
    ]
    if len(covered_subjects) == 1:
        warnings.append(
            "The MMLU figure covers a single subject "
            f"({covered_subjects[0]}); it is not a medical-MMLU average."
        )

    return {
        "overall_accuracy": (total_correct / total_samples if total_samples else 0.0),
        "subject_accuracies": subject_accuracies,
        "subject_results": subject_results,
        "sample_plan": plan,
        "covered_subjects": covered_subjects,
        "total_samples": total_samples,
        "warnings": warnings,
    }


def run_evaluation_suite(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    config: LoRAPlusConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(evaluation, PubMedQA predictions)`` on the reported test split."""
    pubmedqa, predictions = evaluate_pubmedqa(model, tokenizer, dataset, config)
    evaluation = {
        "pubmedqa": pubmedqa,
        "mmlu_medical": evaluate_mmlu_medical(model, tokenizer, config),
    }
    return evaluation, predictions
