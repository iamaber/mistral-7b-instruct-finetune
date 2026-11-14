from typing import Any

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .config import LoRAPlusConfig
from .data import encode_prompt, flatten_context
from .prompts import build_answer_text
from .scoring import (
    PUBMEDQA_LABELS,
    accuracy,
    choose_label,
    confusion_matrix_counts,
    is_collapsed,
    macro_f1,
    majority_class_rate,
    normalise_candidate_scores,
    prediction_counts,
    reference_counts,
    score_candidate,
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


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _score_candidate(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    candidate: str,
) -> tuple[float, int]:
    """Return ``(summed log-prob, candidate token count)`` for one candidate."""
    candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    if not candidate_ids:
        raise ValueError("candidate produced no tokens; cannot score it")

    input_ids = torch.tensor(
        [prompt_ids + candidate_ids],
        dtype=torch.long,
        device=_model_device(model),
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    log_probs = torch.log_softmax(outputs.logits[0], dim=-1)

    def log_prob_at(position: int, token_id: int) -> float:
        return float(log_probs[position, token_id].item())

    return score_candidate(len(prompt_ids), candidate_ids, log_prob_at), len(candidate_ids)


def _predict_from_candidates(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    candidates: dict[str, str],
    order: tuple[str, ...] | list[str],
) -> tuple[str, bool]:
    if not candidates:
        raise ValueError("No candidate labels provided for prediction")

    raw_scores: dict[str, float] = {}
    token_counts: dict[str, int] = {}
    for label, candidate in candidates.items():
        raw_scores[label], token_counts[label] = _score_candidate(
            model,
            tokenizer,
            prompt_ids,
            candidate,
        )

    scores = normalise_candidate_scores(raw_scores, token_counts)
    return choose_label(scores, order=list(order))


def evaluate_pubmedqa(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    config: LoRAPlusConfig,
) -> dict[str, Any]:
    predictions: list[str] = []
    references: list[str] = []
    tied_predictions = 0
    eos_token = tokenizer.eos_token or "</s>"
    label_candidates = {
        label: build_answer_text(label, eos_token) for label in PUBMEDQA_LABELS
    }

    model.eval()

    for example in tqdm(dataset, desc="Evaluating PubMedQA"):
        context = flatten_context(example["context"])
        prompt_ids = encode_prompt(
            tokenizer,
            example["question"],
            context,
            config.max_length,
        )
        prediction, tied = _predict_from_candidates(
            model,
            tokenizer,
            prompt_ids,
            label_candidates,
            order=PUBMEDQA_LABELS,
        )
        tied_predictions += int(tied)
        predictions.append(prediction)
        references.append(example["final_decision"].lower())

    matrix = confusion_matrix_counts(references, predictions, PUBMEDQA_LABELS)
    reference_distribution = reference_counts(references)
    prediction_distribution = prediction_counts(predictions)
    collapsed = is_collapsed(predictions)
    baseline = majority_class_rate(references)

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

    return {
        "accuracy": accuracy(references, predictions),
        "macro_f1": macro_f1(matrix, PUBMEDQA_LABELS),
        "weighted_f1": weighted_f1(matrix, PUBMEDQA_LABELS),
        "majority_class_rate": baseline,
        "collapsed": collapsed,
        "tied_predictions": tied_predictions,
        "label_order": list(PUBMEDQA_LABELS),
        "confusion_matrix": matrix,
        "reference_counts": reference_distribution,
        "prediction_counts": prediction_distribution,
        "num_samples": len(predictions),
        "warnings": warnings,
    }


def evaluate_mmlu_medical(
    model: Any,
    tokenizer: Any,
    config: LoRAPlusConfig,
) -> dict[str, Any]:
    mmlu_dataset = load_dataset(
        config.mmlu_dataset_id,
        config.mmlu_dataset_config,
        split=config.mmlu_split,
    )
    medical_data = mmlu_dataset.filter(lambda row: row["subject"] in MMLU_MEDICAL_TASKS)

    subject_totals: dict[str, int] = {subject: 0 for subject in MMLU_MEDICAL_TASKS}
    for subject in medical_data["subject"]:
        if subject in subject_totals:
            subject_totals[subject] += 1

    plan = subject_sample_plan(
        subject_totals,
        config.mmlu_eval_samples,
        MMLU_MEDICAL_TASKS,
    )
    selected_rows = select_subject_rows(list(medical_data["subject"]), plan)

    subject_results = {
        subject: {"correct": 0, "total": 0} for subject in MMLU_MEDICAL_TASKS
    }
    model.eval()

    for row_index in tqdm(selected_rows, desc="Evaluating MMLU medical"):
        example = medical_data[row_index]
        prompt = "The following are multiple choice questions about medical knowledge.\n"
        prompt += f"Question: {example['question']}\n"
        for index, choice in enumerate(example["choices"]):
            prompt += f"{chr(65 + index)}. {choice}\n"
        prompt += "Answer:"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-config.max_length:]
        prediction, _ = _predict_from_candidates(
            model,
            tokenizer,
            prompt_ids,
            {choice: f" {choice}" for choice in MMLU_CHOICES},
            order=MMLU_CHOICES,
        )
        predicted_index = ord(prediction) - ord("A")
        subject = example["subject"]
        subject_results[subject]["total"] += 1
        if predicted_index == example["answer"]:
            subject_results[subject]["correct"] += 1

    total_correct = sum(result["correct"] for result in subject_results.values())
    total_samples = sum(result["total"] for result in subject_results.values())
    subject_accuracies = {
        subject: (
            result["correct"] / result["total"] if result["total"] else None
        )
        for subject, result in subject_results.items()
    }

    warnings: list[str] = []
    uncovered = [
        subject
        for subject, count in plan.items()
        if count == 0 or subject_results[subject]["total"] == 0
    ]
    if uncovered:
        warnings.append(
            "No MMLU samples were evaluated for: " + ", ".join(sorted(uncovered))
        )
    covered_subjects = [
        subject for subject in MMLU_MEDICAL_TASKS if subject_results[subject]["total"] > 0
    ]
    if len(covered_subjects) == 1:
        warnings.append(
            "The MMLU figure covers a single subject "
            f"({covered_subjects[0]}); it is not a medical-MMLU average."
        )

    return {
        "overall_accuracy": (
            total_correct / total_samples if total_samples else 0.0
        ),
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
) -> dict[str, Any]:
    return {
        "pubmedqa": evaluate_pubmedqa(model, tokenizer, dataset, config),
        "mmlu_medical": evaluate_mmlu_medical(model, tokenizer, config),
    }
