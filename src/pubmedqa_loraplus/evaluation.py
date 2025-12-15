from typing import Any

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from .config import LoRAPlusConfig
from .data import build_answer_text, build_instruction_prompt, flatten_context

PUBMEDQA_LABELS = ("yes", "no", "maybe")
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
    prompt: str,
    candidate: str,
    max_length: int,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    max_prompt_tokens = max(max_length - len(candidate_ids), 1)
    prompt_ids = prompt_ids[:max_prompt_tokens]
    input_ids = torch.tensor(
        [prompt_ids + candidate_ids],
        dtype=torch.long,
        device=_model_device(model),
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    prompt_offset = len(prompt_ids)
    score = 0.0

    for token_index, token_id in enumerate(candidate_ids):
        score += float(log_probs[prompt_offset + token_index, token_id].item())

    return score


def _predict_from_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: dict[str, str],
    max_length: int,
) -> str:
    if not candidates:
        raise ValueError("No candidate labels provided for prediction")
    scores = {
        label: _score_candidate(model, tokenizer, prompt, candidate, max_length)
        for label, candidate in candidates.items()
    }
    return max(scores, key=scores.get)


def evaluate_pubmedqa(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    config: LoRAPlusConfig,
) -> dict[str, Any]:
    predictions: list[str] = []
    references: list[str] = []
    eos_token = tokenizer.eos_token or "</s>"
    bos_token = tokenizer.bos_token or "<s>"
    label_candidates = {
        label: build_answer_text(label, eos_token) for label in PUBMEDQA_LABELS
    }

    model.eval()

    for example in tqdm(dataset, desc="Evaluating PubMedQA"):
        context = flatten_context(example["context"])
        prompt = build_instruction_prompt(example["question"], context, bos_token)
        prediction = _predict_from_candidates(
            model,
            tokenizer,
            prompt,
            label_candidates,
            config.max_length,
        )
        predictions.append(prediction)
        references.append(example["final_decision"].lower())

    cm = confusion_matrix(references, predictions, labels=list(PUBMEDQA_LABELS))
    return {
        "accuracy": accuracy_score(references, predictions),
        "macro_f1": f1_score(references, predictions, average="macro"),
        "weighted_f1": f1_score(references, predictions, average="weighted"),
        "confusion_matrix": cm.tolist(),
        "num_samples": len(predictions),
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
    medical_data = medical_data.select(
        range(min(config.mmlu_eval_samples, len(medical_data)))
    )

    subject_results = {
        subject: {"correct": 0, "total": 0} for subject in MMLU_MEDICAL_TASKS
    }

    model.eval()

    for example in tqdm(medical_data, desc="Evaluating MMLU medical"):
        prompt = "The following are multiple choice questions about medical knowledge.\n"
        prompt += f"Question: {example['question']}\n"
        for index, choice in enumerate(example["choices"]):
            prompt += f"{chr(65 + index)}. {choice}\n"
        prompt += "Answer:"
        prediction = _predict_from_candidates(
            model,
            tokenizer,
            prompt,
            {choice: f" {choice}" for choice in MMLU_CHOICES},
            config.max_length,
        )
        predicted_index = ord(prediction) - ord("A")
        subject = example["subject"]
        subject_results[subject]["total"] += 1
        if predicted_index == example["answer"]:
            subject_results[subject]["correct"] += 1

    total_correct = sum(result["correct"] for result in subject_results.values())
    total_samples = sum(result["total"] for result in subject_results.values())
    return {
        "overall_accuracy": total_correct / total_samples if total_samples else 0.0,
        "subject_results": subject_results,
        "total_samples": total_samples,
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
