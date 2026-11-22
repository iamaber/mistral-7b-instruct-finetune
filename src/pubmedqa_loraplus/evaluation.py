from typing import Any

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .config import LoRAPlusConfig
from .data import encode_prompt
from .prompts import (
    ANSWER_LINE,
    ANSWER_MARKER,
    MMLU_ANSWER_LINE_INSTRUCTION,
    MMLU_FIRST_TOKEN_INSTRUCTION,
    ChatFormat,
    choice_token_ids,
    extract_rationale,
)
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


def _left_padded(model: Any, sequences: list[list[int]], pad_id: int) -> dict[str, torch.Tensor]:
    """Left-pad so every sequence ends at the last column."""
    width = max(len(ids) for ids in sequences)
    device = _model_device(model)
    input_ids = torch.tensor(
        [[pad_id] * (width - len(ids)) + ids for ids in sequences], dtype=torch.long, device=device
    )
    attention_mask = torch.tensor(
        [[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences],
        dtype=torch.long,
        device=device,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _next_token_log_probs(
    model: Any,
    sequences: list[list[int]],
    pad_id: int,
) -> torch.Tensor:
    """``(batch, vocab)`` log-probabilities of the token after each sequence, in one forward."""
    batch = _left_padded(model, sequences, pad_id)
    # Forward (unlike generate) does not derive positions from the mask, so pass them
    # explicitly; otherwise padding would shift every shorter sequence's positions.
    position_ids = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = model(**batch, position_ids=position_ids, logits_to_keep=1).logits
    return torch.log_softmax(logits[:, -1].float(), dim=-1)


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> list[str]:
    """Greedy continuations of left-padded prompts, stopping at the answer line."""
    batch = _left_padded(model, prompts, tokenizer.pad_token_id)
    with torch.no_grad():
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stop_strings=[ANSWER_LINE],
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
        )
    width = batch["input_ids"].shape[1]
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

        sequences = []
        choices = []
        for prompt_ids, rationale in zip(prompts, rationales):
            lead_ids, choice_ids = choice_token_ids(
                self.tokenizer, self._lead_text(rationale), PUBMEDQA_LABELS
            )
            sequences.append(prompt_ids + lead_ids)
            choices.append(choice_ids)
        log_probs = _next_token_log_probs(self.model, sequences, self.tokenizer.pad_token_id)

        records = []
        for row, (example, rationale, choice_ids) in enumerate(zip(examples, rationales, choices)):
            scores = {label: float(log_probs[row, token]) for label, token in choice_ids.items()}
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


def _score_mmlu_batch(
    model: Any,
    sequences: list[list[int]],
    choices: list[dict[str, int]],
    pad_id: int,
) -> list[str]:
    log_probs = _next_token_log_probs(model, sequences, pad_id)
    predictions = []
    for row, choice_ids in enumerate(choices):
        scores = {choice: float(log_probs[row, token]) for choice, token in choice_ids.items()}
        prediction, _ = choose_label(scores, order=MMLU_CHOICES)
        predictions.append(prediction)
    return predictions


def evaluate_mmlu_medical(
    model: Any,
    tokenizer: Any,
    config: LoRAPlusConfig,
) -> dict[str, Any]:
    """Score the same medical questions at the reply start and after a rationale."""
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

    subject_results = {
        subject: {"correct": 0, "total": 0, "answer_line": 0} for subject in MMLU_MEDICAL_TASKS
    }
    answer_line_unmarked = 0
    chat_format = ChatFormat.from_tokenizer(tokenizer, config.chat_template_kwargs)
    lead_ids, choice_ids = choice_token_ids(tokenizer, chat_format.completion_prefix, MMLU_CHOICES)
    model.eval()

    batch_size = max(1, config.eval_batch_size)
    for start in tqdm(range(0, len(selected_rows), batch_size), desc="Evaluating MMLU medical"):
        examples = [medical_data[index] for index in selected_rows[start : start + batch_size]]
        first_token_sequences = []
        rationale_prompts = []
        for example in examples:
            options = "\n".join(
                f"{letter}. {choice}" for letter, choice in zip(MMLU_CHOICES, example["choices"])
            )
            question = (
                "The following is a multiple choice question about medical knowledge.\n\n"
                f"{example['question']}\n{options}\n\n"
            )
            first_token_prompt = chat_format.render_prompt(
                tokenizer, question + MMLU_FIRST_TOKEN_INSTRUCTION
            )
            first_token_ids = tokenizer(first_token_prompt, add_special_tokens=False)["input_ids"]
            first_token_sequences.append(first_token_ids + lead_ids)
            rationale_prompt = chat_format.render_prompt(
                tokenizer, question + MMLU_ANSWER_LINE_INSTRUCTION
            )
            rationale_ids = tokenizer(rationale_prompt, add_special_tokens=False)["input_ids"]
            rationale_prompts.append(rationale_ids)

        first_token_predictions = _score_mmlu_batch(
            model, first_token_sequences, [choice_ids] * len(examples), tokenizer.pad_token_id
        )
        generated = _generate_batch(
            model, tokenizer, rationale_prompts, config.rationale_max_new_tokens
        )
        answer_line_sequences = []
        answer_line_choices = []
        for prompt_ids, text in zip(rationale_prompts, generated):
            answer_line_unmarked += ANSWER_LINE not in text
            rationale = extract_rationale(text)
            lead_text = f"{chat_format.completion_prefix}{rationale}{ANSWER_LINE} "
            rationale_lead_ids, rationale_choice_ids = choice_token_ids(
                tokenizer, lead_text, MMLU_CHOICES
            )
            answer_line_sequences.append(prompt_ids + rationale_lead_ids)
            answer_line_choices.append(rationale_choice_ids)
        answer_line_predictions = _score_mmlu_batch(
            model, answer_line_sequences, answer_line_choices, tokenizer.pad_token_id
        )

        for example, first_token, answer_line in zip(
            examples, first_token_predictions, answer_line_predictions
        ):
            result = subject_results[example["subject"]]
            gold = MMLU_CHOICES[example["answer"]]
            result["total"] += 1
            result["correct"] += first_token == gold
            result["answer_line"] += answer_line == gold

    total_correct = sum(result["correct"] for result in subject_results.values())
    answer_line_correct = sum(result["answer_line"] for result in subject_results.values())
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
        "answer_line_accuracy": (answer_line_correct / total_samples if total_samples else 0.0),
        "answer_line_unmarked": answer_line_unmarked,
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
