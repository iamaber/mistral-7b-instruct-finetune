from typing import Any

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from .config import ExperimentConfig
from .data import flatten_context


def evaluate_pubmedqa(
    model: Any, tokenizer: Any, dataset: Dataset, config: ExperimentConfig
) -> dict[str, Any]:
    predictions: list[str] = []
    references: list[str] = []

    model.eval()

    for example in tqdm(dataset, desc="Evaluating PubMedQA"):
        context = flatten_context(example["context"])
        prompt = (
            f"<s>[INST] Context: {context}\n\nQuestion: {example['question']} [/INST]"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("[/INST]")[-1].strip().lower()

        if "yes" in answer:
            prediction = "yes"
        elif "no" in answer:
            prediction = "no"
        elif "maybe" in answer:
            prediction = "maybe"
        else:
            prediction = "maybe"

        predictions.append(prediction)
        references.append(example["final_decision"].lower())

    cm = confusion_matrix(references, predictions, labels=["yes", "no", "maybe"])
    return {
        "accuracy": accuracy_score(references, predictions),
        "macro_f1": f1_score(references, predictions, average="macro"),
        "weighted_f1": f1_score(references, predictions, average="weighted"),
        "confusion_matrix": cm.tolist(),
        "num_samples": len(predictions),
    }


def evaluate_mmlu_medical(
    model: Any, tokenizer: Any, config: ExperimentConfig
) -> dict[str, Any]:
    mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
    medical_tasks = [
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "medical_genetics",
        "professional_medicine",
        "virology",
    ]
    medical_data = mmlu_dataset.filter(lambda row: row["subject"] in medical_tasks)
    medical_data = medical_data.select(range(min(config.mmlu_eval_samples, len(medical_data))))

    subject_results = {
        subject: {"correct": 0, "total": 0} for subject in medical_tasks
    }

    model.eval()

    for example in tqdm(medical_data, desc="Evaluating MMLU medical"):
        prompt = "The following are multiple choice questions about medical knowledge.\n"
        prompt += f"Question: {example['question']}\n"
        for index, choice in enumerate(example["choices"]):
            prompt += f"{chr(65 + index)}. {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("Answer:")[-1].strip().upper()
        predicted_index = ord(answer[0]) - ord("A") if answer and answer[0] in "ABCD" else 0

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
