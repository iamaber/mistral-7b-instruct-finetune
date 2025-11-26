import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import ExperimentConfig
from .data import flatten_context, load_pubmedqa
from .evaluation import evaluate_mmlu_medical, evaluate_pubmedqa


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PubMedQAExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset = None
        self.tokenized_dataset = None
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def set_seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def model_source(self) -> str:
        if self.config.base_model_path.exists():
            return str(self.config.base_model_path)
        return self.config.base_model_id

    def load_dataset(self) -> None:
        self.dataset = load_pubmedqa(self.config)
        LOGGER.info(
            "Dataset ready: train=%s test=%s",
            len(self.dataset["train"]),
            len(self.dataset["test"]),
        )

    def setup_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source())
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize_baseline_example(self, example: dict[str, Any]) -> dict[str, list[int]]:
        context = flatten_context(example["context"])
        prompt = (
            f"<s>[INST] Context: {context}\n\nQuestion: {example['question']} [/INST] "
            f"{example['final_decision']}</s>"
        )
        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
        )
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    def _tokenize_sft_example(self, example: dict[str, Any]) -> dict[str, list[int]]:
        context = flatten_context(example["context"])
        prompt = (
            f"<s>[INST] Context: {context}\n\nQuestion: {example['question']} [/INST]"
        )
        answer = f" {example['final_decision']}</s>"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        combined = self.tokenizer(
            prompt + answer,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
        )
        labels = list(combined["input_ids"])
        prompt_length = min(len(prompt_ids), self.config.max_length)
        for index in range(prompt_length):
            labels[index] = -100
        for index, mask_value in enumerate(combined["attention_mask"]):
            if mask_value == 0:
                labels[index] = -100
        combined["labels"] = labels
        return combined

    def preprocess(self) -> None:
        if self.config.training_mode == "baseline_full_prompt":
            tokenize_fn = self._tokenize_baseline_example
        elif self.config.training_mode == "sft_masked_answer":
            tokenize_fn = self._tokenize_sft_example
        else:
            raise ValueError(f"Unsupported training mode: {self.config.training_mode}")

        self.tokenized_dataset = self.dataset.map(
            tokenize_fn,
            remove_columns=self.dataset["train"].column_names,
        )

    def setup_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source(),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    @staticmethod
    def data_collator(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        keys = ("input_ids", "attention_mask", "labels")
        return {
            key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
            for key in keys
        }

    def setup_trainer(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        run_name = self.config.wandb_run_name or self.config.experiment_name
        if self.config.use_wandb:
            os.environ.setdefault("WANDB_PROJECT", self.config.wandb_project)
        wandb_enabled = self.config.use_wandb and (
            bool(os.getenv("WANDB_API_KEY"))
            or os.getenv("WANDB_MODE", "").lower() == "offline"
        )
        report_to = ["wandb"] if wandb_enabled else []

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            fp16=False,
            bf16=True,
            optim="adamw_torch",
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            label_names=["labels"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to=report_to,
            run_name=run_name if report_to else None,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            data_collator=self.data_collator,
        )

    def train(self) -> None:
        self.trainer.train()

    def save_outputs(self) -> None:
        self.config.adapter_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(self.config.adapter_output_dir))
        self.tokenizer.save_pretrained(str(self.config.tokenizer_output_dir))

    def evaluate(self) -> dict[str, Any]:
        pubmedqa_results = evaluate_pubmedqa(
            self.model, self.tokenizer, self.dataset["test"], self.config
        )
        mmlu_results = evaluate_mmlu_medical(self.model, self.tokenizer, self.config)
        return {
            "experiment_name": self.config.experiment_name,
            "training_mode": self.config.training_mode,
            "base_model_source": self.model_source(),
            "dataset_disk_path": str(self.config.dataset_disk_path),
            "model_paths": {
                "base_model": str(self.config.base_model_path),
                "adapter": str(self.config.adapter_output_dir),
                "tokenizer": str(self.config.tokenizer_output_dir),
                "checkpoints": str(self.config.output_dir),
            },
            "pubmedqa": pubmedqa_results,
            "mmlu_medical": mmlu_results,
        }

    def save_results(self, results: dict[str, Any]) -> Path:
        self.config.results_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.results_output_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        return self.config.results_output_path

    def run(self) -> dict[str, Any]:
        self.set_seed()
        self.load_dataset()
        self.setup_tokenizer()
        self.preprocess()
        self.setup_model()
        self.setup_trainer()
        self.train()
        self.save_outputs()
        results = self.evaluate()
        self.save_results(results)
        return results


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config = ExperimentConfig.from_file(config_path)
    experiment = PubMedQAExperiment(config)
    return experiment.run()
