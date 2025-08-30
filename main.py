import os
import json
import logging
import random
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MistralPubMedQAFineTuner:
    """Fine-tune Mistral-7B on PubMedQA dataset with LoRA"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self.tokenized_dataset = None

        # Set seeds for reproducibility
        self.set_seed(config.get("seed", 42))

    def set_seed(self, seed: int = 42):
        """Set all seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {seed}")

    def load_dataset(self):
        """Load and split PubMedQA dataset"""
        logger.info("Loading PubMedQA dataset...")
        try:
            pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
            train_test_split = pubmedqa["train"].train_test_split(
                test_size=self.config.get("test_size", 0.1),
                seed=self.config.get("seed", 42),
            )
            self.dataset = train_test_split
            logger.info(
                f"Dataset loaded - Train: {len(self.dataset['train'])}, Test: {len(self.dataset['test'])}"
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def setup_tokenizer(self):
        """Setup tokenizer"""
        logger.info("Setting up tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer setup complete")
        except Exception as e:
            logger.error(f"Error setting up tokenizer: {e}")
            raise

    def format_prompt(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format example into Mistral instruction format"""
        context = example["context"]
        if isinstance(context, dict) and "contexts" in context:
            context = " ".join(context["contexts"])
        elif isinstance(context, list):
            context = " ".join(context)

        prompt = f"<s>[INST] Context: {context}\n\nQuestion: {example['question']} [/INST] {example['final_decision']}</s>"
        return {"text": prompt}

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Tokenize examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.get("max_length", 512),
            padding="max_length",
        )

    def preprocess_data(self):
        """Preprocess and tokenize dataset"""
        logger.info("Preprocessing data...")
        try:
            # Apply formatting
            self.dataset = self.dataset.map(self.format_prompt)

            # Tokenize
            self.tokenized_dataset = self.dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=self.dataset["train"].column_names,
            )
            logger.info("Data preprocessing complete")
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def setup_model(self):
        """Setup model with LoRA"""
        logger.info("Setting up model...")
        try:
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.get("lora_r", 16),
                lora_alpha=self.config.get("lora_alpha", 32),
                lora_dropout=self.config.get("lora_dropout", 0.05),
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

            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info("Model setup complete")

        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise

    def custom_data_collator(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Custom data collator for causal language modeling"""
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = batch["input_ids"].clone()
        return batch

    def setup_trainer(self):
        """Setup trainer"""
        logger.info("Setting up trainer...")
        try:
            training_args = TrainingArguments(
                output_dir=self.config.get("output_dir", "./mistral-pubmedqa"),
                per_device_train_batch_size=self.config.get("train_batch_size", 4),
                per_device_eval_batch_size=self.config.get("eval_batch_size", 4),
                gradient_accumulation_steps=self.config.get(
                    "gradient_accumulation_steps", 4
                ),
                learning_rate=self.config.get("learning_rate", 2e-4),
                num_train_epochs=self.config.get("num_epochs", 3),
                logging_steps=self.config.get("logging_steps", 10),
                save_steps=self.config.get("save_steps", 500),
                eval_strategy="steps",
                eval_steps=self.config.get("eval_steps", 500),
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
            )

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["test"],
                data_collator=self.custom_data_collator,
            )
            logger.info("Trainer setup complete")

        except Exception as e:
            logger.error(f"Error setting up trainer: {e}")
            raise

    def train(self):
        """Execute training"""
        logger.info("Starting training...")
        try:
            self.trainer.train()
            logger.info("Training complete")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def save_model(self):
        """Save the fine-tuned model and tokenizer"""
        logger.info("Saving model...")
        try:
            # Save the full model
            model_path = self.config.get(
                "model_save_path", "./mistral-pubmedqa-finetune"
            )
            self.trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)

            # Save LoRA adapter separately
            adapter_path = self.config.get(
                "adapter_save_path", "./mistral-pubmedqa-adapter"
            )
            self.model.save_pretrained(adapter_path)

            logger.info(f"Model saved to {model_path}")
            logger.info(f"Adapter saved to {adapter_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def evaluate_pubmedqa(self, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate model on PubMedQA dataset"""
        logger.info("Evaluating on PubMedQA...")
        predictions = []
        references = []

        self.model.eval()

        for example in tqdm(dataset, desc="Evaluating PubMedQA"):
            # Format prompt for inference
            context = example["context"]
            if isinstance(context, dict) and "contexts" in context:
                context = " ".join(context["contexts"])
            elif isinstance(context, list):
                context = " ".join(context)

            prompt = f"<s>[INST] Context: {context}\n\nQuestion: {example['question']} [/INST]"

            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.split("[/INST]")[-1].strip().lower()

            # Normalize answer
            if "yes" in answer:
                pred = "yes"
            elif "no" in answer:
                pred = "no"
            elif "maybe" in answer:
                pred = "maybe"
            else:
                pred = "maybe"

            predictions.append(pred)
            references.append(example["final_decision"].lower())

        # Calculate metrics
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average="weighted")
        cm = confusion_matrix(references, predictions, labels=["yes", "no", "maybe"])

        results = {
            "accuracy": accuracy,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(predictions),
        }

        logger.info(f"PubMedQA Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results

    def evaluate_mmlu_medical(self) -> Dict[str, Any]:
        """Evaluate model on medical MMLU tasks"""
        logger.info("Evaluating on MMLU medical tasks...")
        try:
            mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")

            medical_tasks = [
                "anatomy",
                "clinical_knowledge",
                "college_medicine",
                "medical_genetics",
                "professional_medicine",
                "virology",
            ]

            medical_data = mmlu_dataset.filter(lambda x: x["subject"] in medical_tasks)
            # Limit for faster evaluation
            medical_data = medical_data.select(range(min(50, len(medical_data))))

            subject_results = {
                subject: {"correct": 0, "total": 0} for subject in medical_tasks
            }

            self.model.eval()

            for example in tqdm(medical_data, desc="Evaluating MMLU"):
                prompt = f"The following are multiple choice questions about medical knowledge.\n\n"
                prompt += f"Question: {example['question']}\n"
                for i, choice in enumerate(example["choices"]):
                    prompt += f"{chr(65 + i)}. {choice}\n"
                prompt += "Answer:"

                inputs = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device
                )

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                answer = generated_text.split("Answer:")[-1].strip().upper()

                if answer and answer[0] in ["A", "B", "C", "D"]:
                    predicted_idx = ord(answer[0]) - ord("A")
                else:
                    predicted_idx = 0

                subject = example["subject"]
                subject_results[subject]["total"] += 1
                if predicted_idx == example["answer"]:
                    subject_results[subject]["correct"] += 1

            # Calculate results
            total_correct = sum([v["correct"] for v in subject_results.values()])
            total_total = sum([v["total"] for v in subject_results.values()])
            overall_accuracy = total_correct / total_total if total_total > 0 else 0

            results = {
                "overall_accuracy": overall_accuracy,
                "subject_results": subject_results,
                "total_samples": total_total,
            }

            logger.info(f"MMLU Medical Accuracy: {overall_accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error in MMLU evaluation: {e}")
            return {"error": str(e)}

    def run_full_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        logger.info("Starting full fine-tuning pipeline...")

        # Setup
        self.load_dataset()
        self.setup_tokenizer()
        self.preprocess_data()
        self.setup_model()
        self.setup_trainer()

        # Train
        self.train()

        # Save
        self.save_model()

        # Evaluate
        pubmedqa_results = self.evaluate_pubmedqa(self.dataset["test"])
        mmlu_results = self.evaluate_mmlu_medical()

        # Save results
        evaluation_results = {
            "pubmedqa": pubmedqa_results,
            "mmlu_medical": mmlu_results,
            "config": self.config,
        }

        results_path = "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info(f"Evaluation results saved to {results_path}")
        logger.info("Pipeline complete!")

        return evaluation_results


def main():
    """Main execution function"""

    # Configuration
    config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "output_dir": "./mistral-pubmedqa",
        "model_save_path": "./mistral-pubmedqa-finetune",
        "adapter_save_path": "./mistral-pubmedqa-adapter",
        "seed": 42,
        "test_size": 0.1,
        "max_length": 512,
        # LoRA config
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        # Training config
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
    }

    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        logger.warning("No GPU available, using CPU")

    # Initialize and run fine-tuning
    fine_tuner = MistralPubMedQAFineTuner(config)
    results = fine_tuner.run_full_pipeline()

    print("\n=== Final Results ===")
    print(f"PubMedQA Accuracy: {results['pubmedqa']['accuracy']:.4f}")
    print(f"PubMedQA F1 Score: {results['pubmedqa']['f1']:.4f}")
    print(f"MMLU Medical Accuracy: {results['mmlu_medical']['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
