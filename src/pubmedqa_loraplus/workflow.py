import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import LoRAPlusConfig
from .data import CompletionOnlyCollator, load_pubmedqa, tokenize_dataset
from .evaluation import run_evaluation_suite

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_dtype(config: LoRAPlusConfig) -> torch.dtype:
    if config.model_dtype == "float32":
        return torch.float32
    if config.model_dtype == "float16":
        return torch.float16
    if config.model_dtype == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _uses_bfloat16(config: LoRAPlusConfig) -> bool:
    return resolve_model_dtype(config) == torch.bfloat16


def _uses_float16(config: LoRAPlusConfig) -> bool:
    return resolve_model_dtype(config) == torch.float16


def _wandb_report_targets(config: LoRAPlusConfig) -> list[str]:
    if not config.use_wandb:
        return []
    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    if bool(os.getenv("WANDB_API_KEY")) or os.getenv("WANDB_MODE", "").lower() == "offline":
        return ["wandb"]
    return []


def load_tokenizer(config: LoRAPlusConfig) -> Any:
    tokenizer_source = (
        str(config.tokenizer_output_dir)
        if config.tokenizer_output_dir.exists()
        else config.base_model_source
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: LoRAPlusConfig) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_source,
        torch_dtype=resolve_model_dtype(config),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def apply_lora_plus_adapter(model: Any, config: LoRAPlusConfig) -> Any:
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


class LoRAPlusTrainer(Trainer):
    def __init__(self, *args: Any, loraplus_config: LoRAPlusConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loraplus_config = loraplus_config

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = create_loraplus_optimizer(
                self.model,
                torch.optim.AdamW,
                lr=self.args.learning_rate,
                loraplus_lr_ratio=self.loraplus_config.loraplus_lr_ratio,
                loraplus_lr_embedding=self.loraplus_config.loraplus_lr_embedding,
                loraplus_weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer


def build_training_arguments(config: LoRAPlusConfig) -> TrainingArguments:
    report_to = _wandb_report_targets(config)
    run_name = config.wandb_run_name or config.run_name
    return TrainingArguments(
        output_dir=str(config.checkpoints_dir),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        fp16=torch.cuda.is_available() and _uses_float16(config),
        bf16=torch.cuda.is_available() and _uses_bfloat16(config),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        group_by_length=config.group_by_length,
        max_grad_norm=config.max_grad_norm,
        report_to=report_to,
        run_name=run_name if report_to else None,
        save_safetensors=True,
    )


def _build_report(
    config: LoRAPlusConfig,
    dataset: Any,
    trainer: Trainer,
    train_result: Any,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    train_metrics = dict(train_result.metrics)
    return {
        "run_name": config.run_name,
        "base_model_id": config.base_model_id,
        "base_model_source": config.base_model_source,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "dataset_disk_path": str(config.dataset_disk_path),
        "train_examples": len(dataset["train"]),
        "eval_examples": len(dataset["test"]),
        "training": {
            "learning_rate": config.learning_rate,
            "loraplus_lr_ratio": config.loraplus_lr_ratio,
            "num_epochs": config.num_epochs,
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "global_step": trainer.state.global_step,
            "train_metrics": train_metrics,
        },
        "artifacts": {
            "checkpoints": str(config.checkpoints_dir),
            "adapter": str(config.adapter_output_dir),
            "tokenizer": str(config.tokenizer_output_dir),
            "merged": str(config.merged_output_dir),
            "report": str(config.report_path),
            "config_snapshot": str(config.config_snapshot_path),
            "training_log": str(config.training_log_path),
        },
        "evaluation": evaluation,
    }


def train_model(config: LoRAPlusConfig) -> dict[str, Any]:
    set_seed(config.seed)
    config.ensure_output_dirs()
    config.save_snapshot()

    dataset = load_pubmedqa(config)
    tokenizer = load_tokenizer(config)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config)
    model = apply_lora_plus_adapter(load_base_model(config), config)
    trainer = LoRAPlusTrainer(
        loraplus_config=config,
        model=model,
        args=build_training_arguments(config),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )

    LOGGER.info(
        "Starting LoRA+ training: train=%s eval=%s base=%s",
        len(dataset["train"]),
        len(dataset["test"]),
        config.base_model_id,
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_state()
    config.adapter_output_dir.mkdir(parents=True, exist_ok=True)
    config.tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.adapter_output_dir))
    tokenizer.save_pretrained(str(config.tokenizer_output_dir))

    evaluation = run_evaluation_suite(model, tokenizer, dataset["test"], config)
    report = _build_report(config, dataset, trainer, train_result, evaluation)
    _write_json(config.report_path, report)
    _write_json(config.training_log_path, trainer.state.log_history)
    return report


def load_model_for_artifact(
    config: LoRAPlusConfig,
    artifact: str,
    model_path: str | Path | None = None,
) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(config)

    if artifact == "merged":
        source_dir = Path(model_path) if model_path is not None else config.merged_output_dir
        model = AutoModelForCausalLM.from_pretrained(
            str(source_dir),
            torch_dtype=resolve_model_dtype(config),
            trust_remote_code=True,
        )
    elif artifact == "adapter":
        source_dir = Path(model_path) if model_path is not None else config.adapter_output_dir
        base_model = load_base_model(config)
        model = PeftModel.from_pretrained(base_model, str(source_dir))
    else:
        raise ValueError("artifact must be 'adapter' or 'merged'")

    model.config.use_cache = True
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def evaluate_artifact(
    config: LoRAPlusConfig,
    artifact: str,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset = load_pubmedqa(config)
    model, tokenizer = load_model_for_artifact(config, artifact, model_path=model_path)
    evaluation = run_evaluation_suite(model, tokenizer, dataset["test"], config)
    output_path = config.reports_root / f"{config.run_name}_{artifact}_eval.json"
    payload = {
        "run_name": config.run_name,
        "artifact": artifact,
        "model_path": str(model_path) if model_path is not None else None,
        "evaluation": evaluation,
    }
    _write_json(output_path, payload)
    payload["report_path"] = str(output_path)
    return payload


def export_artifacts(
    config: LoRAPlusConfig,
    checkpoint: str | Path | None = None,
) -> dict[str, str]:
    config.ensure_output_dirs()
    source_dir = Path(checkpoint) if checkpoint is not None else config.adapter_output_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {source_dir}")

    tokenizer = load_tokenizer(config)
    base_model = load_base_model(config)
    adapter_model = PeftModel.from_pretrained(base_model, str(source_dir))

    config.adapter_output_dir.mkdir(parents=True, exist_ok=True)
    config.tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    config.merged_output_dir.mkdir(parents=True, exist_ok=True)

    adapter_model.save_pretrained(str(config.adapter_output_dir))
    tokenizer.save_pretrained(str(config.tokenizer_output_dir))

    merged_model = adapter_model.merge_and_unload()
    merged_model.save_pretrained(
        str(config.merged_output_dir),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(config.merged_output_dir))

    return {
        "adapter": str(config.adapter_output_dir),
        "tokenizer": str(config.tokenizer_output_dir),
        "merged": str(config.merged_output_dir),
    }
