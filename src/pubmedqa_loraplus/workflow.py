import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import peft
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .artifact import prepare_adapter_for_publish
from .config import LoRAPlusConfig, baseline_cache_key, planned_optimizer_steps
from .data import CompletionOnlyCollator, build_train_dataset, load_pubmedqa, tokenize_dataset
from .evaluation import evaluate_pubmedqa, run_evaluation_suite
from .prompts import ChatFormat
from .report import build_run_report
from .scoring import accuracy, macro_f1_score, paired_bootstrap_delta
from .splits import OFFICIAL_TEST_SPLIT

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_source, revision=_hub_revision(config)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _hub_revision(config: LoRAPlusConfig) -> str | None:
    """The pinned revision applies to Hub loads, not to a local snapshot directory."""
    return None if config.base_model_path.exists() else config.base_model_revision


def load_base_model(config: LoRAPlusConfig) -> Any:
    # The causal LM auto class can select a VLM checkpoint's text-only model
    # when Transformers registers that architecture for causal LM loading.
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_source,
        dtype=resolve_model_dtype(config),
        revision=_hub_revision(config),
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
    """``Trainer`` with the LoRA+ optimizer and classifier metrics at every evaluation.

    ``evaluate`` adds ``eval_accuracy``/``eval_macro_f1`` from the same predictor
    the test report uses, so ``metric_for_best_model="macro_f1"`` selects the
    checkpoint on decisions instead of on token loss.
    """

    def __init__(
        self,
        *args: Any,
        loraplus_config: LoRAPlusConfig,
        selection_examples: Dataset,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loraplus_config = loraplus_config
        self.selection_examples = selection_examples

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

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        summary, _ = evaluate_pubmedqa(
            self.model,
            self.processing_class,
            self.selection_examples,
            self.loraplus_config,
            bootstrap_samples=0,
            show_progress=False,
        )
        selection = {
            f"{metric_key_prefix}_{key}": float(summary[key])
            for key in ("accuracy", "macro_f1", "max_prediction_share")
        }
        self.log(selection)
        metrics.update(selection)
        return metrics


def build_training_arguments(config: LoRAPlusConfig) -> TrainingArguments:
    report_to = _wandb_report_targets(config)
    run_name = config.wandb_run_name or config.run_name
    return TrainingArguments(
        output_dir=str(config.checkpoints_dir),
        seed=config.seed,
        data_seed=config.seed,
        per_device_train_batch_size=config.train_batch_size,
        # eval_batch_size sizes the generation predictor; the token-loss pass holds
        # full-vocabulary logits, so it uses the training micro-batch and keeps none.
        per_device_eval_batch_size=config.train_batch_size,
        prediction_loss_only=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        fp16=torch.cuda.is_available() and _uses_float16(config),
        bf16=torch.cuda.is_available() and _uses_bfloat16(config),
        activation_checkpointing=True,
        # Reentrant checkpointing drops the graph when the frozen embeddings output
        # no grad, so the LoRA weights would never receive a gradient.
        activation_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        # A float below 1 is a fraction of total steps (Transformers v5 has no warmup_ratio).
        warmup_steps=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        train_sampling_strategy="group_by_length" if config.group_by_length else "random",
        max_grad_norm=config.max_grad_norm,
        report_to=report_to,
        run_name=run_name if report_to else None,
    )


def _latest_logged_metric(log_history: list[dict[str, Any]], key: str) -> float | None:
    for entry in reversed(log_history):
        if key in entry:
            return float(entry[key])
    return None


def build_training_summary(trainer: Trainer, train_result: Any) -> dict[str, Any]:
    best_metric = trainer.state.best_metric
    metrics = dict(train_result.metrics)
    history = trainer.state.log_history
    return {
        "selection_metric": "eval_macro_f1",
        "best_validation_macro_f1": float(best_metric) if best_metric is not None else None,
        "latest_validation_macro_f1": _latest_logged_metric(history, "eval_macro_f1"),
        "latest_train_loss": _latest_logged_metric(history, "loss"),
        "latest_eval_loss": _latest_logged_metric(history, "eval_loss"),
        "train_runtime": metrics.get("train_runtime"),
        "train_loss": metrics.get("train_loss"),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def compare_to_baseline(
    predictions: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    config: LoRAPlusConfig,
) -> dict[str, Any]:
    """Paired bootstrap deltas of the fine-tune over the base model on the same questions."""
    if [r["pubid"] for r in predictions] != [r["pubid"] for r in baseline_predictions]:
        raise ValueError("Baseline and fine-tuned predictions cover different questions")
    references = [record["gold"] for record in predictions]
    tuned = [record["prediction"] for record in predictions]
    base = [record["prediction"] for record in baseline_predictions]
    if config.bootstrap_samples < 1:
        return {
            name: {"delta": metric(references, tuned) - metric(references, base)}
            for name, metric in (("accuracy", accuracy), ("macro_f1", macro_f1_score))
        }
    samples = config.bootstrap_samples
    return {
        name: paired_bootstrap_delta(references, tuned, base, metric, samples, config.seed)
        for name, metric in (("accuracy", accuracy), ("macro_f1", macro_f1_score))
    }


def _dataset_fields(config: LoRAPlusConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "base_model_id": config.base_model_id,
        "base_model_source": config.base_model_source,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "dataset_disk_path": str(config.dataset_disk_path),
        "test_split": OFFICIAL_TEST_SPLIT,
    }


def train_model(config: LoRAPlusConfig) -> dict[str, Any]:
    set_seed(config.seed)
    config.ensure_output_dirs()
    config.save_snapshot()

    dataset = load_pubmedqa(config)
    train_examples = build_train_dataset(dataset, config)
    planned_steps = planned_optimizer_steps(
        config, len(train_examples), world_size=int(os.getenv("WORLD_SIZE", "1"))
    )
    tokenizer = load_tokenizer(config)
    chat_format = ChatFormat.from_tokenizer(tokenizer, config.chat_template_kwargs)
    model = apply_lora_plus_adapter(load_base_model(config), config)
    trainer = LoRAPlusTrainer(
        loraplus_config=config,
        selection_examples=dataset["validation"],
        model=model,
        args=build_training_arguments(config),
        train_dataset=tokenize_dataset(train_examples, tokenizer, chat_format, config),
        eval_dataset=tokenize_dataset(dataset["validation"], tokenizer, chat_format, config),
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    baseline = None
    if config.evaluate_base_model:

        def evaluate_base() -> tuple[dict[str, Any], list[dict[str, Any]]]:
            with trainer.model.disable_adapter():
                return run_evaluation_suite(trainer.model, tokenizer, dataset["test"], config)

        baseline, baseline_predictions = cached_baseline(config, evaluate_base)
        _write_jsonl(config.baseline_predictions_path, baseline_predictions)

    LOGGER.info(
        "Starting LoRA+ training: optimizer_examples=%s labeled_train=%s validation=%s "
        "test=%s planned_steps=%s base=%s",
        len(train_examples),
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
        planned_steps,
        config.base_model_id,
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_state()
    config.adapter_output_dir.mkdir(parents=True, exist_ok=True)
    config.tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.adapter_output_dir))
    tokenizer.save_pretrained(str(config.tokenizer_output_dir))
    prepare_adapter_for_publish(config.adapter_output_dir, config.base_model_id)

    evaluation, predictions = run_evaluation_suite(
        trainer.model, tokenizer, dataset["test"], config
    )
    _write_jsonl(config.predictions_path, predictions)
    if baseline is not None:
        evaluation["baseline"] = baseline
        evaluation["delta_vs_baseline"] = compare_to_baseline(
            predictions, baseline_predictions, config
        )

    report = build_run_report(
        **_dataset_fields(config),
        train_examples=len(train_examples),
        labeled_train_examples=len(dataset["train"]),
        validation_examples=len(dataset["validation"]),
        eval_examples=len(dataset["test"]),
        training={
            "target_format": config.target_format,
            "learning_rate": config.learning_rate,
            "loraplus_lr_ratio": config.loraplus_lr_ratio,
            "num_epochs": config.num_epochs,
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "balance_train_classes": config.balance_train_classes,
            "pqa_artificial_samples": config.pqa_artificial_samples,
            "planned_steps": planned_steps,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "global_step": trainer.state.global_step,
            "train_metrics": dict(train_result.metrics),
        },
        training_summary=build_training_summary(trainer, train_result),
        artifacts={
            "checkpoints": str(config.checkpoints_dir),
            "adapter": str(config.adapter_output_dir),
            "tokenizer": str(config.tokenizer_output_dir),
            "merged": str(config.merged_output_dir),
            "report": str(config.report_path),
            "predictions": str(config.predictions_path),
            "config_snapshot": str(config.config_snapshot_path),
            "training_log": str(config.training_log_path),
        },
        evaluation=evaluation,
    )
    _write_json(config.report_path, report)
    _write_json(config.training_log_path, trainer.state.log_history)
    return report


ARTIFACTS = ("base", "adapter", "merged")


def load_model_for_artifact(
    config: LoRAPlusConfig,
    artifact: str,
    model_path: str | Path | None = None,
) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(config)

    if artifact == "merged":
        source_dir = Path(model_path) if model_path is not None else config.merged_output_dir
        model = AutoModelForCausalLM.from_pretrained(
            str(source_dir), dtype=resolve_model_dtype(config)
        )
    elif artifact == "adapter":
        source_dir = Path(model_path) if model_path is not None else config.adapter_output_dir
        model = PeftModel.from_pretrained(load_base_model(config), str(source_dir))
    elif artifact == "base":
        model = load_base_model(config)
    else:
        raise ValueError(f"artifact must be one of {ARTIFACTS}")

    model.config.use_cache = True
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def cached_baseline(
    config: LoRAPlusConfig,
    compute: Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse the zero-shot baseline for identical base-model settings, else compute it.

    The baseline ignores training hyperparameters, so sweep variants and reruns
    share one evaluation instead of re-scoring the base model every time.
    """
    # Library upgrades change kernels and tokenization, so the versions are part of the key.
    versions = f"transformers-{transformers.__version__}_peft-{peft.__version__}"
    path = config.reports_root / "baselines" / versions / f"{baseline_cache_key(config)}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        LOGGER.info("Reusing cached base-model baseline %s", path)
        return payload["evaluation"], payload["predictions"]
    evaluation, predictions = compute()
    _write_json(path, {"evaluation": evaluation, "predictions": predictions})
    return evaluation, predictions


def evaluate_artifact(
    config: LoRAPlusConfig,
    artifact: str,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset = load_pubmedqa(config)

    def evaluate_model() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        model, tokenizer = load_model_for_artifact(config, artifact, model_path=model_path)
        return run_evaluation_suite(model, tokenizer, dataset["test"], config)

    if artifact == "base":
        evaluation, predictions = cached_baseline(config, evaluate_model)
    else:
        evaluation, predictions = evaluate_model()
    stem = f"{config.run_name}_{artifact}"
    output_path = config.reports_root / f"{stem}_eval.json"
    predictions_path = _write_jsonl(config.reports_root / f"{stem}_predictions.jsonl", predictions)
    payload = build_run_report(
        **_dataset_fields(config),
        validation_examples=len(dataset["validation"]),
        eval_examples=len(dataset["test"]),
        training={
            "artifact": artifact,
            "target_format": config.target_format,
            "model_path": str(model_path) if model_path is not None else None,
        },
        training_summary={},
        artifacts={"predictions": str(predictions_path)},
        evaluation=evaluation,
    )
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
    prepare_adapter_for_publish(config.adapter_output_dir, config.base_model_id)

    merged_model = adapter_model.merge_and_unload()
    merged_model.save_pretrained(str(config.merged_output_dir))
    tokenizer.save_pretrained(str(config.merged_output_dir))

    return {
        "adapter": str(config.adapter_output_dir),
        "tokenizer": str(config.tokenizer_output_dir),
        "merged": str(config.merged_output_dir),
    }
