import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import resolve_repo_path
from .prompts import TARGET_FORMATS

VALID_MODEL_DTYPES = {"auto", "float32", "float16", "bfloat16"}


@dataclass
class LoRAPlusConfig:
    run_name: str
    base_model_id: str
    base_model_path: Path
    dataset_id: str
    dataset_config: str
    dataset_disk_path: Path
    output_root: Path
    reports_root: Path
    # The notebook mounts the volume and reads this as its default root; the
    # GPU, timeout, secret and image are properties of the Modal app, not of a
    # training run, so they are not part of the config.
    modal_mount_root: str = "/vol/pubmedqa-loraplus"
    base_model_revision: str | None = None
    dataset_revision: str | None = None
    model_dtype: str = "auto"
    validation_size: float = 0.1
    seed: int = 42
    max_length: int = 1536
    target_format: str = "rationale"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    rationale_max_new_tokens: int = 192
    balance_train_classes: bool = True
    pqa_artificial_samples: int = 0
    train_batch_size: int = 4
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    loraplus_lr_ratio: float = 4.0
    loraplus_lr_embedding: float = 1e-6
    num_epochs: int = 3
    logging_steps: int = 10
    save_steps: int = 20
    eval_steps: int = 20
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    group_by_length: bool = True
    max_grad_norm: float = 1.0
    evaluate_base_model: bool = True
    bootstrap_samples: int = 1000
    mmlu_eval_samples: int = 300
    mmlu_dataset_id: str = "cais/mmlu"
    mmlu_dataset_config: str = "all"
    mmlu_split: str = "test"
    mmlu_revision: str | None = None
    use_wandb: bool = True
    wandb_project: str = "pubmedqa-loraplus"
    wandb_run_name: str | None = None
    adapter_hub_repo_id: str | None = None
    merged_hub_repo_id: str | None = None
    hub_private: bool = True
    hub_token_env: str = "HF_TOKEN"
    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        path_fields = [
            "base_model_path",
            "dataset_disk_path",
            "output_root",
            "reports_root",
        ]
        for field_name in path_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                setattr(self, field_name, resolve_repo_path(value))

        if self.loraplus_lr_ratio < 1.0:
            raise ValueError("loraplus_lr_ratio must be >= 1.0")
        if self.model_dtype not in VALID_MODEL_DTYPES:
            raise ValueError(f"model_dtype must be one of {sorted(VALID_MODEL_DTYPES)}")
        if self.target_format not in TARGET_FORMATS:
            raise ValueError(f"target_format must be one of {list(TARGET_FORMATS)}")
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1")
        if self.pqa_artificial_samples < 0:
            raise ValueError("pqa_artificial_samples must be >= 0")
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = str(resolve_repo_path(self.resume_from_checkpoint))

    @property
    def base_model_source(self) -> str:
        if self.base_model_path.exists():
            return str(self.base_model_path)
        return self.base_model_id

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def adapter_output_dir(self) -> Path:
        return self.output_root / "adapter"

    @property
    def merged_output_dir(self) -> Path:
        return self.output_root / "merged"

    @property
    def tokenizer_output_dir(self) -> Path:
        return self.output_root / "tokenizer"

    @property
    def report_path(self) -> Path:
        return self.reports_root / f"{self.run_name}.json"

    @property
    def predictions_path(self) -> Path:
        return self.reports_root / f"{self.run_name}_predictions.jsonl"

    @property
    def baseline_predictions_path(self) -> Path:
        return self.reports_root / f"{self.run_name}_baseline_predictions.jsonl"

    @property
    def training_log_path(self) -> Path:
        return self.reports_root / f"{self.run_name}_train_history.json"

    @property
    def config_snapshot_path(self) -> Path:
        return self.reports_root / f"{self.run_name}_config.json"

    @classmethod
    def from_file(cls, path: str | Path) -> "LoRAPlusConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for key, value in list(raw.items()):
            if isinstance(value, Path):
                raw[key] = str(value)
        raw["checkpoints_dir"] = str(self.checkpoints_dir)
        raw["adapter_output_dir"] = str(self.adapter_output_dir)
        raw["merged_output_dir"] = str(self.merged_output_dir)
        raw["tokenizer_output_dir"] = str(self.tokenizer_output_dir)
        raw["report_path"] = str(self.report_path)
        raw["training_log_path"] = str(self.training_log_path)
        raw["config_snapshot_path"] = str(self.config_snapshot_path)
        raw["base_model_source"] = self.base_model_source
        return raw

    def ensure_output_dirs(self) -> None:
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.reports_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create output directories: {e}") from e

    def save_snapshot(self) -> Path:
        self.ensure_output_dirs()
        self.config_snapshot_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        return self.config_snapshot_path


def planned_optimizer_steps(
    config: LoRAPlusConfig, train_examples: int, world_size: int = 1
) -> int:
    """Optimizer steps ``Trainer`` will run, raising if selection would never evaluate.

    The published run planned ~150 steps against ``eval_steps=200``, so the
    validation split was never scored and ``load_best_model_at_end`` silently
    kept the last step. Failing here makes that configuration impossible.
    """
    batches = math.ceil(train_examples / (config.train_batch_size * world_size))
    steps = math.ceil(batches / config.gradient_accumulation_steps) * config.num_epochs
    if steps < config.eval_steps:
        raise ValueError(
            f"Training runs {steps} optimizer steps but eval_steps={config.eval_steps}: "
            "checkpoint selection would never evaluate. Lower eval_steps/save_steps."
        )
    if config.save_steps % config.eval_steps:
        raise ValueError("save_steps must be a multiple of eval_steps for best-model selection")
    return steps
