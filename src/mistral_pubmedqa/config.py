from dataclasses import dataclass
import json
from pathlib import Path

from .paths import resolve_repo_path


@dataclass
class ExperimentConfig:
    experiment_name: str
    training_mode: str
    base_model_id: str
    base_model_path: Path
    dataset_id: str
    dataset_config: str
    dataset_disk_path: Path
    output_dir: Path
    adapter_output_dir: Path
    tokenizer_output_dir: Path
    results_output_path: Path
    test_size: float = 0.1
    seed: int = 42
    max_length: int = 512
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_new_tokens: int = 10
    mmlu_eval_samples: int = 50
    use_wandb: bool = True
    wandb_project: str = "pubmedqa-mistral-comparison"
    wandb_run_name: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        path_keys = [
            "base_model_path",
            "dataset_disk_path",
            "output_dir",
            "adapter_output_dir",
            "tokenizer_output_dir",
            "results_output_path",
        ]
        for key in path_keys:
            raw[key] = resolve_repo_path(raw[key])
        return cls(**raw)
