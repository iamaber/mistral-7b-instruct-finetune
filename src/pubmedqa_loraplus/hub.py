import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .config import LoRAPlusConfig


def _load_report(config: LoRAPlusConfig) -> dict[str, Any] | None:
    if not config.report_path.exists():
        return None
    return json.loads(config.report_path.read_text(encoding="utf-8"))


def build_model_card(config: LoRAPlusConfig, artifact: str) -> str:
    report = _load_report(config)
    metrics_lines = ["Metrics not available yet."]
    if report is not None:
        pubmedqa = report.get("evaluation", {}).get("pubmedqa", {})
        mmlu = report.get("evaluation", {}).get("mmlu_medical", {})
        metrics_lines = [
            f"- PubMedQA accuracy: {pubmedqa.get('accuracy', 0.0):.4f}",
            f"- PubMedQA macro F1: {pubmedqa.get('macro_f1', 0.0):.4f}",
            f"- Medical MMLU accuracy: {mmlu.get('overall_accuracy', 0.0):.4f}",
        ]

    base_model_line = f"- Base model: `{config.base_model_id}`"
    dataset_line = f"- Dataset: `{config.dataset_id}` / `{config.dataset_config}`"
    artifact_line = f"- Artifact type: `{artifact}`"
    training_line = (
        f"- LoRA+: rank={config.lora_r}, alpha={config.lora_alpha}, "
        f"dropout={config.lora_dropout}, lr_ratio={config.loraplus_lr_ratio}"
    )

    return "\n".join(
        [
            "---",
            "license: apache-2.0",
            "library_name: transformers",
            "tags:",
            "  - peft",
            "  - lora",
            "  - loraplus",
            "  - pubmedqa",
            "  - text-generation",
            "---",
            "",
            f"# {config.run_name}",
            "",
            "PubMedQA LoRA+ fine-tuning artifact prepared for Hugging Face publication.",
            "",
            "## Training setup",
            base_model_line,
            dataset_line,
            artifact_line,
            training_line,
            "",
            "## Evaluation",
            *metrics_lines,
            "",
            "## Usage",
            "Load the tokenizer from the published artifact directory. "
            "For adapter artifacts, load the base model first and attach the adapter with PEFT.",
        ]
    )


def push_artifact_to_hub(
    config: LoRAPlusConfig,
    artifact: str,
    source_dir: str | Path | None = None,
) -> dict[str, str]:
    if artifact not in {"adapter", "merged"}:
        raise ValueError("artifact must be 'adapter' or 'merged'")

    repo_id = (
        config.adapter_hub_repo_id if artifact == "adapter" else config.merged_hub_repo_id
    )
    if not repo_id:
        raise ValueError(f"No Hugging Face repo configured for {artifact} artifact")

    artifact_dir = Path(source_dir) if source_dir is not None else (
        config.adapter_output_dir if artifact == "adapter" else config.merged_output_dir
    )
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")

    token = os.getenv(config.hub_token_env) or None
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=config.hub_private,
        exist_ok=True,
    )

    readme_path = artifact_dir / "README.md"
    readme_path.write_text(build_model_card(config, artifact), encoding="utf-8")
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=repo_id,
        repo_type="model",
    )

    return {
        "artifact": artifact,
        "repo_id": repo_id,
        "source_dir": str(artifact_dir),
    }
