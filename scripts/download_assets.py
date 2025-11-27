import argparse
import logging
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from huggingface_hub import snapshot_download

from src.mistral_pubmedqa.config import ExperimentConfig
from src.mistral_pubmedqa.env import load_env_file


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_dataset(config: ExperimentConfig) -> None:
    if config.dataset_disk_path.exists():
        LOGGER.info("Dataset already present at %s", config.dataset_disk_path)
        return

    config.dataset_disk_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(config.dataset_id, config.dataset_config)
    dataset.save_to_disk(str(config.dataset_disk_path))
    LOGGER.info("Saved dataset to %s", config.dataset_disk_path)


def download_model(config: ExperimentConfig) -> None:
    if config.base_model_path.exists():
        LOGGER.info("Base model already present at %s", config.base_model_path)
        return

    config.base_model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.base_model_id,
        local_dir=str(config.base_model_path),
        local_dir_use_symlinks=False,
        token=os.getenv("HF_TOKEN") or None,
    )
    LOGGER.info("Saved base model to %s", config.base_model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PubMedQA and the base model.")
    parser.add_argument(
        "--config",
        default="configs/sft_masked_lora.json",
        help="Config file used to resolve download locations.",
    )
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = ExperimentConfig.from_file(args.config)

    if not args.model_only:
        download_dataset(config)
    if not args.dataset_only:
        download_model(config)


if __name__ == "__main__":
    main()
