import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mistral_pubmedqa.env import load_env_file
from src.mistral_pubmedqa.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline current-LoRA run.")
    parser.add_argument(
        "--config",
        default="configs/baseline_current_lora.json",
        help="Path to the baseline experiment config.",
    )
    args = parser.parse_args()
    load_env_file()
    run_experiment(Path(args.config))


if __name__ == "__main__":
    main()
