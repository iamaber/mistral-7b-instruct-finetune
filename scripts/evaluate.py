import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pubmedqa_loraplus import LoRAPlusConfig, evaluate_artifact
from src.pubmedqa_loraplus.env import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved LoRA+ artifact.")
    parser.add_argument(
        "--config",
        default="configs/loraplus_train.json",
        help="Path to the LoRA+ config file.",
    )
    parser.add_argument(
        "--artifact",
        choices=("adapter", "merged"),
        default="adapter",
        help="Saved artifact type to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit adapter or merged-model directory.",
    )
    args = parser.parse_args()

    load_env_file()
    report = evaluate_artifact(
        LoRAPlusConfig.from_file(args.config),
        args.artifact,
        model_path=args.model_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
