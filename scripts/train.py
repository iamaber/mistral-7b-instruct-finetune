import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pubmedqa_loraplus import LoRAPlusConfig, train_model
from src.pubmedqa_loraplus.env import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PubMedQA LoRA+ run.")
    parser.add_argument(
        "--config",
        default="configs/loraplus_train.json",
        help="Path to the LoRA+ config file.",
    )
    args = parser.parse_args()

    load_env_file()
    report = train_model(LoRAPlusConfig.from_file(args.config))
    print(json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()
