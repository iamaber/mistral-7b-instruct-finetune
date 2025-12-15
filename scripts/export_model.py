import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pubmedqa_loraplus import LoRAPlusConfig, export_artifacts
from src.pubmedqa_loraplus.env import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export adapter and merged LoRA+ artifacts.")
    parser.add_argument(
        "--config",
        default="configs/loraplus_train.json",
        help="Path to the LoRA+ config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional adapter checkpoint directory to export from.",
    )
    args = parser.parse_args()

    load_env_file()
    result = export_artifacts(
        LoRAPlusConfig.from_file(args.config),
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
