import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pubmedqa_loraplus import LoRAPlusConfig, push_artifact_to_hub
from src.pubmedqa_loraplus.env import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a LoRA+ artifact to Hugging Face Hub.")
    parser.add_argument(
        "--config",
        default="configs/loraplus_train.json",
        help="Path to the LoRA+ config file.",
    )
    parser.add_argument(
        "--artifact",
        choices=("adapter", "merged"),
        required=True,
        help="Artifact type to publish.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional explicit source directory instead of the config default.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help=(
            "Make the Hub repo public. Publishing is private unless this flag is "
            "given: the artifact is an experimental, collapsed model."
        ),
    )
    args = parser.parse_args()

    load_env_file()
    config = LoRAPlusConfig.from_file(args.config)
    if args.public:
        config.hub_private = False
    result = push_artifact_to_hub(
        config,
        artifact=args.artifact,
        source_dir=args.source_dir,
    )
    result["private"] = str(config.hub_private)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
