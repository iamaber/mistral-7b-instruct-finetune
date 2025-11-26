from pathlib import Path

from src.mistral_pubmedqa.env import load_env_file
from src.mistral_pubmedqa.pipeline import run_experiment


def main() -> None:
    load_env_file()
    run_experiment(Path("configs/baseline_current_lora.json"))


if __name__ == "__main__":
    main()
