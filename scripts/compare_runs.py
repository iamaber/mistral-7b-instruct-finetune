import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two experiment result files.")
    parser.add_argument(
        "--baseline",
        default="reports/evaluations/baseline_current_lora.json",
        help="Path to the baseline metrics JSON.",
    )
    parser.add_argument(
        "--candidate",
        default="reports/evaluations/sft_masked_lora.json",
        help="Path to the candidate metrics JSON.",
    )
    args = parser.parse_args()

    baseline = load_metrics(Path(args.baseline))
    candidate = load_metrics(Path(args.candidate))

    print("metric,baseline,candidate,delta")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        baseline_value = baseline["pubmedqa"][metric]
        candidate_value = candidate["pubmedqa"][metric]
        delta = candidate_value - baseline_value
        print(f"{metric},{baseline_value:.4f},{candidate_value:.4f},{delta:+.4f}")


if __name__ == "__main__":
    main()
