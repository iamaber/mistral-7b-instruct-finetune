"""Run PubMedQA LoRA+ on Modal.

One-time setup, if the workspace has no such secret yet:
modal secret create huggingface-secret HF_TOKEN=...

modal run modal_app.py --action tests
modal run modal_app.py --action smoke
modal run modal_app.py --action train
modal run modal_app.py --action evaluate --artifact base
"""

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from src.pubmedqa_loraplus.config import LoRAPlusConfig

LOCAL_REPO = Path(__file__).resolve().parent
CONTAINER_REPO = Path("/root/repo")
VOLUME_ROOT = Path("/vol/pubmedqa-loraplus")
DEFAULT_GPU = "H100"
SMOKE_OVERRIDES = {
    "run_name": "smoke-qwen2.5-0.5b",
    "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "base_model_revision": None,
    "num_epochs": 1,
    "eval_steps": 10,
    "save_steps": 10,
    "mmlu_eval_samples": 30,
    "rationale_max_new_tokens": 64,
    "bootstrap_samples": 200,
}

app = modal.App("pubmedqa-loraplus")
volume = modal.Volume.from_name("pubmedqa-loraplus", create_if_missing=True)
huggingface = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_REPO / "requirements.lock.txt"))
    .workdir(CONTAINER_REPO)
    .env({"PYTHONPATH": f"{CONTAINER_REPO}:{CONTAINER_REPO.parent}"})
)


def _add_repo_sources(image: modal.Image) -> modal.Image:
    for directory in ("src", "configs", "scripts", "tests"):
        image = image.add_local_dir(
            LOCAL_REPO / directory, str(CONTAINER_REPO / directory), ignore=["**/__pycache__"]
        )
    return image.add_local_file(
        LOCAL_REPO / "pyproject.toml", str(CONTAINER_REPO / "pyproject.toml")
    )


image = _add_repo_sources(base_image)
test_image = _add_repo_sources(base_image.pip_install("pytest>=8"))


def _parse_overrides(overrides_json: str) -> dict[str, Any]:
    overrides = json.loads(overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("Overrides must be a JSON object")
    return overrides


def _build_config(overrides_json: str) -> "LoRAPlusConfig":
    from src.pubmedqa_loraplus.config import LoRAPlusConfig

    config_path = CONTAINER_REPO / "configs" / "loraplus_train.json"
    defaults = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = _parse_overrides(overrides_json)
    model_id = overrides.get("base_model_id", defaults["base_model_id"])
    run_name = overrides.get("run_name", defaults["run_name"])
    if model_id != defaults["base_model_id"]:
        overrides.setdefault("base_model_revision", None)
    return LoRAPlusConfig(
        **{
            **defaults,
            "base_model_path": VOLUME_ROOT / "base_models" / model_id.replace("/", "--"),
            "dataset_disk_path": VOLUME_ROOT / "dataset",
            "output_root": VOLUME_ROOT / "runs" / run_name,
            "reports_root": VOLUME_ROOT / "reports",
            "modal_mount_root": str(VOLUME_ROOT),
            "use_wandb": False,
            **overrides,
        }
    )


def _download_if_missing(
    config: "LoRAPlusConfig", field: str, download: Callable[["LoRAPlusConfig"], None]
) -> None:
    target = getattr(config, field)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted download must leave the final cache path absent for the next run.
    with TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as staging:
        staged = Path(staging) / target.name
        download(replace(config, **{field: staged}))
        staged.rename(target)


def _summarize(report: dict[str, Any], report_path: str) -> dict[str, Any]:
    evaluation = report["evaluation"]
    pubmedqa = evaluation["pubmedqa"]
    baseline = None
    if (base := evaluation.get("baseline")) is not None:
        baseline = {"pubmedqa": {key: base["pubmedqa"][key] for key in ("accuracy", "macro_f1")}}
    training = report["training"]
    return {
        "run_name": report["run_name"],
        "test": {
            "pubmedqa": {
                key: pubmedqa.get(key)
                for key in (
                    "accuracy",
                    "macro_f1",
                    "confidence_intervals",
                    "prediction_counts",
                    "collapsed",
                )
            },
            "mmlu_medical": {"overall_accuracy": evaluation["mmlu_medical"]["overall_accuracy"]},
        },
        "baseline": baseline,
        "delta_vs_baseline": evaluation.get("delta_vs_baseline"),
        "best_model_checkpoint": training.get("best_model_checkpoint"),
        "global_step": training.get("global_step"),
        "report_path": report_path,
        "report": report,
    }


@app.function(image=test_image)
def run_tests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=CONTAINER_REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    return result.stdout


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes={VOLUME_ROOT: volume},
    secrets=[huggingface],
)
def train(overrides_json: str) -> dict:
    from scripts.prepare_assets import download_dataset, download_model
    from src.pubmedqa_loraplus.workflow import train_model

    config = _build_config(overrides_json)
    volume.reload()
    _download_if_missing(config, "dataset_disk_path", download_dataset)
    _download_if_missing(config, "base_model_path", download_model)
    report = train_model(config)
    volume.commit()
    return _summarize(report, str(config.report_path))


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes={VOLUME_ROOT: volume},
    secrets=[huggingface],
)
def evaluate(artifact: str, overrides_json: str) -> dict:
    from scripts.prepare_assets import download_dataset, download_model
    from src.pubmedqa_loraplus.workflow import ARTIFACTS, evaluate_artifact

    if artifact not in ARTIFACTS:
        raise ValueError(f"artifact must be one of {ARTIFACTS}")
    config = _build_config(overrides_json)
    volume.reload()
    _download_if_missing(config, "dataset_disk_path", download_dataset)
    if artifact != "merged":
        _download_if_missing(config, "base_model_path", download_model)
    report = evaluate_artifact(config, artifact)
    volume.commit()
    return _summarize(report, report["report_path"])


@app.local_entrypoint()
def main(
    action: str = "train", overrides: str = "{}", gpu: str = "H100", artifact: str = "adapter"
) -> None:
    if action == "tests":
        run_tests.remote()
        return
    if action == "smoke":
        overrides = json.dumps({**SMOKE_OVERRIDES, **_parse_overrides(overrides)})
        result = train.with_options(gpu="L4").remote(overrides)
    elif action == "train":
        result = train.with_options(gpu=gpu).remote(overrides)
    elif action == "evaluate":
        result = evaluate.with_options(gpu=gpu).remote(artifact, overrides)
    else:
        raise ValueError("action must be one of tests, train, evaluate, smoke")

    report = result.pop("report")
    report_path = LOCAL_REPO / "reports" / "modal" / f"{result['run_name']}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
