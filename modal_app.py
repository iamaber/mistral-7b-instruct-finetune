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
    merged = {**defaults, **overrides}
    # Cache keys carry the revision so a warm volume never serves a different pin.
    model_key = f"{model_id.replace('/', '--')}@{merged['base_model_revision'] or 'main'}"
    dataset_key = f"{merged['dataset_config']}@{merged['dataset_revision'] or 'main'}"
    return LoRAPlusConfig(
        **{
            **defaults,
            "base_model_path": VOLUME_ROOT / "base_models" / model_key,
            "dataset_disk_path": VOLUME_ROOT / "datasets" / dataset_key,
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
    # Room to stage a ~55 GB checkpoint on the host before it moves to the GPU.
    memory=96 * 1024,
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
    memory=96 * 1024,
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


@app.function(image=image, timeout=60 * 60, volumes={VOLUME_ROOT: volume}, secrets=[huggingface])
def push(overrides_json: str, seed_runs: list[str]) -> dict:
    """Publish a trained adapter with a card that also shows its seed spread."""
    import statistics

    from src.pubmedqa_loraplus.hub import build_model_card, push_artifact_to_hub

    config = _build_config(overrides_json)
    volume.reload()
    report = json.loads(config.report_path.read_text(encoding="utf-8"))
    runs = []
    for name in [config.run_name, *seed_runs]:
        other = json.loads((config.reports_root / f"{name}.json").read_text(encoding="utf-8"))
        snapshot = json.loads(
            (config.reports_root / f"{name}_config.json").read_text(encoding="utf-8")
        )
        pubmedqa = other["evaluation"]["pubmedqa"]
        runs.append(
            {
                "seed": snapshot["seed"],
                "accuracy": pubmedqa["accuracy"],
                "macro_f1": pubmedqa["macro_f1"],
            }
        )
    if len(runs) > 1:
        report["seed_summary"] = {
            "published_seed": config.seed,
            "runs": runs,
            **{
                f"{key}_{stat}": fn([run[key] for run in runs])
                for key in ("accuracy", "macro_f1")
                for stat, fn in (("mean", statistics.mean), ("sd", statistics.stdev))
            },
        }
    # A later standalone re-evaluation (evaluate --artifact base/adapter) may hold the
    # two-way MMLU score the training run predates; the card should show the newer one.
    evals = {
        artifact: config.reports_root / f"{config.run_name}_{artifact}_eval.json"
        for artifact in ("base", "adapter")
    }
    if all(path.exists() for path in evals.values()):
        mmlu = {
            artifact: json.loads(path.read_text(encoding="utf-8"))["evaluation"]["mmlu_medical"]
            for artifact, path in evals.items()
        }
        if all("answer_line_accuracy" in result for result in mmlu.values()):
            report["evaluation"]["mmlu_medical"] = mmlu["adapter"]
            report["evaluation"].setdefault("baseline", {})["mmlu_medical"] = mmlu["base"]
    config.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result = push_artifact_to_hub(config, "adapter")
    volume.commit()
    return {**result, "card": build_model_card(config, "adapter")}


def _baseline_key(overrides: dict[str, Any]) -> str:
    from src.pubmedqa_loraplus.config import LoRAPlusConfig, baseline_cache_key

    defaults = json.loads((LOCAL_REPO / "configs" / "loraplus_train.json").read_text())
    if overrides.get("base_model_id", defaults["base_model_id"]) != defaults["base_model_id"]:
        overrides = {"base_model_revision": None, **overrides}
    return baseline_cache_key(LoRAPlusConfig(**{**defaults, **overrides}))


def _write_local_report(stem: str, report: dict[str, Any]) -> None:
    report_path = LOCAL_REPO / "reports" / "modal" / f"{stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _sweep_row(result: dict[str, Any]) -> str:
    pubmedqa = result["test"]["pubmedqa"]
    intervals = pubmedqa.get("confidence_intervals") or {}
    delta = (result.get("delta_vs_baseline") or {}).get("macro_f1", {})
    return (
        f"| {result['run_name']} | {pubmedqa['accuracy']:.3f} | {pubmedqa['macro_f1']:.3f} "
        f"| {intervals.get('macro_f1')} | {delta.get('delta')} | {delta.get('interval')} |"
    )


def _run_sweep(sweep_path: str, gpu: str) -> None:
    """Train every variant in parallel, after one shared baseline per distinct base setup."""
    variants = json.loads(Path(sweep_path).read_text(encoding="utf-8"))
    names = [variant.get("run_name") for variant in variants]
    if None in names or len(set(names)) != len(names):
        raise ValueError("Every sweep variant needs a unique run_name")

    first_by_key: dict[str, dict[str, Any]] = {}
    for variant in variants:
        first_by_key.setdefault(_baseline_key(variant), variant)
    # Warm the baseline cache first so parallel variants reuse it instead of each
    # re-scoring the base model.
    list(
        evaluate.with_options(gpu=gpu).starmap(
            [("base", json.dumps(variant)) for variant in first_by_key.values()]
        )
    )

    rows = []
    results = train.with_options(gpu=gpu).map(
        [json.dumps(variant) for variant in variants], return_exceptions=True
    )
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            rows.append(f"| {name} | failed: {result!r} | | | | |")
            continue
        _write_local_report(result["run_name"], result.pop("report"))
        rows.append(_sweep_row(result))
    print("| run | accuracy | macro F1 | macro F1 95% CI | delta macro F1 vs base | delta CI |")
    print("| --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))


@app.local_entrypoint()
def main(
    action: str = "train",
    overrides: str = "{}",
    gpu: str = "H100",
    artifact: str = "adapter",
    sweep: str = "configs/sweeps/phase2.json",
    seed_runs: str = "",
) -> None:
    if action == "tests":
        run_tests.remote()
        return
    if action == "sweep":
        _run_sweep(sweep, gpu)
        return
    if action == "push":
        names = [name for name in seed_runs.split(",") if name]
        result = push.remote(overrides, names)
        card_path = LOCAL_REPO / "reports" / "modal" / f"{result['repo_id'].split('/')[-1]}.md"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(result.pop("card"), encoding="utf-8")
        print(json.dumps({**result, "card": str(card_path)}, indent=2))
        return
    if action == "smoke":
        overrides = json.dumps({**SMOKE_OVERRIDES, **_parse_overrides(overrides)})
        result = train.with_options(gpu="L4").remote(overrides)
    elif action == "train":
        result = train.with_options(gpu=gpu).remote(overrides)
    elif action == "evaluate":
        result = evaluate.with_options(gpu=gpu).remote(artifact, overrides)
    else:
        raise ValueError("action must be one of tests, train, evaluate, smoke, sweep, push")

    stem = result["run_name"] if action != "evaluate" else f"{result['run_name']}_{artifact}_eval"
    _write_local_report(stem, result.pop("report"))
    print(json.dumps(result, indent=2))
