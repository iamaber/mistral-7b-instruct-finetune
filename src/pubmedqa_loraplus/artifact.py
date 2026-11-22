"""Post-save fixups for published adapter artifacts.

Kept independent of torch so the rewrite can be exercised by unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAINER_SIDECARS = ("training_args.bin",)


def is_hub_model_id(value: str) -> bool:
    """Return True when ``value`` is an ``org/name`` Hub id and not a local path."""
    if not value or value.startswith(("/", ".", "~")):
        return False
    parts = value.split("/")
    if len(parts) != 2:
        return False
    if any(not part or part in {".", ".."} for part in parts):
        return False
    # A directory that exists on this machine is a trainer-local path, whatever
    # it is called; the published value must resolve remotely instead.
    return not Path(value).exists()


def rewrite_base_model_reference(
    adapter_config_path: str | Path,
    base_model_id: str,
    base_model_revision: str | None = None,
) -> dict[str, Any]:
    """Point ``base_model_name_or_path`` at the Hub id instead of a trainer path.

    ``Trainer.save_model`` writes the adapter while the base model is loaded
    from a local directory (for example ``/vol/pubmedqa-loraplus/base_model``),
    so PEFT records that local path in ``adapter_config.json``. On the Hub the
    path does not exist, and ``PeftModel.from_pretrained`` then either fails or
    silently falls back to the wrong checkpoint. Storing the public Hub id makes
    the artifact self-describing.
    """
    path = Path(adapter_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter config not found: {path}")
    if not is_hub_model_id(base_model_id):
        raise ValueError(
            f"base_model_id must be a Hugging Face Hub id, not a local path: {base_model_id!r}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["base_model_name_or_path"] = base_model_id
    # PEFT loads the base at ``revision``; without it a consumer gets whatever the
    # base repo's main branch holds at load time, not the weights this adapter fits.
    if base_model_revision:
        payload["revision"] = base_model_revision
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def remove_trainer_sidecars(artifact_dir: str | Path) -> list[str]:
    """Delete Trainer bookkeeping files that do not belong in a model repo.

    ``training_args.bin`` is a pickled ``TrainingArguments`` object: it embeds
    absolute trainer paths, is not needed for inference, and loading it executes
    a pickle. It is removed so the published adapter contains only what PEFT
    needs.
    """
    directory = Path(artifact_dir)
    removed: list[str] = []
    for filename in TRAINER_SIDECARS:
        candidate = directory / filename
        if candidate.exists():
            candidate.unlink()
            removed.append(filename)
    return removed


def prepare_adapter_for_publish(
    artifact_dir: str | Path,
    base_model_id: str,
    base_model_revision: str | None = None,
) -> dict[str, Any]:
    """Apply every publish-time fixup to an adapter directory."""
    directory = Path(artifact_dir)
    config = rewrite_base_model_reference(
        directory / "adapter_config.json", base_model_id, base_model_revision
    )
    removed = remove_trainer_sidecars(directory)
    return {
        "adapter_dir": str(directory),
        "base_model_name_or_path": config["base_model_name_or_path"],
        "removed_files": removed,
    }
