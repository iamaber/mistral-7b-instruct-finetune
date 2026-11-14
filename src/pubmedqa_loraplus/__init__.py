"""LoRA+ training utilities for PubMedQA.

Submodules are imported lazily so the pure helpers (``scoring``, ``prompts``,
``artifact``) can be imported and unit-tested without pulling in torch,
transformers or peft.
"""

from typing import Any

__all__ = [
    "LoRAPlusConfig",
    "build_model_card",
    "evaluate_artifact",
    "export_artifacts",
    "prepare_adapter_for_publish",
    "push_artifact_to_hub",
    "rewrite_base_model_reference",
    "train_model",
]

_LAZY_IMPORTS = {
    "LoRAPlusConfig": ".config",
    "build_model_card": ".hub",
    "push_artifact_to_hub": ".hub",
    "evaluate_artifact": ".workflow",
    "export_artifacts": ".workflow",
    "train_model": ".workflow",
    "prepare_adapter_for_publish": ".artifact",
    "rewrite_base_model_reference": ".artifact",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
