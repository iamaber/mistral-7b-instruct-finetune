from datasets import DatasetDict, load_dataset, load_from_disk

from .config import ExperimentConfig


def flatten_context(context: object) -> str:
    if isinstance(context, dict) and "contexts" in context:
        return " ".join(context["contexts"])
    if isinstance(context, list):
        return " ".join(str(part) for part in context)
    return str(context)


def load_pubmedqa(config: ExperimentConfig) -> DatasetDict:
    if config.dataset_disk_path.exists():
        dataset = load_from_disk(str(config.dataset_disk_path))
        train_split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    else:
        source = load_dataset(config.dataset_id, config.dataset_config)
        train_split = source["train"]

    return train_split.train_test_split(test_size=config.test_size, seed=config.seed)
