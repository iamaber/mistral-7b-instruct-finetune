from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
