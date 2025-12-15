import os
from pathlib import Path

from .paths import REPO_ROOT


def load_env_file(env_path: str | Path = ".env") -> Path:
    path = Path(env_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        return path

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))

    return path
