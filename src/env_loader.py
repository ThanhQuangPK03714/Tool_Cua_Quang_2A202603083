"""Load simple KEY=VALUE environment files without third-party packages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def load_dotenv(dotenv_path: Optional[Union[str, Path]] = None) -> bool:
    """Read a .env file, keeping variables already present in the environment."""
    path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
    if not path.is_file():
        return False

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value

    return True
