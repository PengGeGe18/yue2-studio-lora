"""Small, dependency-free helpers shared by the CLI."""

from __future__ import annotations

import hashlib
import os
import shlex
import sys
from pathlib import Path
from typing import Iterable


def eprint(*args, **kwargs) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def sha256_of(path: os.PathLike, chunk: int = 8 << 20, progress=None) -> str:
    digest = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    return digest.hexdigest()


def ensure_dir(path: os.PathLike) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def quote(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def find_on_path(name: str) -> str | None:
    from shutil import which

    return which(name)


def is_windows() -> bool:
    return os.name == "nt"


def platform_key() -> str:
    import platform

    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"


def replace_path(value: object, root: Path) -> Path:
    """Resolve a config path against the config file's folder."""

    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path)
