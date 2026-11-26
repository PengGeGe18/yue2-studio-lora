"""Finding, downloading or building the ``ace-train`` helper.

YuE2-Studio ships the trainer as a prebuilt Windows x64 + CUDA archive. On
Linux the same trainer is built from the pinned HOT-Step commit; on macOS the
joint trainer cannot be built (its loss kernel is CUDA-only), so a path to a
trainer produced elsewhere has to be supplied.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .assets import (
    TRAINER,
    TRAINER_COMMIT,
    TRAINER_FOLDER,
    TRAINER_REPOSITORY,
    TRAINER_SHIPPED_AS,
)
from .download import download, extract_zip, find_file
from .util import eprint, find_on_path, is_windows, platform_key

ACE_TRAIN = "ace-train.exe" if is_windows() else "ace-train"


class TrainerError(RuntimeError):
    pass


def locate(workspace: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    """Look for a trainer without downloading anything."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    from_env = os.environ.get("YUE_TRAIN_BIN") or os.environ.get("YUE2_TRAIN_BIN")
    if from_env and Path(from_env).is_file():
        return Path(from_env)
    trainer_dir = workspace / "trainer"
    for name in (TRAINER_SHIPPED_AS, ACE_TRAIN):
        direct = trainer_dir / name
        if direct.is_file():
            return direct
    if trainer_dir.is_dir():
        found = find_file(trainer_dir, TRAINER_SHIPPED_AS) or find_file(trainer_dir, ACE_TRAIN)
        if found:
            return found
    on_path = find_on_path("ace-train") or find_on_path("music-train")
    return Path(on_path) if on_path else None


def ensure(workspace: Path, explicit: Optional[Path] = None, build: bool = False,
           allow_unsupported: bool = False) -> Path:
    """Return a usable trainer, downloading or building one when needed."""
    found = locate(workspace, explicit)
    if found:
        return found
    if explicit:
        raise TrainerError(f"the trainer set in the config does not exist: {explicit}")
    system = platform_key()
    if system == "windows":
        return install_prebuilt(workspace)
    if build or system == "linux":
        return build_from_source(workspace)
    if allow_unsupported:
        raise TrainerError(
            "no trainer found. YuE2 joint training needs a CUDA build of ace-train; "
            "point train.trainer at a binary built on a CUDA machine."
        )
    raise TrainerError(
        "no trainer found and this platform cannot build one: the YuE2 joint "
        "training loss is CUDA-only. Build ace-train on a CUDA Linux/Windows "
        "machine and set train.trainer to its path."
    )


def install_prebuilt(workspace: Path) -> Path:
    """Download and unpack YuE2-Studio's prebuilt Windows trainer."""
    target_dir = workspace / "trainer"
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / TRAINER.filename
    if not archive.is_file() or archive.stat().st_size != TRAINER.bytes:
        eprint(f"[trainer] downloading {TRAINER.label} ({TRAINER.bytes} bytes)")
        download(TRAINER.url, archive, TRAINER.bytes)
    unpacked = target_dir / TRAINER_FOLDER
    if not unpacked.is_dir():
        eprint(f"[trainer] unpacking into {unpacked}")
        extract_zip(archive, unpacked)
    found = find_file(unpacked, TRAINER_SHIPPED_AS)
    if not found:
        raise TrainerError(f"{TRAINER_SHIPPED_AS} was not found in {archive}")
    return found


def build_from_source(workspace: Path, jobs: Optional[int] = None) -> Path:
    """Build ``ace-train`` from the pinned HOT-Step commit (Linux, CUDA)."""
    source = workspace / "trainer" / "HOT-Step-CPP"
    build = workspace / "trainer" / "build"
    if not source.is_dir():
        _require("git")
        eprint(f"[trainer] cloning {TRAINER_REPOSITORY} at {TRAINER_COMMIT[:8]}")
        subprocess.run(["git", "clone", TRAINER_REPOSITORY, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", TRAINER_COMMIT], check=True)
        subprocess.run(
            ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
    _require("cmake")
    build.mkdir(parents=True, exist_ok=True)
    eprint("[trainer] configuring (CUDA)")
    subprocess.run(
        [
            "cmake", "-S", str(source / "engine"), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={'native'}",
        ],
        check=True,
    )
    eprint("[trainer] building ace-train")
    command = ["cmake", "--build", str(build), "--target", "ace-train"]
    if jobs:
        command += ["--parallel", str(jobs)]
    else:
        command += ["--parallel"]
    subprocess.run(command, check=True)
    found = find_file(build, ACE_TRAIN)
    if not found:
        raise TrainerError("the build finished but ace-train was not found")
    destination = workspace / "trainer" / ACE_TRAIN
    shutil.copy2(found, destination)
    return destination


def _require(tool: str) -> None:
    if not find_on_path(tool):
        raise TrainerError(f"{tool} is required to build the trainer")


def probe(trainer: Path) -> str:
    """Run ``--help`` and return its output; used by ``doctor``."""
    if not trainer.is_file():
        raise TrainerError(f"no trainer at {trainer}")
    result = subprocess.run(
        [str(trainer), "--help"], capture_output=True, text=True, timeout=120
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode not in (0, 2) or "yue2-joint-train" not in output:
        raise TrainerError(
            f"{trainer} does not look like ace-train (exit {result.returncode})"
        )
    return output
