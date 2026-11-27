"""Finding or downloading the ``ace-train`` helper.

The official HOT-Step release ships ``engine/ace-train`` for Linux x64 and
Windows x64 (CUDA). Nothing is compiled here: the archive is downloaded,
verified against its SHA-256 and extracted under ``trainer/``. macOS has no
usable trainer - YuE2's joint-training loss is CUDA-only - so the code says so
before any download starts and asks for ``[train] trainer`` instead.

Version note: these are HOT-Step ``v1.6.0`` archives. Their ``ace-train``
exposes the same ``yue2-*`` subcommands as the commit YuE2-Studio pins, but
training numerics can drift between versions; ``ace-train --fixture-replay``
can gate that before trusting a prebuilt over a local build.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from .assets import HOTSTEP_RELEASES
from .download import archive_stem, ensure_asset, extract_archive, find_file
from .util import eprint, find_on_path, is_windows, platform_key

ACE_TRAIN = "ace-train.exe" if is_windows() else "ace-train"

#: Older YuE2-Studio tool packages shipped the trainer under this name.
TRAINER_SHIPPED_AS = "music-train.exe"

#: Only CUDA builds can run the joint loss; Vulkan/ROCm/CPU/Metal cannot.
TRAIN_VARIANTS = {
    name: asset for name, asset in HOTSTEP_RELEASES.items() if "cuda" in name
}


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


def ensure(workspace: Path, explicit: Optional[Path] = None) -> Path:
    """Return a usable trainer, downloading the official prebuilt if needed."""
    found = locate(workspace, explicit)
    if found:
        return found
    if explicit:
        raise TrainerError(f"the trainer set in the config does not exist: {explicit}")
    return install_release(workspace)


def choose_variant(variant: Optional[str] = None) -> str:
    """Pick the CUDA release variant for this machine."""
    override = variant or os.environ.get("YUE2_TRAIN_VARIANT")
    if override:
        if override not in TRAIN_VARIANTS:
            raise TrainerError(
                f"unknown or non-CUDA YUE2_TRAIN_VARIANT {override!r}; "
                "choose one of: " + ", ".join(sorted(TRAIN_VARIANTS))
            )
        return override
    system = platform_key()
    machine = platform.machine().lower()
    if system in ("linux", "windows"):
        if machine not in ("x86_64", "amd64", "x64"):
            raise TrainerError(
                f"the official HOT-Step releases are x64 only, so there is no "
                f"prebuilt ace-train for {system} {machine}. Run on an x64 CUDA "
                "machine, or set [train] trainer to a locally built binary."
            )
        return "linux-x64-cuda12.8" if system == "linux" else "win-x64-cuda12.8"
    if system == "macos":
        raise TrainerError(
            "YuE2 joint training needs CUDA, so the macOS build cannot train it. "
            "Run training on a Linux/Windows CUDA machine, or build ace-train there "
            "and point [train] trainer at it. Dataset prep, captioning and "
            "`yue2-lora stages` still run on macOS."
        )
    raise TrainerError(
        f"no official HOT-Step trainer for {system!r}; set [train] trainer to a "
        "CUDA build made on Linux or Windows."
    )


def install_release(workspace: Path, variant: Optional[str] = None) -> Path:
    """Download the official prebuilt HOT-Step archive and return ``ace-train``.

    Nothing is compiled. ``engine/ace-train`` sits beside its libggml libraries,
    so the extracted folder is kept intact under ``trainer/``.
    """
    chosen = choose_variant(variant)
    asset = HOTSTEP_RELEASES[chosen]
    target = workspace / "trainer"
    target.mkdir(parents=True, exist_ok=True)
    eprint(f"[trainer] fetching prebuilt {asset.label}")
    archive = ensure_asset(asset, target)
    folder = target / archive_stem(asset.filename)
    if not folder.is_dir():
        eprint(f"[trainer] extracting into {folder}")
        extract_archive(archive, folder)
    found = None
    for name in (ACE_TRAIN, TRAINER_SHIPPED_AS):
        found = find_file(folder / "engine", name) or find_file(folder, name)
        if found:
            break
    if not found:
        raise TrainerError(f"{ACE_TRAIN} was not found in {archive}")
    if not is_windows():
        found.chmod(0o755)
    return found


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
