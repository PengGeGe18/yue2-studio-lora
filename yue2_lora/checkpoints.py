"""Checkpoints a run wrote, and where a stopped run continues from.

A translation of ``resume_point`` / ``checkpoints`` in YuE2-Studio's
``yue_train.rs``. The trainer writes ``output/checkpoint-step<N>/`` folders;
a continuation writes into its own ``output-<id>``.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MAGIC = b"Y2RSUME1"
HEADER_BYTES = 44
MAX_PAYLOAD = 16 << 30
MAX_METADATA = 16 << 20


@dataclass
class Checkpoint:
    step: int
    files: List[Path]


def _checkpoint_dirs(run: Path) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    if not run.is_dir():
        return found
    for entry in run.iterdir():
        name = entry.name
        if not entry.is_dir() or not (name == "output" or name.startswith("output-")):
            continue
        for checkpoint in entry.iterdir():
            if not checkpoint.is_dir():
                continue
            prefix = "checkpoint-step"
            if not checkpoint.name.startswith(prefix):
                continue
            try:
                step = int(checkpoint.name[len(prefix):])
            except ValueError:
                continue
            found.append((step, checkpoint))
    return found


def complete_resume_file(path: Path) -> bool:
    """Whether an ``optimizer.resume`` record is complete, without reading it all."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            header = handle.read(HEADER_BYTES)
    except OSError:
        return False
    if len(header) != HEADER_BYTES or header[:8] != MAGIC:
        return False
    version = struct.unpack("<I", header[8:12])[0]
    payload = struct.unpack("<Q", header[12:20])[0]
    metadata = struct.unpack("<Q", header[20:28])[0]
    return (
        1 <= version <= 3
        and 1 <= payload <= MAX_PAYLOAD
        and metadata <= MAX_METADATA
        and header[20:28] == header[36:44]
        and size == HEADER_BYTES + metadata + payload
    )


def normalize_device(device: str) -> str:
    """Check a ggml device name: CUDA0, Vulkan1, MTL0, CPU..."""
    text = (device or "").strip()
    if not text:
        raise ValueError("device cannot be empty")
    upper = text.upper()
    if upper == "CPU":
        return "CPU"
    for prefix, canonical in (("CUDA", "CUDA"), ("VULKAN", "Vulkan"), ("MTL", "MTL"), ("METAL", "MTL")):
        if upper.startswith(prefix):
            rest = text[len(prefix):]
            if rest.startswith(":"):
                rest = rest[1:]
            if not rest:
                return f"{canonical}0"
            if not rest.isdigit():
                raise ValueError(f"invalid device {device!r}; use e.g. CUDA0, Vulkan1, MTL0 or CPU")
            return f"{canonical}{int(rest)}"
    raise ValueError(f"invalid device {device!r}; use e.g. CUDA0, Vulkan1, MTL0 or CPU")


def resume_point(run: Path) -> Optional[Tuple[int, Path]]:
    """The latest checkpoint that kept its optimizer state."""
    checkpoints = _checkpoint_dirs(run)

    def newest(path: Path) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    checkpoints.sort(
        key=lambda item: (item[0], newest(item[1] / "optimizer.resume")), reverse=True
    )
    for step, directory in checkpoints:
        state = directory / "optimizer.resume"
        if complete_resume_file(state):
            return step, state
    return None


def _native_export(directory: Path, half: str) -> Optional[Path]:
    native = directory / f"native-{half}.safetensors"
    if native.is_file():
        return native
    needle = f"_joint_{half}_step"
    candidates = [
        entry for entry in directory.iterdir()
        if entry.is_file() and entry.name.endswith(".safetensors") and needle in entry.name
    ] if directory.is_dir() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def checkpoints(run: Path) -> List[Checkpoint]:
    """The run's checkpoints, latest first, with both native exports present."""
    found: List[Checkpoint] = []
    for step, directory in _checkpoint_dirs(run):
        ar = _native_export(directory, "ar")
        nar = _native_export(directory, "nar")
        if ar is None or nar is None:
            continue
        found.append(Checkpoint(step=step, files=[ar, nar]))
    found.sort(key=lambda checkpoint: (checkpoint.step, checkpoint.files[0].stat().st_mtime), reverse=True)
    unique: List[Checkpoint] = []
    seen = set()
    for checkpoint in found:
        if checkpoint.step in seen:
            continue
        seen.add(checkpoint.step)
        unique.append(checkpoint)
    return unique


def next_output_dir(run: Path) -> Path:
    """``output`` for the first run, ``output-<n>`` for each continuation."""
    first = run / "output"
    if not first.exists():
        return first
    index = 2
    while (run / f"output-{index}").exists():
        index += 1
    return run / f"output-{index}"
