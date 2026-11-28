"""Running the trainer, stage by stage, with progress from its JSON lines."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .stages import Stage
from .util import eprint, gpu_env, quote


class StageFailed(RuntimeError):
    def __init__(self, stage: str, returncode: int) -> None:
        super().__init__(f"stage {stage!r} exited with code {returncode}")
        self.stage = stage
        self.returncode = returncode


def _step(line: str) -> Optional[dict]:
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if value.get("stage") != "joint":
        return None

    def number(key: str) -> Optional[float]:
        raw = value.get(key)
        return float(raw) if isinstance(raw, (int, float)) else None

    ar_ce = number("ar_ce")
    ar_kl = number("ar_kl")
    nar_mse = number("nar_mse")
    step = value.get("step")
    if ar_ce is None or ar_kl is None or nar_mse is None or step is None:
        return None
    return {
        "step": int(step),
        "loss": ar_ce + 0.2 * ar_kl + nar_mse,
        "ar_kl": ar_kl,
    }


def run_stage(trainer: Path, stage: Stage, total_steps: Optional[int] = None) -> None:
    argv = [str(trainer), *stage.args]
    eprint(f"[{stage.id}] {quote(argv)}")
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=gpu_env(trainer),
    )
    assert process.stdout is not None
    last_progress = False
    try:
        for line in process.stdout:
            progress = _step(line)
            if progress is not None and stage.id == "train":
                total = f"/{total_steps}" if total_steps else ""
                sys.stderr.write(
                    f"\r[{stage.id}] step {progress['step']}{total}  "
                    f"loss {progress['loss']:.4f}  ar_kl {progress['ar_kl']:.4f}"
                )
                sys.stderr.flush()
                last_progress = True
                continue
            if last_progress:
                sys.stderr.write("\n")
                last_progress = False
            eprint(f"[{stage.id}] {line.rstrip()}")
        returncode = process.wait()
    except KeyboardInterrupt:
        eprint(f"\n[{stage.id}] interrupted; stopping the trainer")
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    if last_progress:
        sys.stderr.write("\n")
    if returncode != 0:
        raise StageFailed(stage.id, returncode)


def run_stages(trainer: Path, stages: Iterable[Stage], total_steps: Optional[int] = None) -> None:
    for stage in stages:
        run_stage(trainer, stage, total_steps=total_steps)


def print_stages(trainer: Optional[Path], stages: List[Stage]) -> None:
    binary = str(trainer) if trainer else "ace-train"
    for stage in stages:
        print(f"# {stage.id}")
        print(quote([binary, *stage.args]))
        print()
