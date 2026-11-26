"""The training pipeline, stage by stage.

A translation of ``training_stages`` from YuE2-Studio's ``yue_train.rs``: the
same subcommands, the same flags and the same order. A dataset reaches the
trainer as a folder of 48 kHz WAV files, each with a ``<stem>.txt`` sidecar
(``caption:``, ``is_instrumental:``, ``lyrics:`` last) and a ``<stem>.yue2.txt``
holding the one-line style the planner is prompted with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .assets import TRAINING_FILES
from .recipe import Recipe, train_args

_VAE = next(asset for asset in TRAINING_FILES if asset.id == "yue2-vae-encoder").filename
_SEMANTIC = next(asset for asset in TRAINING_FILES if asset.id == "yue2-semantic-tokenizer").filename
_SHEETSAGE = next(asset for asset in TRAINING_FILES if asset.id == "yue2-sheetsage").filename
_ALIGNER = next(asset for asset in TRAINING_FILES if asset.id == "yue2-lyric-aligner").filename


@dataclass
class Stage:
    """One trainer invocation."""

    id: str
    args: List[str] = field(default_factory=list)


@dataclass
class TrainingInputs:
    audio: Path
    models: Path
    checkpoint: Path
    tokenizer: Path
    companion: Path
    run: Path
    vocals: Path
    trigger: str
    recipe: Recipe
    device: str = "CUDA0"


def sidecars(style: str, lyrics: str, instrumental: bool) -> "list[tuple[str, str]]":
    """The two sidecars a song needs beside its audio."""
    normalized = " ".join(style.split())
    return [
        (
            ".txt",
            f"caption: {normalized}\nis_instrumental: {str(instrumental).lower()}\nlyrics:\n{lyrics.strip()}\n",
        ),
        (".yue2.txt", f"{normalized}\n"),
    ]


def training_stages(inputs: TrainingInputs) -> List[Stage]:
    models = inputs.models
    cache = inputs.run / "cache"
    manifest = cache / "yue2_preprocess.json"
    prepared = inputs.run / "prepared"
    recipe = inputs.recipe

    def model(name: str, filename: str) -> str:
        return f"{name}={models / filename}"

    latents = Stage(
        "latents",
        [
            "yue2-preprocess",
            "--audio", str(inputs.audio),
            "--out", str(cache),
            "--models", str(models),
            "--vae", "standard",
            "--caption-mode", "yue2",
            # the recordings as mastered: the base was trained on them
            "--loudness-lufs", "0",
        ],
    )
    codes = Stage("codes", ["yue2-tokenize", "--manifest", str(manifest), "--models", str(models)])
    align = Stage(
        "align",
        [
            "yue2-align",
            "--manifest", str(manifest),
            "--mmsfa", str(models / _ALIGNER),
            "--stems", str(inputs.vocals),
        ],
    )
    scores = Stage("scores", ["yue2-sheet", "--manifest", str(manifest), "--models", str(models)])

    prepare_args = [
        "yue2-prepare-aitk",
        "--legacy-manifest", str(manifest),
        "--checkpoint", str(inputs.checkpoint),
        "--tokenizer", str(inputs.tokenizer),
        "--output", str(prepared),
        "--model", model("vae", _VAE),
        "--model", model("semantic", _SEMANTIC),
        "--model", model("sheetsage", _SHEETSAGE),
        "--lyric-timing", "1" if recipe.lyric_timing else "0",
    ]
    if inputs.trigger.strip():
        prepare_args += ["--trigger", inputs.trigger.strip()]
    prepare = Stage("prepare", prepare_args)

    train = Stage(
        "train",
        train_args(
            inputs.checkpoint,
            inputs.companion,
            prepared / "dataset.json",
            recipe,
            inputs.run / "output",
            inputs.device,
        ),
    )

    stages = [latents, codes, align, scores, prepare, train]
    if not recipe.lyric_timing:
        stages = [stage for stage in stages if stage.id != "align"]
    return stages


def continuation_stage(
    inputs: TrainingInputs,
    output: Path,
    resume: Path,
) -> Stage:
    """A finished or stopped run trained further, up to ``recipe.steps``."""
    from .recipe import continuation_args

    return Stage(
        "train",
        continuation_args(
            inputs.checkpoint,
            inputs.companion,
            inputs.run / "prepared" / "dataset.json",
            inputs.recipe,
            output,
            resume,
            inputs.device,
        ),
    )
