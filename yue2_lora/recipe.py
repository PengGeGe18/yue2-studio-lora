"""Training recipes and the arguments they turn into.

This is a direct translation of YuE2-Studio's ``crates/music-engine/src/yue_train.rs``:
the ``Recipe`` fields, the three base-matched presets (``fast`` / ``balanced`` /
``thorough``) plus the older ``tuned`` recipe, the forced arguments of each, and
the ``yue2-joint-train`` command line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The base-matched sizes: updates, songs per update, checkpoint every.
PRESETS = {
    "fast": (100, 4, 10),
    "balanced": (200, 4, 20),
    "thorough": (300, 8, 30),
}

#: The decoder window of a base-matched run, 60 s.
BASE_MATCHED_DECODER_CROP = "1500"

#: The adapter every base-matched preset trains: LoKr dim 128, factor 4, alpha 256.
BASE_MATCHED_LOKR = (128, 4, 256.0)

_KNOWN_PRESETS = set(PRESETS) | {"tuned"}


@dataclass
class Recipe:
    """What a run is asked for."""

    preset: str = "balanced"
    stop: str = "kl"
    epochs: int = 20
    steps: int = 750
    save_every: int = 50
    seed: int = 42
    target_kl: float = 1.4
    adapter: str = "lokr"
    rank: int = 64
    alpha: float = 256.0
    lokr_dim: int = 64
    lokr_factor: int = 4
    optimizer: str = "prodigy"
    learning_rate: float = 2e-4
    planner_lr_scale: float = 0.3
    lyric_timing: bool = True
    cursor_weight: float = 0.08
    grad_accum: int = 1
    warmup: int = 0

    # -- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "Recipe":
        recipe = cls()
        for key, value in (values or {}).items():
            if not hasattr(recipe, key):
                raise ValueError(f"unknown recipe setting {key!r}")
            setattr(recipe, key, value)
        return recipe

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- presets ----------------------------------------------------------

    def for_songs(self, songs: int) -> "Recipe":
        """The recipe the trainer runs for a dataset of ``songs`` songs."""
        recipe = Recipe.from_dict(self.to_dict())
        if recipe.preset in PRESETS:
            updates, songs_per_update, save_every = PRESETS[recipe.preset]
            recipe.steps = updates
            recipe.grad_accum = songs_per_update
            recipe.save_every = save_every
            recipe.adapter = "lokr"
            recipe.lokr_dim, recipe.lokr_factor, recipe.alpha = BASE_MATCHED_LOKR
            recipe.target_kl = 0.0
            recipe.lyric_timing = False
            recipe.cursor_weight = 0.0
            recipe.warmup = max(1, int(round(updates * 0.03)))
            return recipe
        recipe.grad_accum = 1
        if recipe.stop == "epochs":
            recipe.steps = recipe.epochs * max(1, songs)
            recipe.target_kl = 0.0
        return recipe

    def base_matched(self) -> bool:
        return self.preset != "tuned"

    # -- validation -------------------------------------------------------

    def check(self) -> None:
        if self.steps == 0 or self.save_every == 0 or self.epochs == 0:
            raise ValueError("steps, epochs and the checkpoint interval must be at least 1")
        if self.preset not in _KNOWN_PRESETS:
            raise ValueError(f"unknown training preset {self.preset}")
        if self.grad_accum == 0:
            raise ValueError("songs per update must be at least 1")
        if self.stop not in ("kl", "epochs"):
            raise ValueError(f"unknown stopping rule {self.stop}")
        if self.adapter not in ("lokr", "lora"):
            raise ValueError(f"unknown adapter type {self.adapter}")
        if self.optimizer not in ("prodigy", "adamw"):
            raise ValueError(f"unknown optimizer {self.optimizer}")
        numbers = [
            self.target_kl,
            self.alpha,
            self.learning_rate,
            self.planner_lr_scale,
            self.cursor_weight,
        ]
        if any(value != value or value in (float("inf"), float("-inf")) for value in numbers):
            raise ValueError("a recipe number is not finite")
        if (
            self.target_kl < 0.0
            or self.alpha <= 0.0
            or self.learning_rate <= 0.0
            or self.planner_lr_scale <= 0.0
            or not (0.0 <= self.cursor_weight <= 10.0)
        ):
            raise ValueError("a recipe number is out of range")
        if self.rank == 0 or self.lokr_dim == 0 or self.lokr_factor == 0:
            raise ValueError("rank, LoKr dimension and factor must be at least 1")


def train_args(
    checkpoint: Path,
    companion: Path,
    prepared_dataset: Path,
    recipe: Recipe,
    output: Path,
    device: str = "CUDA0",
) -> List[str]:
    """The ``yue2-joint-train`` arguments for one run."""
    args: List[str] = [
        "yue2-joint-train",
        "--checkpoint", str(checkpoint),
        "--companion", str(companion),
        "--dataset", str(prepared_dataset),
        "--output", str(output),
        "--steps", str(recipe.steps),
        "--save-every", str(max(1, recipe.save_every)),
        "--seed", str(recipe.seed),
        "--device", device,
        "--rank", str(recipe.rank),
        "--alpha", _number(recipe.alpha),
        "--adapter-type", recipe.adapter,
    ]
    if recipe.adapter == "lokr":
        args += ["--lokr-dim", str(recipe.lokr_dim), "--lokr-factor", str(recipe.lokr_factor)]
    if recipe.base_matched():
        args += [
            "--optimizer", "adamw-lm",
            "--lr", "1e-4",
            "--weight-decay", "0.1",
            "--beta1", "0.9",
            "--beta2", "0.95",
            "--kl-weight", "0",
            "--caption-dropout", "0",
            "--abc-dropout", "0.5",
            "--planner-lr-scale", "1",
            "--lr-schedule", "cosine-floor",
            "--lr-floor", "0.1",
            "--ar-loss-weight", "0.25",
            "--ar-targets", "base",
            "--nar-crop-frames", BASE_MATCHED_DECODER_CROP,
            "--text-dropout", "0.1",
            "--lyric-dropout", "0.1",
            "--both-dropout", "0.1",
            "--cursor-weight", "0",
            "--warmup", str(max(1, recipe.warmup)),
            "--grad-accum", str(max(1, recipe.grad_accum)),
        ]
        return args
    args += [
        "--optimizer", recipe.optimizer,
        "--planner-lr-scale", _number(recipe.planner_lr_scale),
        "--cursor-weight", _number(recipe.cursor_weight) if recipe.lyric_timing else "0",
    ]
    if recipe.optimizer == "prodigy":
        args += ["--prodigy-d0", "1e-6"]
    else:
        args += ["--lr", _number(recipe.learning_rate)]
    if recipe.target_kl > 0.0:
        args += ["--target-kl", _number(recipe.target_kl)]
    return args


def _number(value: float) -> str:
    """Format a float the way the Rust trainer prints it (shortest round-trip)."""

    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text


def continuation_args(
    checkpoint: Path,
    companion: Path,
    prepared_dataset: Path,
    recipe: Recipe,
    output: Path,
    resume: Path,
    device: str = "CUDA0",
) -> List[str]:
    args = train_args(checkpoint, companion, prepared_dataset, recipe, output, device)
    args += ["--resume", str(resume)]
    return args
