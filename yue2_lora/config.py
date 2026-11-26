"""Configuration loading: one file describes a training run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from . import _toml
from .assets import COMPANION, TOKENIZER, TRAINING_FILES
from .checkpoints import normalize_device
from .recipe import Recipe
from .util import replace_path


class ConfigError(ValueError):
    pass


_STOCK_BASE = next(asset for asset in TRAINING_FILES if asset.id == "yue2-convrot-base")


@dataclass
class Config:
    path: Path
    root: Path
    workspace: Path
    models_dir: Path
    runs_dir: Path
    trainer_dir: Path
    audio: Path
    vocals: Path
    trigger: str
    normalize: bool
    prepared_dir: Path
    captioner: Optional[Path]
    caption_seconds: int
    base: Path
    companion: Path
    tokenizer: Path
    device: str
    ffmpeg: Optional[str]
    trainer: Optional[Path]
    recipe: Recipe

    @property
    def training_audio(self) -> Path:
        """The folder a run reads: the prepared 48 kHz copy, or the source."""
        return self.prepared_dir if self.normalize else self.audio

    def describe(self) -> Dict[str, Any]:
        return {
            "config": str(self.path),
            "workspace": str(self.workspace),
            "dataset": {
                "audio": str(self.audio),
                "vocals": str(self.vocals),
                "trigger": self.trigger,
                "normalize": self.normalize,
                "prepared": str(self.prepared_dir),
                "captioner": str(self.captioner) if self.captioner else None,
                "caption_seconds": self.caption_seconds,
                "training_audio": str(self.training_audio),
            },
            "models": {
                "base": str(self.base),
                "companion": str(self.companion),
                "tokenizer": str(self.tokenizer),
            },
            "train": {
                "device": self.device,
                "ffmpeg": self.ffmpeg,
                "trainer": str(self.trainer) if self.trainer else None,
            },
            "recipe": self.recipe.to_dict(),
        }


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _optional_path(values: Dict[str, Any], key: str, root: Path) -> Optional[Path]:
    raw = values.get(key)
    if raw in (None, "", False):
        return None
    return replace_path(raw, root)


def load(path: Path) -> Config:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            try:
                import tomllib  # type: ignore[import-not-found]
            except ModuleNotFoundError:
                data = _toml.loads(text)
            else:
                data = tomllib.loads(text)
    except (ValueError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the config must be a table at the top level")

    root = path.parent
    workspace_section = _section(data, "workspace")
    workspace_dir = replace_path(workspace_section.get("dir", "."), root)
    models_dir = replace_path(workspace_section.get("models", "models"), workspace_dir)
    runs_dir = replace_path(workspace_section.get("runs", "runs"), workspace_dir)
    trainer_dir = replace_path(workspace_section.get("trainer", "trainer"), workspace_dir)

    dataset = _section(data, "dataset")
    audio = replace_path(dataset.get("audio", "dataset/audio"), root)
    vocals_default = audio.parent / "vocals"
    vocals = replace_path(dataset.get("vocals", vocals_default), root)
    trigger = str(dataset.get("trigger", "") or "").strip()
    normalize = bool(dataset.get("normalize", False))
    prepared_default = audio.parent / "prepared-audio"
    prepared_dir = replace_path(dataset.get("prepared", prepared_default), root)
    captioner = _optional_path(dataset, "captioner", root)
    try:
        caption_seconds = int(dataset.get("caption_seconds", 120) or 120)
    except (TypeError, ValueError):
        raise ConfigError("[dataset] caption_seconds must be a whole number")
    if caption_seconds <= 0:
        raise ConfigError("[dataset] caption_seconds must be positive")

    model = _section(data, "model")
    base = _optional_path(model, "base", root) or (models_dir / _STOCK_BASE.filename)
    companion = _optional_path(model, "companion", root) or (models_dir / COMPANION.filename)
    tokenizer = _optional_path(model, "tokenizer", root) or (models_dir / TOKENIZER.filename)

    train = _section(data, "train")
    device = normalize_device(str(train.get("device", "CUDA0")))
    ffmpeg_raw = train.get("ffmpeg", "ffmpeg")
    ffmpeg = None if str(ffmpeg_raw).strip().lower() in ("", "none", "off") else str(ffmpeg_raw)
    trainer = _optional_path(train, "trainer", root)

    recipe = Recipe.from_dict(_section(data, "recipe"))

    return Config(
        path=path,
        root=root,
        workspace=workspace_dir,
        models_dir=models_dir,
        runs_dir=runs_dir,
        trainer_dir=trainer_dir,
        audio=audio,
        vocals=vocals,
        trigger=trigger,
        normalize=normalize,
        prepared_dir=prepared_dir,
        captioner=captioner,
        caption_seconds=caption_seconds,
        base=base,
        companion=companion,
        tokenizer=tokenizer,
        device=device,
        ffmpeg=ffmpeg,
        trainer=trainer,
        recipe=recipe,
    )
