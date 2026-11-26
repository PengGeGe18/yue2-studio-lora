"""The pinned model and trainer assets a YuE2 LoRA run needs.

Everything here is copied from YuE2-Studio's ``yue_train`` / ``training`` /
``model_manager`` modules: the Hugging Face repository and commit the training
weights come from, the flat file names the trainer looks for, and the decoder
companion the studio freezes under every LoRA it trains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

HUGGING_FACE = "https://huggingface.co"

#: YuE2 training pack (ConvRot base + VAE + semantic tokenizer + SheetSage2 + MMS aligner).
WEIGHTS_REPOSITORY = "scragnog/YuE2-GGUF"
WEIGHTS_REVISION = "eb7de0903bf4dfbd14a2384ae0c0d14150cfa2c7"

#: The decoder adapter every render merges, frozen under the LoRA being trained.
COMPANION_REPOSITORY = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
COMPANION_REVISION = "e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5"

#: A YuE2 generation GGUF, read only for its text tokenizer during ``yue2-prepare-aitk``.
TOKENIZER_REPOSITORY = "Serveurperso/YuE2-GGUF"
TOKENIZER_REVISION = "e630f2b8f6aedf8254c3186438633c1b5eeaf80a"

#: MOSS-Music-8B, the local captioner behind YuE2-Studio's "按声音描述".
#: HOT-Step's native GGML port; the studio pins this exact revision.
MOSS_REPOSITORY = "scragnog/MOSS-Music-8B-Instruct-GGUF"
MOSS_REVISION = "9fe86435bd542bf29cbf4984e9e0ab0ea8ca7e00"

#: Prebuilt trainer release from YuE2-Studio (Windows x64 + CUDA).
TRAINER_RELEASE_TAG = "hotstep-tools-479cbeb8-xcr0"
TRAINER_ASSET = "hotstep-tools-479cbeb8-xcr0-cuda-windows-x64.zip"
TRAINER_ASSET_BYTES = 125_490_974
TRAINER_SHIPPED_AS = "music-train.exe"
TRAINER_REPOSITORY = "https://github.com/timoncool/HOT-Step-CPP.git"
TRAINER_COMMIT = "479cbeb8f5e7de98f85f2ac8c3da305cb73839e8"
TRAINER_FOLDER = TRAINER_RELEASE_TAG

#: Video memory the default recipe needs, in GB.
MIN_VRAM_GB = 11


@dataclass(frozen=True)
class Asset:
    """One downloadable file."""

    id: str
    label: str
    filename: str
    url: str
    bytes: int
    sha256: Optional[str] = None


def _huggingface(repo: str, revision: str, source: str) -> str:
    return f"{HUGGING_FACE}/{repo}/resolve/{revision}/{source}"


def _training(id: str, label: str, source: str, filename: str, size: int,
              sha256: Optional[str] = None) -> Asset:
    return Asset(
        id=id,
        label=label,
        filename=filename,
        url=_huggingface(WEIGHTS_REPOSITORY, WEIGHTS_REVISION, source),
        bytes=size,
        sha256=sha256,
    )


#: The five files of the training pack, stored flat in the models folder.
TRAINING_FILES = [
    _training(
        "yue2-convrot-base",
        "YuE2 base for training (INT8 ConvRot)",
        "checkpoints/yue2_3b_int8_convrot.safetensors",
        "yue2_3b_int8_convrot.safetensors",
        3_960_938_800,
        # Published SHA-256 checked by YuE2-Studio's resolve_training_base.
        "96fe199377309001ed8cd26a944baeee8cc31a20ba7c36d1d3c0a7e1f4149db6",
    ),
    _training(
        "yue2-vae-encoder",
        "YuE2 VAE with its encoder",
        "yue2-vae-standard-f32.gguf",
        "yue2-vae-standard-f32.gguf",
        530_348_768,
    ),
    _training(
        "yue2-semantic-tokenizer",
        "Semantic tokenizer",
        "yue2-tok-f16.gguf",
        "yue2-tok-f16.gguf",
        1_211_632_224,
    ),
    _training(
        "yue2-sheetsage",
        "SheetSage2 score transcriber",
        "sheetsage2-f16.gguf",
        "sheetsage2-f16.gguf",
        1_360_020_736,
    ),
    _training(
        "yue2-lyric-aligner",
        "MMS forced aligner (lyric timing)",
        "mms-fa/mms-fa-f32.gguf",
        "mms-fa-f32.gguf",
        1_261_892_736,
    ),
]

#: The decoder companion frozen under the LoRA.
COMPANION = Asset(
    id="yue2-companion-v9",
    label="Decoder companion (nar_lora_joint_v9)",
    filename="nar_lora_joint_v9.safetensors",
    url=_huggingface(COMPANION_REPOSITORY, COMPANION_REVISION, "nar_lora_joint_v9.safetensors"),
    bytes=140_560_592,
    sha256="585f303da1d5252d228d1e8ac6d4c4d11d970df9297406935cc8bdafa49cfa7e",
)

#: The YuE2 GGUF read for its text tokenizer. Q5_K_M keeps the download small;
#: only the vocabulary is used.
TOKENIZER = Asset(
    id="yue2-tokenizer",
    label="YuE2 generation GGUF (text tokenizer)",
    filename="YuE2-3B-Q5_K_M.gguf",
    url=_huggingface(TOKENIZER_REPOSITORY, TOKENIZER_REVISION, "YuE2-3B-Q5_K_M.gguf"),
    bytes=2_622_936_832,
    sha256="cd3efd250b734a229172800f08f33a893ad1f66b521666e176cdae4a0729b281",
)

#: The prebuilt trainer archive (Windows only). The same archive also ships
#: ``ace-caption.exe``, the MOSS captioner that describes a song by ear.
TRAINER = Asset(
    id="music-train",
    label=f"Trainer (HOT-Step {TRAINER_COMMIT[:8]})",
    filename=TRAINER_ASSET,
    url=f"https://github.com/timoncool/YuE2-Studio/releases/download/{TRAINER_RELEASE_TAG}/{TRAINER_ASSET}",
    bytes=TRAINER_ASSET_BYTES,
)

#: MOSS-Music-8B weights the captioner reads, stored flat in ``models/moss``.
MOSS_AUDIO = Asset(
    id="moss-audio",
    label="MOSS-Music audio tower (F16)",
    filename="moss-aud-f16.gguf",
    url=_huggingface(MOSS_REPOSITORY, MOSS_REVISION, "moss-aud-f16.gguf"),
    bytes=1_724_480_640,
)

MOSS_LM = Asset(
    id="moss-lm",
    label="MOSS-Music-8B language model (Q8_0)",
    filename="moss-lm-q8_0.gguf",
    url=_huggingface(MOSS_REPOSITORY, MOSS_REVISION, "moss-lm-q8_0.gguf"),
    bytes=8_709_513_792,
)

#: What the captioner needs to listen to a song.
LISTEN_FILES = [MOSS_LM, MOSS_AUDIO]


def listen_assets() -> list[Asset]:
    return list(LISTEN_FILES)




def all_models(include_tokenizer: bool = True) -> list[Asset]:
    assets = list(TRAINING_FILES) + [COMPANION]
    if include_tokenizer:
        assets.append(TOKENIZER)
    return assets


def asset_by_id(identifier: str) -> Optional[Asset]:
    for asset in all_models():
        if asset.id == identifier:
            return asset
    return None
