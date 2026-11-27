"""Describing a song by ear, the way YuE2-Studio's "按声音描述" does.

MOSS-Music-8B hears the recording through HOT-Step's ``ace-caption`` and writes
a five-field prose caption (``genre`` / ``caption`` / ``bpm`` / ``key`` /
``signature``). The one-line style that goes into ``<stem>.yue2.txt`` is then
composed from what it heard, exactly as the studio's no-assistant path
(``StyleState::Heard``) does:

    heard_style = normalize_yue2("<genre>. <caption>", bpm)

The prompt, the prose parser and the style normaliser are copied verbatim from
YuE2-Studio (``crates/music-server/src/listen.rs`` and ``prepare.rs``); the
``ace-caption`` command line matches ``listen::hear_batch``.

Notes:
* The studio replaces MOSS's guessed tempo with Beat This! (an ONNX model run
  through the Rust ``ort`` crate). A pure-stdlib Python CLI cannot run ONNX, so
  the tempo here is the one MOSS states in its ``bpm:`` field.
* The studio's writing assistant (a llama.cpp sidecar) rewrites the caption
  into a tighter sentence. Without it the studio keeps what MOSS heard, which
  is what this module does.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .assets import (
    LISTEN_FILES,
    MOSS_LM,
    TRAINER_COMMIT,
    TRAINER_FOLDER,
    TRAINER_REPOSITORY,
)
from .download import ensure_asset, find_file
from .util import eprint, find_on_path, is_windows, platform_key

#: ``ace-caption`` is the captioner binary shipped beside the trainer.
CAPTIONER = "ace-caption.exe" if is_windows() else "ace-caption"

#: How many seconds of a song the captioner hears. The studio's processor
#: default; a longer song is heard from its middle.
DEFAULT_SECONDS = 120

#: ``captionPrompt.ts`` CAPTION_INSTRUCTIONS, with the genre line first, exactly
#: as HOT-Step / YuE2-Studio send it.
PROSE_PROMPT = (
    "Write music dataset metadata grounded in the song's audible content. If audio is attached, "
    "describe what you actually HEAR and use title, artist, and lyrics only as weak secondary context.\n"
    "Return EXACTLY 5 lines in plain text and nothing else. Each field must start at the beginning of "
    "its own new line. Never place two fields on the same line.\n"
    "Use this exact output template:\n"
    "genre: <comma-separated genre/style tags, most specific first>\n"
    "caption: <2 to 4 sentences on one line>\n"
    "bpm: <estimated BPM as integer, e.g. 120>\n"
    "key: <note plus lowercase mode, e.g. 'C minor' or 'F# major'>\n"
    "signature: <numerator only — one of 2, 3, 4, 6>\n"
    "Caption rules:\n"
    "- Line 1 is REQUIRED and must begin with `genre:`. Line 2 is REQUIRED and must begin with "
    "`caption:` followed by the description. Never omit it, never leave it blank, and never answer "
    "with the metadata fields alone. If you are unsure of everything else, still write the caption.\n"
    "- The caption is 2 to 4 sentences, roughly 25 to 60 words, on a single line. Reference captions "
    "average about 30 words; a longer caption is not a better one, and padding it with invented "
    "detail is worse than stopping.\n"
    "- Cover these, woven into flowing description rather than listed:\n"
    "    - genre and subgenre, named plainly\n"
    "    - the instruments actually present, named concretely\n"
    "    - vocal character (or state that the track is instrumental and name what carries the lead line)\n"
    "    - mood and atmosphere\n"
    "    - production style and sonic character\n"
    "- NEVER state BPM, key, or time signature in the caption text. They have dedicated fields below, "
    "and repeating them in the caption does not match how this model was trained.\n"
    "- The genre you name in the caption MUST agree with the `genre:` field. Contradicting yourself "
    "between the two is worse than naming neither.\n"
    "- Describe the WHOLE track, weighted by how much of it each part occupies. Intros, outros, "
    "interludes and breakdowns are often unrepresentative: a quiet piano or percussion intro before a "
    "heavy song does not make the song ambient or classical. Base the genre and caption on the style "
    "that dominates most of the running time; mention a contrasting intro or outro only as a secondary "
    "detail.\n"
    "- Name things concretely: `808 bass`, `brushed snare`, `detuned saw lead`, `palm-muted guitar`, "
    "`upright piano` — not `interesting textures` or `lush soundscapes`.\n"
    "- No vague imagery or stacked adjectives ('neon skies, electric hearts'), no marketing copy, and "
    "no listener-reaction language ('keeps you moving', 'emotionally resonant').\n"
    "- Avoid generic openings like 'This track is' when more specific wording can be used immediately.\n"
    "- If the track is instrumental, say so and name the instrument carrying the lead line.\n"
    "- Start `genre:` on line 1, `caption:` on line 2, `bpm:` on line 3, `key:` on line 4, and "
    "`signature:` on line 5.\n"
    "- Do not merge fields together. For example, do not output `genre: ... bpm: ... key: ...` on one line.\n"
    "- Do not use markdown, bullets, numbering, code fences, labels before the template, or commentary "
    "after the template.\n"
    "- Do not mention the artist name or song title in the caption.\n"
    "- If audio is not attached or a field cannot be determined from available evidence, write N/A for "
    "that field instead of guessing."
)

_LABEL_LINE = re.compile(r"^\s*(genre|caption|bpm|key|signature)\s*:\s*(.*)$", re.IGNORECASE)
_FENCE = re.compile(r"(?s)```(?:[a-z]*)\n(.*?)```")
_LEADING_LABEL = re.compile(r"(?i)^\s*(caption|yue2 caption|style)\s*:\s*")
_WS = re.compile(r"\s+")
_TAIL_BPM = re.compile(r"(?i),?\s*(?:at\s+|around\s+|~\s*)?\d{2,3}\s*bpm\.?\s*$")
_BPM_WORD = re.compile(r"\b(\d{2,3})(\s*)(BPM|bpm)\b")


class CaptionError(RuntimeError):
    pass


@dataclass
class Heard:
    """What MOSS heard in one recording."""

    genre: str = ""
    caption: str = ""
    bpm: int = 0

    @property
    def style(self) -> str:
        return heard_style(self.genre, self.caption, self.bpm)


def _is_na(value: str) -> bool:
    return value.strip().lower() == "n/a" or not value.strip()


def parse_prose(text: str) -> Dict[str, str]:
    """Split the captioner's five-line prose into its fields."""
    fields: Dict[str, str] = {}
    last: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.lstrip("*- ")
        match = _LABEL_LINE.match(line)
        if match:
            key = match.group(1).lower()
            value = match.group(2).strip().strip("*").strip()
            fields[key] = value
            last = key
        elif last is not None:
            fields[last] = (fields.get(last, "") + " " + line).strip()
    return fields


def correct_facts_in_prose(text: str, bpm: int) -> str:
    """Replace any stated BPM with the measured one (studio: Beat This!)."""
    if bpm <= 0:
        return text
    return _BPM_WORD.sub(lambda m: f"{bpm}{m.group(2)}{m.group(3)}", text)


def normalize_yue2(raw: str, bpm: int) -> str:
    """Turn a raw caption into the single line the YuE2 planner reads."""
    fenced = _FENCE.search(raw)
    text = fenced.group(1) if fenced else raw
    text = _LEADING_LABEL.sub("", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("**", "")
    text = text.strip().strip("\"'“”‘’").strip()
    text = _WS.sub(" ", text)
    text = _TAIL_BPM.sub("", text)
    text = text.rstrip(".,; ")
    if bpm > 0:
        return f"{text}, {bpm} BPM"
    return text


def heard_style(genre: str, caption: str, bpm: int) -> str:
    """The style line, genre tags first, closed with the tempo."""
    genre = genre.strip()
    caption = caption.strip()
    if not genre:
        heard = caption
    elif not caption:
        heard = genre
    else:
        heard = f"{genre}. {caption}"
    return normalize_yue2(heard, bpm)


def from_prose(text: str) -> Heard:
    """Read a captioner run into a :class:`Heard`, or fail like the studio."""
    fields = parse_prose(text)
    genre = fields.get("genre", "")
    genre = "" if _is_na(genre) else genre
    caption = fields.get("caption", "")
    caption = "" if _is_na(caption) else caption
    if not caption:
        raise CaptionError("the captioner wrote no description")
    bpm = 0
    raw_bpm = fields.get("bpm", "")
    digits = re.search(r"\d{2,3}", raw_bpm)
    if digits:
        bpm = int(digits.group(0))
    return Heard(genre=genre, caption=caption, bpm=bpm)


# -- the captioner binary --------------------------------------------------


def locate(workspace: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    """Find ``ace-caption`` without building or downloading anything."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    from_env = None
    import os

    for key in ("YUE_CAPTION_BIN", "YUE2_CAPTION_BIN"):
        value = os.environ.get(key)
        if value and Path(value).is_file():
            from_env = Path(value)
            break
    if from_env:
        return from_env
    trainer_dir = workspace / "trainer"
    for name in (CAPTIONER, "ace-caption"):
        direct = trainer_dir / name
        if direct.is_file():
            return direct
    if trainer_dir.is_dir():
        found = find_file(trainer_dir, CAPTIONER) or find_file(trainer_dir, "ace-caption")
        if found:
            return found
    on_path = find_on_path("ace-caption") or find_on_path("ace-caption.exe")
    return Path(on_path) if on_path else None


def install(workspace: Path, build: bool = False) -> Path:
    """Return a usable captioner, downloading or building one when needed."""
    found = locate(workspace)
    if found:
        return found
    system = platform_key()
    if system == "windows":
        return install_prebuilt(workspace)
    if build or system in ("linux", "macos"):
        return build_from_source(workspace)
    raise CaptionError(
        "no ace-caption binary found; build HOT-Step's ace-caption on this machine "
        "and set [dataset] captioner to its path"
    )


def install_prebuilt(workspace: Path) -> Path:
    """Extract ``ace-caption.exe`` from the pinned HOT-Step tools archive."""
    from . import trainer

    trainer.install_prebuilt(workspace)  # same zip, same folder
    folder = workspace / "trainer" / TRAINER_FOLDER
    found = find_file(folder, CAPTIONER)
    if not found:
        raise CaptionError(f"{CAPTIONER} was not found in the HOT-Step tools archive")
    return found


def build_from_source(workspace: Path, jobs: Optional[int] = 1) -> Path:
    """Build ``ace-caption`` from the pinned HOT-Step commit (Linux / macOS).

    Defaults to one compile job at a time so the build's peak memory stays at
    a single translation unit; pass a larger ``jobs`` to trade memory for speed.
    """
    source = workspace / "trainer" / "HOT-Step-CPP"
    build_dir = workspace / "trainer" / "build-caption"
    if not source.is_dir():
        if not find_on_path("git"):
            raise CaptionError("git is required to build the captioner")
        eprint(f"[caption] cloning {TRAINER_REPOSITORY} at {TRAINER_COMMIT[:8]}")
        subprocess.run(["git", "clone", TRAINER_REPOSITORY, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", TRAINER_COMMIT], check=True)
        subprocess.run(
            ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
    if not find_on_path("cmake"):
        raise CaptionError("cmake is required to build the captioner")
    build_dir.mkdir(parents=True, exist_ok=True)
    backend = ["-DGGML_METAL=ON"] if platform_key() == "macos" else ["-DGGML_CUDA=ON",
                                                                   "-DCMAKE_CUDA_ARCHITECTURES=native"]
    eprint("[caption] configuring ace-caption")
    subprocess.run(
        ["cmake", "-S", str(source / "engine"), "-B", str(build_dir),
         "-DCMAKE_BUILD_TYPE=Release"] + backend,
        check=True,
    )
    eprint("[caption] building ace-caption")
    command = ["cmake", "--build", str(build_dir), "--target", "ace-caption"]
    command += ["--parallel", str(jobs)] if jobs else ["--parallel"]
    subprocess.run(command, check=True)
    found = find_file(build_dir, CAPTIONER)
    if not found:
        raise CaptionError("the build finished but ace-caption was not found")
    destination = workspace / "trainer" / CAPTIONER
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(found.read_bytes())
    if not is_windows():
        destination.chmod(0o755)
    return destination


# -- the captioner models --------------------------------------------------


def models_dir(workspace: Path) -> Path:
    return workspace / "models" / "moss"


def missing_models(directory: Path) -> List[str]:
    return [asset.filename for asset in LISTEN_FILES if not (directory / asset.filename).is_file()]


def ensure_models(directory: Path) -> None:
    """Download MOSS-Music's weights into ``directory`` when missing."""
    directory.mkdir(parents=True, exist_ok=True)
    for asset in LISTEN_FILES:
        ensure_asset(asset, directory)


# -- running the captioner -------------------------------------------------


def _probe_seconds(path: Path, ffmpeg: Optional[str]) -> Optional[float]:
    ffprobe = find_on_path("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def _prepare_audio(source: Path, target: Path, seconds: int, ffmpeg: Optional[str]) -> Path:
    """A 16 kHz mono WAV of the song (its middle when it is longer than ``seconds``)."""
    if not ffmpeg:
        return source
    duration = _probe_seconds(source, ffmpeg)
    command = [ffmpeg, "-nostdin", "-y", "-loglevel", "error"]
    if duration is not None and duration > seconds:
        start = (duration - seconds) / 2.0
        command += ["-ss", f"{start:.2f}", "-t", str(seconds)]
    command += ["-i", str(source), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target)]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return source
    return target


def describe(
    songs: Iterable[Path],
    binary: Path,
    moss: Path,
    ffmpeg: Optional[str] = None,
    seconds: int = DEFAULT_SECONDS,
) -> Dict[Path, Heard]:
    """Hear every song once; return ``{audio path: Heard}``.

    The model loads once for the whole batch, the way the studio labels a
    dataset. Each output is read back from its per-song file.
    """
    songs = [Path(song) for song in songs]
    if not songs:
        return {}
    with tempfile.TemporaryDirectory(prefix="yue2-hear-") as work:
        work_path = Path(work)
        prompt = work_path / "prompt.prose.txt"
        prompt.write_text(PROSE_PROMPT, encoding="utf-8")

        inputs: List[str] = []
        sources: List[Path] = []
        for index, song in enumerate(songs):
            wav = _prepare_audio(song, work_path / f"{index}.source.wav", seconds, ffmpeg)
            inputs.append(f"{wav}\t{work_path / str(index)}")
            sources.append(song)
        listing = work_path / "songs.tsv"
        listing.write_text("\n".join(inputs) + "\n", encoding="utf-8")

        command = [
            str(binary),
            "--models", str(moss),
            "--lm", str(moss / MOSS_LM.filename),
            "--max-seconds", str(seconds),
            "--src-list", str(listing),
            "--mode", "prose",
            "--temperature", "0",
            "--rep-penalty", "1.0",
            "--freq-penalty", "0.3",
            "--prompt-file", f"prose={prompt}",
        ]
        if ffmpeg:
            command += ["--ffmpeg", ffmpeg]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise CaptionError(f"the captioner failed: {error}") from error

        heard: Dict[Path, Heard] = {}
        for index, song in enumerate(sources):
            base = work_path / str(index)
            output = base if base.is_file() else base.with_name(f"{index}.prose.txt")
            if not output.is_file():
                raise CaptionError(f"the captioner wrote nothing for {song.name}")
            heard[song] = from_prose(output.read_text(encoding="utf-8"))
        return heard


def styles_for(
    songs: Iterable[Path],
    binary: Path,
    moss: Path,
    ffmpeg: Optional[str] = None,
    seconds: int = DEFAULT_SECONDS,
) -> Dict[str, str]:
    """One style line per song, keyed by file name (what ``dataset`` writes)."""
    heard = describe(songs, binary, moss, ffmpeg, seconds)
    return {path.name: result.style for path, result in heard.items()}
