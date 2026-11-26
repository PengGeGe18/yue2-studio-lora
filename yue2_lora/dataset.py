"""Datasets: the folder of songs a run trains on.

A dataset is a folder of audio files plus, for every song:

* ``<stem>.yue2.txt`` - the one-line style the planner is prompted with. When
  it is missing or empty it is filled by the captioner (MOSS-Music, see
  :mod:`yue2_lora.caption`) - a description of the song heard by ear, exactly
  as YuE2-Studio's "按声音描述" - or from an existing ``<stem>.txt`` caption;
* ``<stem>.lrc`` - the lyrics as an LRC file; timestamps are stripped;
* ``<stem>.txt`` - generated/merged from the two above: ``caption:`` is the
  ``.yue2.txt`` line, ``is_instrumental:`` defaults to false, ``lyrics:`` is
  the LRC text.

The trainer itself reads only the sidecars, so ``<stem>.txt`` is the source of
truth once a run starts.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .stages import sidecars
from .util import ensure_dir, eprint, find_on_path

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
WAV_FORMAT_PCM = 1
WAV_FORMAT_FLOAT = 3

#: One or more ``[mm:ss.xx]`` timestamps at the start of an LRC line.
_LRC_TIMESTAMP = re.compile(r"^\s*(?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\])+")
#: ``[ti:...]``, ``[ar:...]`` and the other LRC metadata tags.
_LRC_METADATA = re.compile(r"^\s*\[[A-Za-z][A-Za-z0-9_-]*:.*\]\s*$")


class DatasetError(RuntimeError):
    pass


@dataclass
class Song:
    stem: str
    file: str
    style: str = ""
    lyrics: str = ""
    instrumental: bool = False


def _audio_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise DatasetError(f"the audio folder does not exist: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


# -- sidecars --------------------------------------------------------------


def parse_sidecar(text: str) -> Tuple[str, str, bool]:
    """Read ``caption:`` / ``is_instrumental:`` / ``lyrics:`` from a sidecar."""
    style = ""
    instrumental = False
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        lowered = line.lower()
        if lowered.startswith("caption:"):
            style = line[len("caption:"):].strip()
        elif lowered.startswith("is_instrumental:"):
            instrumental = line[len("is_instrumental:"):].strip().lower() in ("true", "1", "yes")
        elif lowered.startswith("lyrics:"):
            index += 1
            break
        index += 1
    lyrics = "\n".join(lines[index:]).strip()
    return style, lyrics, instrumental


def _read_sidecar(stem: Path) -> Optional[Tuple[str, str, bool]]:
    text_path = stem.with_suffix(".txt")
    if not text_path.is_file():
        return None
    return parse_sidecar(text_path.read_text(encoding="utf-8"))


def _read_style(stem: Path) -> str:
    """The one-line style from ``<stem>.yue2.txt``, or an empty string."""
    path = Path(f"{stem}.yue2.txt")
    if not path.is_file():
        return ""
    return " ".join(path.read_text(encoding="utf-8").split())


def parse_lrc(text: str) -> str:
    """The plain lyrics of an LRC file: timestamps and metadata stripped."""
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.lstrip("\ufeff").strip()
        if not line or _LRC_METADATA.match(line):
            continue
        line = _LRC_TIMESTAMP.sub("", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def write_sidecars(folder: Path, songs: List[Song]) -> None:
    for song in songs:
        lyrics = "" if song.instrumental else song.lyrics
        for extension, text in sidecars(song.style, lyrics, song.instrumental):
            Path(folder / f"{song.stem}{extension}").write_text(text, encoding="utf-8")


def songs_needing_style(audio: Path) -> List[Path]:
    """Audio files without a style: no ``.yue2.txt`` and no ``.txt`` caption."""
    files = _audio_files(audio)
    needed: List[Path] = []
    for path in files:
        stem = path.with_suffix("")
        if _read_style(stem):
            continue
        existing = _read_sidecar(stem)
        if existing and existing[0].strip():
            continue
        needed.append(path)
    return needed


def collect_songs(audio: Path, styles: Optional[Dict[str, str]] = None) -> List[Song]:
    """Build the sidecars from ``.lrc`` files, filling missing styles.

    The style comes from ``<stem>.yue2.txt`` (or an existing ``<stem>.txt``
    caption). When both are missing, ``styles`` - the captioner's one-line
    description keyed by file name - supplies it and it is written to
    ``<stem>.yue2.txt``. Without an entry the song is reported as a problem
    telling the caller to run the captioner first.
    The lyrics come from ``<stem>.lrc``. ``<stem>.txt`` is then merged.
    """
    files = _audio_files(audio)
    if not files:
        raise DatasetError(f"no audio files ({', '.join(AUDIO_EXTENSIONS)}) in {audio}")
    styles = styles or {}
    songs: List[Song] = []
    problems: List[str] = []
    for path in files:
        stem = path.with_suffix("")
        existing = _read_sidecar(stem)

        style = _read_style(stem)
        if not style:
            source = (existing[0].strip() if existing else "") or styles.get(path.name, "").strip()
            if not source:
                problems.append(
                    f"{path.name}: no style - run `yue2-lora caption -c <config> --install` "
                    f"or write {path.stem}.yue2.txt"
                )
                continue
            Path(f"{stem}.yue2.txt").write_text(f"{source}\n", encoding="utf-8")
            style = source

        instrumental = bool(existing[2]) if existing else False
        lyrics = ""
        if not instrumental:
            lrc = path.with_suffix(".lrc")
            if lrc.is_file():
                lyrics = parse_lrc(lrc.read_text(encoding="utf-8"))
            elif existing and existing[1]:
                lyrics = existing[1]
            else:
                problems.append(
                    f"{path.name}: no {lrc.name} for its lyrics "
                    "(or keep is_instrumental: true in an existing .txt)"
                )
                continue

        for extension, text in sidecars(style, lyrics, instrumental):
            if extension == ".txt":
                Path(f"{stem}.txt").write_text(text, encoding="utf-8")
        songs.append(
            Song(
                stem=path.stem,
                file=path.name,
                style=style,
                lyrics=lyrics,
                instrumental=instrumental,
            )
        )
    if problems:
        raise DatasetError("cannot build the sidecars:\n  " + "\n  ".join(problems))
    return songs


@dataclass
class Inspection:
    """What a prepared dataset currently has, without writing anything."""

    songs: List[Song]
    problems: List[str]


def inspect_songs(audio: Path) -> Inspection:
    """Read the existing sidecars only; never generate or modify files."""
    try:
        files = _audio_files(audio)
    except DatasetError as error:
        return Inspection([], [str(error)])
    if not files:
        return Inspection([], [f"no audio files ({', '.join(AUDIO_EXTENSIONS)}) in {audio}"])
    songs: List[Song] = []
    problems: List[str] = []
    for path in files:
        stem = path.with_suffix("")
        style = _read_style(stem)
        if not style:
            problems.append(f"{path.name}: missing {path.stem}.yue2.txt")
            continue
        existing = _read_sidecar(stem)
        if existing is None:
            problems.append(f"{path.name}: missing {path.stem}.txt")
            continue
        _style, lyrics, instrumental = existing
        songs.append(
            Song(
                stem=path.stem,
                file=path.name,
                style=style,
                lyrics=lyrics,
                instrumental=instrumental,
            )
        )
    return Inspection(songs, problems)


def require_prepared(audio: Path) -> List[Song]:
    """The songs of a prepared dataset, or an error telling the user to prepare it."""
    inspection = inspect_songs(audio)
    if inspection.problems:
        raise DatasetError(
            "the dataset is not prepared; run `yue2-lora dataset -c config.toml` first:\n  "
            + "\n  ".join(inspection.problems)
        )
    return inspection.songs


# -- audio checks ----------------------------------------------------------


def wav_sample_rate(path: Path) -> Optional[int]:
    """The sample rate of a RIFF/WAVE file, or None when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return None
            while True:
                chunk = handle.read(8)
                if len(chunk) < 8:
                    return None
                chunk_id, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
                if chunk_id == b"fmt ":
                    body = handle.read(size)
                    if len(body) < 16:
                        return None
                    _format, _channels, rate = struct.unpack("<HHI", body[:8])
                    return int(rate)
                handle.seek(size + (size & 1), 1)
    except OSError:
        return None


def probe_sample_rate(path: Path, ffprobe: Optional[str]) -> Optional[int]:
    if path.suffix.lower() == ".wav":
        rate = wav_sample_rate(path)
        if rate is not None:
            return rate
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def check_audio(audio: Path, songs: List[Song]) -> List[str]:
    """Warn about songs that are not 48 kHz WAV files, as the trainer expects."""
    ffprobe = find_on_path("ffprobe")
    problems: List[str] = []
    for song in songs:
        path = audio / song.file
        if path.suffix.lower() != ".wav":
            problems.append(f"{song.file}: not a WAV file (the trainer expects 48 kHz WAV)")
            continue
        rate = probe_sample_rate(path, ffprobe)
        if rate is None:
            problems.append(f"{song.file}: could not read the sample rate")
        elif rate != 48000:
            problems.append(f"{song.file}: {rate} Hz, expected 48000 Hz")
    return problems


def normalize_audio(audio: Path, songs: List[Song], destination: Path, ffmpeg: Optional[str]) -> Path:
    """Convert every song to a 48 kHz 16-bit WAV in ``destination``."""
    if not ffmpeg:
        raise DatasetError("48 kHz conversion needs ffmpeg on PATH (or set paths.ffmpeg)")
    ensure_dir(destination)
    for index, song in enumerate(songs, 1):
        source = audio / song.file
        target = destination / f"{song.stem}.wav"
        eprint(f"[audio] {index}/{len(songs)} {song.file} -> {target.name}")
        subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
             "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(target)],
            check=True,
        )
        for extension in (".txt", ".yue2.txt"):
            sidecar = audio / f"{song.stem}{extension}"
            if sidecar.is_file():
                shutil.copy2(sidecar, destination / sidecar.name)
    return destination


def prepare_dataset(
    audio: Path,
    normalize: bool = False,
    prepared_dir: Optional[Path] = None,
    ffmpeg: Optional[str] = None,
    styles: Optional[Dict[str, str]] = None,
) -> Tuple[List[Song], Path]:
    """Build the sidecars and, when asked, the 48 kHz audio a run trains on.

    ``styles`` are the captioner's descriptions (keyed by file name) for songs
    that have no ``.yue2.txt`` yet. Returns the songs and the folder the
    trainer's ``--audio`` points at: the source folder, or a prepared 48 kHz
    copy. This is the only dataset step that writes files; ``train`` just
    checks the result.
    """
    songs = collect_songs(audio, styles)
    if normalize:
        target = prepared_dir or (audio.parent / "prepared-audio")
        return songs, normalize_audio(audio, songs, target, ffmpeg)
    problems = check_audio(audio, songs)
    if problems:
        raise DatasetError(
            "the audio is not 48 kHz WAV; set [dataset] normalize = true to convert it:\n  "
            + "\n  ".join(problems)
        )
    return songs, audio


def dataset_summary(songs: List[Song]) -> str:
    instrumental = sum(1 for song in songs if song.instrumental)
    return f"{len(songs)} songs ({len(songs) - instrumental} with lyrics, {instrumental} instrumental)"
