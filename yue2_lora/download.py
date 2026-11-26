"""Resumable downloads with size and SHA-256 checks."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from .assets import Asset
from .util import eprint, human_bytes, sha256_of


class DownloadError(RuntimeError):
    pass


class Progress:
    """A single-line progress display that works on every platform."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self.start = time.monotonic()
        self.last = 0.0
        self.last_bytes = -1

    def __call__(self, done: int, total: Optional[int] = None) -> None:
        total = total or self.total
        now = time.monotonic()
        if now - self.last < 0.2 and done != total:
            return
        self.last = now
        elapsed = max(1e-6, now - self.start)
        speed = done / elapsed
        if total:
            percent = 100.0 * done / total
            text = (
                f"\r[{self.label}] {percent:5.1f}%  {human_bytes(done)} / {human_bytes(total)}  "
                f"{human_bytes(int(speed))}/s"
            )
        else:
            text = f"\r[{self.label}] {human_bytes(done)}  {human_bytes(int(speed))}/s"
        eprint(text, end="", flush=True)

    def finish(self) -> None:
        eprint()


def _remote_size(url: str) -> Optional[int]:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length and length.isdigit() else None
    except (urllib.error.URLError, ValueError):
        return None


def download(url: str, destination: Path, expected_bytes: Optional[int] = None) -> Path:
    """Download ``url`` to ``destination``, resuming a partial ``.part`` file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    total = expected_bytes or _remote_size(url)
    for attempt in range(1, 6):
        existing = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "yue2-lora/0.1"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                mode = "ab" if existing and response.status == 206 else "wb"
                if mode == "wb":
                    existing = 0
                progress = Progress(destination.name, total or 0)
                with open(part, mode) as handle:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        handle.write(block)
                        existing += len(block)
                        progress(existing, total)
                progress.finish()
            if total is None:
                total = part.stat().st_size
            if part.stat().st_size < total:
                raise DownloadError(
                    f"incomplete download: {part.stat().st_size} of {total} bytes"
                )
            part.replace(destination)
            return destination
        except (urllib.error.URLError, ConnectionError, TimeoutError, DownloadError) as error:
            if attempt == 5:
                raise DownloadError(f"could not download {url}: {error}") from error
            wait = min(30, 2 ** attempt)
            eprint(f"[download] attempt {attempt} failed ({error}); retrying in {wait}s")
            time.sleep(wait)
    raise DownloadError(f"could not download {url}")


def ensure_asset(asset: Asset, models_dir: Path) -> Path:
    """Download an asset into ``models_dir`` and verify it."""
    destination = models_dir / asset.filename
    if destination.is_file() and _verify(destination, asset):
        return destination
    eprint(f"[download] {asset.label} -> {destination}")
    download(asset.url, destination, asset.bytes)
    if not _verify(destination, asset):
        destination.unlink(missing_ok=True)
        raise DownloadError(f"{destination} failed verification")
    return destination


def _verify(path: Path, asset: Asset) -> bool:
    size = path.stat().st_size
    if size != asset.bytes:
        eprint(f"[verify] {path.name}: {size} bytes, expected {asset.bytes}")
        return False
    if asset.sha256:
        eprint(f"[verify] {path.name}: hashing")
        digest = sha256_of(path)
        if digest.lower() != asset.sha256.lower():
            eprint(f"[verify] {path.name}: sha256 {digest} != {asset.sha256}")
            return False
    return True


def extract_zip(archive: Path, destination: Path) -> Path:
    """Extract ``archive`` under ``destination`` and return the folder used."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise DownloadError(f"unsafe path in {archive}: {member.filename}")
            bundle.extract(member, destination)
    return destination


def find_file(folder: Path, name: str) -> Optional[Path]:
    """Find ``name`` anywhere under ``folder`` (case-insensitive on Windows)."""
    wanted = name.lower() if os.name == "nt" else name
    for candidate in folder.rglob("*"):
        if not candidate.is_file():
            continue
        key = candidate.name.lower() if os.name == "nt" else candidate.name
        if key == wanted:
            return candidate
    return None
