"""Small, dependency-free helpers shared by the CLI."""

from __future__ import annotations

import hashlib
import os
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def eprint(*args, **kwargs) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def sha256_of(path: os.PathLike, chunk: int = 8 << 20, progress=None) -> str:
    digest = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    return digest.hexdigest()


def ensure_dir(path: os.PathLike) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def quote(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def find_on_path(name: str) -> str | None:
    from shutil import which

    return which(name)


def is_windows() -> bool:
    return os.name == "nt"


def platform_key() -> str:
    import platform

    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"


def replace_path(value: object, root: Path) -> Path:
    """Resolve a config path against the config file's folder."""

    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path)


#: SONAMEs the prebuilt ``libggml-cuda.so`` links against (not bundled).
_CUDA_RUNTIME = {"libcudart.so.12": "cudart", "libcublas.so.12": "cublas"}


class GpuUnavailable(RuntimeError):
    """The CUDA backend exists but its runtime libraries cannot be found."""


def _python_lib_roots() -> List[Path]:
    roots = set()
    try:
        import site

        roots.update(Path(entry) for entry in site.getsitepackages())
        user = site.getusersitepackages()
        if user:
            roots.add(Path(user))
    except Exception:
        pass
    for entry in sys.path:
        if entry and ("site-packages" in entry or "dist-packages" in entry):
            roots.add(Path(entry))
    return sorted(roots)


def cuda_library_dirs(roots: Optional[Iterable[os.PathLike]] = None) -> List[Path]:
    """Folders holding CUDA 12 runtime libs that may need ``LD_LIBRARY_PATH``.

    Only the libraries the prebuilt ggml CUDA backend actually links against
    are considered. Linux only.
    """
    if not sys.platform.startswith("linux"):
        return []
    candidates: List[Path] = []
    for root in (Path(entry) for entry in (roots if roots is not None else _python_lib_roots())):
        candidates.extend(sorted(root.glob("nvidia/*/lib")))
        candidates.append(root / "torch" / "lib")
    candidates.append(Path("/usr/local/cuda/lib64"))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-12*/lib64")))
    seen = set()
    found: List[Path] = []
    for path in candidates:
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        if any((path / name).exists() for name in _CUDA_RUNTIME):
            found.append(path)
    return found


def _runtime_visible() -> bool:
    from ctypes.util import find_library

    return all(find_library(soname) for soname in _CUDA_RUNTIME.values())


def gpu_env(binary: os.PathLike, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Prepend CUDA runtime folders to ``LD_LIBRARY_PATH`` for a child process.

    The official HOT-Step builds dlopen ``libggml-cuda.so``, which needs
    ``libcudart.so.12`` / ``libcublas.so.12``. Those are not bundled, and when
    CUDA only arrives through PyTorch wheels the loader cannot see them, so
    ggml silently falls back to CPU. This makes the GPU backend loadable without
    the user exporting anything. A no-op on macOS/Windows, on CPU-only builds,
    and when the runtime is already resolvable; raises :class:`GpuUnavailable`
    when the CUDA build exists but no runtime can be found anywhere.
    """
    base = dict(os.environ if env is None else env)
    if not sys.platform.startswith("linux"):
        return base
    engine = Path(binary).parent
    if not any(engine.glob("libggml-cuda.so*")):
        return base
    existing = [part for part in base.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    already_there = any(
        (Path(folder) / soname).exists()
        for folder in existing
        for soname in _CUDA_RUNTIME
    )
    if already_there or _runtime_visible():
        return base
    dirs = cuda_library_dirs()
    if not dirs:
        raise GpuUnavailable(
            "the prebuilt CUDA backend needs libcudart.so.12 / libcublas.so.12, "
            "but neither is available and no pip-installed copy was found. "
            "Running on CPU would be pointless, so stopping. Install them:\n"
            "    pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12\n"
            "or use a system CUDA Toolkit registered with ldconfig. "
            "To run on CPU on purpose, install the CPU build instead:\n"
            "    YUE2_CAPTION_VARIANT=linux-x64-cpu yue2-lora caption -c <config> --install"
        )
    fresh = [str(path) for path in dirs if str(path) not in existing]
    if not fresh:
        return base
    base["LD_LIBRARY_PATH"] = os.pathsep.join(fresh + existing)
    eprint("[gpu] CUDA runtime not on the loader path; adding: " + ", ".join(fresh))
    return base
