import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from yue2_lora import util


class CudaDirsTests(unittest.TestCase):
    def test_finds_pip_cuda_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cudart = root / "nvidia" / "cuda_runtime" / "lib"
            cublas = root / "nvidia" / "cublas" / "lib"
            cudart.mkdir(parents=True)
            cublas.mkdir(parents=True)
            (cudart / "libcudart.so.12").write_text("")
            (cublas / "libcublas.so.12").write_text("")
            with mock.patch("yue2_lora.util.sys.platform", "linux"):
                dirs = util.cuda_library_dirs(roots=[root])
            self.assertIn(cudart, dirs)
            self.assertIn(cublas, dirs)

    def test_non_linux_has_no_dirs(self):
        with mock.patch("yue2_lora.util.sys.platform", "darwin"):
            self.assertEqual(util.cuda_library_dirs(roots=["/tmp"]), [])


class GpuEnvTests(unittest.TestCase):
    def test_prepends_when_runtime_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "engine"
            engine.mkdir()
            (engine / "libggml-cuda.so").write_text("")
            libdir = Path(tmp) / "cudalibs"
            libdir.mkdir()
            stderr = io.StringIO()
            with mock.patch("yue2_lora.util.sys.platform", "linux"), \
                    mock.patch("yue2_lora.util._runtime_visible", return_value=False), \
                    mock.patch("yue2_lora.util.cuda_library_dirs", return_value=[libdir]), \
                    contextlib.redirect_stderr(stderr):
                env = util.gpu_env(engine / "ace-caption", {"LD_LIBRARY_PATH": "/old"})
            self.assertEqual(env["LD_LIBRARY_PATH"], f"{libdir}{os.pathsep}/old")
            self.assertIn("adding", stderr.getvalue())

    def test_noop_when_runtime_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "engine"
            engine.mkdir()
            (engine / "libggml-cuda.so").write_text("")
            with mock.patch("yue2_lora.util.sys.platform", "linux"), \
                    mock.patch("yue2_lora.util._runtime_visible", return_value=True):
                env = util.gpu_env(engine / "ace-caption", {})
            self.assertNotIn("LD_LIBRARY_PATH", env)

    def test_raises_when_no_cuda_libs_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "engine"
            engine.mkdir()
            (engine / "libggml-cuda.so").write_text("")
            with mock.patch("yue2_lora.util.sys.platform", "linux"), \
                    mock.patch("yue2_lora.util._runtime_visible", return_value=False), \
                    mock.patch("yue2_lora.util.cuda_library_dirs", return_value=[]):
                with self.assertRaises(util.GpuUnavailable):
                    util.gpu_env(engine / "ace-caption", {})

    def test_noop_without_cuda_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "engine"
            engine.mkdir()
            with mock.patch("yue2_lora.util.sys.platform", "linux"), \
                    mock.patch("yue2_lora.util._runtime_visible", return_value=False):
                env = util.gpu_env(engine / "ace-caption", {})
            self.assertNotIn("LD_LIBRARY_PATH", env)


if __name__ == "__main__":
    unittest.main()
