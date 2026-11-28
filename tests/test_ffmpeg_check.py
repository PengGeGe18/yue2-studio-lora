import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from yue2_lora.cli import main


def run_dataset(config_text: str):
    root = Path(tempfile.mkdtemp(prefix="y2ff-"))
    config = root / "config.toml"
    config.write_text(config_text, encoding="utf-8")
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = main(["dataset", "-c", str(config)])
    return code, stderr.getvalue()


class FFmpegCheckTests(unittest.TestCase):
    def test_missing_ffmpeg_stops_before_work(self):
        code, err = run_dataset(
            '[dataset]\naudio = "audio"\nnormalize = true\n'
            '[train]\nffmpeg = "definitely-not-ffmpeg-xyz"\n'
        )
        self.assertEqual(code, 1)
        self.assertIn("ffmpeg not found", err)
        self.assertIn("install", err.lower())

    def test_disabled_ffmpeg_with_normalize_errors(self):
        code, err = run_dataset(
            '[dataset]\naudio = "audio"\nnormalize = true\n'
            '[train]\nffmpeg = "none"\n'
        )
        self.assertEqual(code, 1)
        self.assertIn("normalize", err)


if __name__ == "__main__":
    unittest.main()
