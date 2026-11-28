import json
import tempfile
import unittest
from pathlib import Path

from yue2_lora import _toml
from yue2_lora.config import load


class TomlTests(unittest.TestCase):
    def test_sections_strings_numbers_and_comments(self):
        data = _toml.loads(
            """
            # top
            [workspace]
            dir = "."
            models = "models" # trailing
            [recipe]
            preset = "tuned"
            steps = 750
            alpha = 256.0
            lyric_timing = false
            """
        )
        self.assertEqual(data["workspace"]["models"], "models")
        self.assertEqual(data["recipe"]["preset"], "tuned")
        self.assertEqual(data["recipe"]["steps"], 750)
        self.assertEqual(data["recipe"]["alpha"], 256.0)
        self.assertFalse(data["recipe"]["lyric_timing"])

    def test_arrays(self):
        data = _toml.loads('items = ["a", "b", 3]\n')
        self.assertEqual(data["items"], ["a", "b", 3])


class ConfigTests(unittest.TestCase):
    def test_defaults_and_relative_paths(self):
        root = Path(tempfile.mkdtemp(prefix="y2cfg-"))
        (root / "dataset" / "audio").mkdir(parents=True)
        (root / "config.toml").write_text(
            """
            [dataset]
            audio = "dataset/audio"
            trigger = "sks"
            [recipe]
            preset = "thorough"
            """,
            encoding="utf-8",
        )
        config = load(root / "config.toml")
        self.assertEqual(config.audio, (root / "dataset/audio").resolve())
        self.assertEqual(config.models_dir, (root / "models").resolve())
        self.assertEqual(config.vocals, (root / "dataset" / "vocals").resolve())
        self.assertEqual(config.recipe.preset, "thorough")
        self.assertTrue(config.normalize)
        self.assertEqual(config.prepared_dir, (root / "dataset" / "prepared-audio").resolve())
        self.assertEqual(config.training_audio, config.prepared_dir)
        self.assertEqual(config.caption_seconds, 120)
        self.assertIsNone(config.captioner)

    def test_caption_settings_are_read(self):
        root = Path(tempfile.mkdtemp(prefix="y2cfg-"))
        (root / "dataset" / "audio").mkdir(parents=True)
        (root / "config.toml").write_text(
            '[dataset]\naudio = "dataset/audio"\ncaption_seconds = 90\ncaptioner = "bin/ace-caption"\n',
            encoding="utf-8",
        )
        config = load(root / "config.toml")
        self.assertEqual(config.caption_seconds, 90)
        self.assertEqual(config.captioner, (root / "bin" / "ace-caption").resolve())

    def test_dataset_style_key_is_ignored_without_error(self):
        """A stale [dataset] style key must not crash load; it is simply ignored."""
        root = Path(tempfile.mkdtemp(prefix="y2cfg-"))
        (root / "dataset" / "audio").mkdir(parents=True)
        (root / "config.toml").write_text(
            '[dataset]\naudio = "dataset/audio"\nstyle = "indie pop, warm vocal"\n',
            encoding="utf-8",
        )
        config = load(root / "config.toml")
        # The stale key was silently discarded; the captioner fills styles at runtime.
        self.assertEqual(config.audio, (root / "dataset" / "audio").resolve())

    def test_normalize_points_training_at_the_prepared_folder(self):
        root = Path(tempfile.mkdtemp(prefix="y2cfg-"))
        (root / "dataset" / "audio").mkdir(parents=True)
        (root / "config.toml").write_text(
            '[dataset]\naudio = "dataset/audio"\nnormalize = true\nprepared = "dataset/p48"\n',
            encoding="utf-8",
        )
        config = load(root / "config.toml")
        self.assertEqual(config.training_audio, (root / "dataset" / "p48").resolve())

    def test_json_config_is_accepted(self):
        root = Path(tempfile.mkdtemp(prefix="y2cfg-"))
        (root / "dataset" / "audio").mkdir(parents=True)
        (root / "config.json").write_text(
            json.dumps({"dataset": {"audio": "dataset/audio"}, "train": {"device": "cpu"}}),
            encoding="utf-8",
        )
        config = load(root / "config.json")
        self.assertEqual(config.device, "CPU")


if __name__ == "__main__":
    unittest.main()
