import os
import platform
import unittest

from yue2_lora import trainer
from yue2_lora.assets import HOTSTEP_RELEASES
from yue2_lora.util import platform_key


class TrainerVariantTests(unittest.TestCase):
    def test_only_cuda_variants_are_trainable(self):
        self.assertTrue(trainer.TRAIN_VARIANTS)
        for name in trainer.TRAIN_VARIANTS:
            self.assertIn("cuda", name)
            self.assertIn(name, HOTSTEP_RELEASES)

    def test_env_override_picks_a_cuda_build(self):
        os.environ["YUE2_TRAIN_VARIANT"] = "linux-x64-cuda13.1"
        try:
            self.assertEqual(trainer.choose_variant(), "linux-x64-cuda13.1")
        finally:
            del os.environ["YUE2_TRAIN_VARIANT"]

    def test_non_cuda_and_unknown_variants_are_rejected(self):
        with self.assertRaises(trainer.TrainerError):
            trainer.choose_variant("linux-x64-vulkan")
        with self.assertRaises(trainer.TrainerError):
            trainer.choose_variant("no-such-build")

    def test_platform_default(self):
        system = platform_key()
        machine = platform.machine().lower()
        if system == "macos":
            with self.assertRaises(trainer.TrainerError):
                trainer.choose_variant()
        elif system in ("linux", "windows") and machine in ("x86_64", "amd64", "x64"):
            variant = trainer.choose_variant()
            self.assertIn(variant, {"linux-x64-cuda12.8", "win-x64-cuda12.8"})


if __name__ == "__main__":
    unittest.main()
