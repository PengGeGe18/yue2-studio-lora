import unittest

from yue2_lora.recipe import Recipe, continuation_args, train_args
from yue2_lora.assets import TRAINING_FILES


def pairs(args):
    return [tuple(args[i:i + 2]) for i in range(len(args) - 1)]


class RecipeTests(unittest.TestCase):
    def test_presets_fix_the_base_matched_recipe(self):
        balanced = Recipe().for_songs(61)
        self.assertEqual(
            (balanced.steps, balanced.grad_accum, balanced.warmup, balanced.save_every),
            (200, 4, 6, 20),
        )
        self.assertEqual(
            (balanced.adapter, balanced.lokr_dim, balanced.lokr_factor, balanced.alpha),
            ("lokr", 128, 4, 256.0),
        )
        self.assertFalse(balanced.lyric_timing)
        self.assertEqual(balanced.target_kl, 0.0)

        thorough = Recipe(preset="thorough").for_songs(61)
        self.assertEqual(
            (thorough.steps, thorough.grad_accum, thorough.warmup, thorough.save_every),
            (300, 8, 9, 30),
        )
        fast = Recipe(preset="fast").for_songs(3)
        self.assertEqual((fast.steps, fast.grad_accum, fast.warmup, fast.save_every), (100, 4, 3, 10))

    def test_a_preset_does_not_carry_a_hand_set_adapter(self):
        hand = Recipe(adapter="lora", lokr_dim=32).for_songs(61)
        self.assertEqual((hand.adapter, hand.lokr_dim), ("lokr", 128))
        tuned = Recipe(preset="tuned", lokr_dim=32).for_songs(61)
        self.assertEqual(tuned.lokr_dim, 32)

    def test_epochs_become_steps_and_turn_the_kl_stop_off(self):
        tuned = Recipe(preset="tuned")
        by_epochs = Recipe(preset="tuned", stop="epochs", epochs=20).for_songs(61)
        self.assertEqual(by_epochs.steps, 1220)
        self.assertEqual(by_epochs.target_kl, 0.0)
        by_likeness = tuned.for_songs(61)
        self.assertEqual((by_likeness.steps, by_likeness.target_kl, by_likeness.grad_accum), (750, 1.4, 1))

    def test_check_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            Recipe(preset="slow").check()
        with self.assertRaises(ValueError):
            Recipe(steps=0).check()
        with self.assertRaises(ValueError):
            Recipe(adapter="dora").check()


class TrainArgsTests(unittest.TestCase):
    def test_base_matched_args_match_the_studio(self):
        recipe = Recipe().for_songs(10)
        args = train_args("base.safetensors", "companion.safetensors", "prepared/dataset.json", recipe, "output")
        self.assertEqual(args[0], "yue2-joint-train")
        pairs_found = pairs(args)
        for pair in [
            ("--optimizer", "adamw-lm"),
            ("--lr", "1e-4"),
            ("--grad-accum", "4"),
            ("--warmup", "6"),
            ("--ar-targets", "base"),
            ("--cursor-weight", "0"),
            ("--lokr-dim", "128"),
            ("--nar-crop-frames", "1500"),
            ("--steps", "200"),
            ("--save-every", "20"),
            ("--adapter-type", "lokr"),
        ]:
            self.assertIn(pair, pairs_found)
        self.assertNotIn("--target-kl", args)
        self.assertNotIn("--prodigy-d0", args)

    def test_tuned_args_use_prodigy_and_the_likeness_stop(self):
        recipe = Recipe(preset="tuned")
        args = train_args("b", "c", "d", recipe, "o")
        pairs_found = pairs(args)
        for pair in [
            ("--optimizer", "prodigy"),
            ("--adapter-type", "lokr"),
            ("--planner-lr-scale", "0.3"),
            ("--target-kl", "1.4"),
            ("--cursor-weight", "0.08"),
            ("--prodigy-d0", "1e-6"),
        ]:
            self.assertIn(pair, pairs_found)
        self.assertNotIn("--lr", args)

    def test_a_lora_without_lyric_timing_gets_zero_cursor(self):
        recipe = Recipe(preset="tuned", adapter="lora", lyric_timing=False, optimizer="adamw", learning_rate=2e-4)
        args = train_args("b", "c", "d", recipe, "o")
        pairs_found = pairs(args)
        self.assertIn(("--cursor-weight", "0"), pairs_found)
        self.assertIn(("--lr", "0.0002"), pairs_found)
        self.assertNotIn("--lokr-dim", args)

    def test_continuation_passes_resume_and_steps(self):
        recipe = Recipe(preset="tuned", steps=1400, target_kl=0.0)
        args = continuation_args("b", "c", "d", recipe, "output-x", "output/checkpoint-step1000/optimizer.resume")
        pairs_found = pairs(args)
        self.assertIn(("--resume", "output/checkpoint-step1000/optimizer.resume"), pairs_found)
        self.assertIn(("--output", "output-x"), pairs_found)
        self.assertIn(("--steps", "1400"), pairs_found)
        self.assertNotIn("--target-kl", args)


if __name__ == "__main__":
    unittest.main()
