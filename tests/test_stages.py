import unittest
from pathlib import Path

from yue2_lora.recipe import Recipe
from yue2_lora.stages import TrainingInputs, sidecars, training_stages


def inputs(recipe):
    return TrainingInputs(
        audio=Path("audio"),
        models=Path("models"),
        checkpoint=Path("models/base.safetensors"),
        tokenizer=Path("models/token.safetensors"),
        companion=Path("models/companion.safetensors"),
        run=Path("run"),
        vocals=Path("vocals"),
        trigger="sks",
        recipe=recipe,
    )


def flat(stage):
    return [str(arg) for arg in stage.args]


def pairs(args):
    return [tuple(args[i:i + 2]) for i in range(len(args) - 1)]


class StageTests(unittest.TestCase):
    def test_a_base_matched_run_skips_alignment_and_trains_last(self):
        stages = training_stages(inputs(Recipe().for_songs(10)))
        self.assertEqual([stage.id for stage in stages], ["latents", "codes", "scores", "prepare", "train"])
        self.assertIn(("--loudness-lufs", "0"), pairs(flat(stages[0])))
        self.assertIn(("--lyric-timing", "0"), pairs(flat(stages[3])))
        self.assertIn(("--trigger", "sks"), pairs(flat(stages[3])))
        self.assertIn(("--companion", "models/companion.safetensors"), pairs(flat(stages[4])))

    def test_a_tuned_run_aligns_lyrics(self):
        stages = training_stages(inputs(Recipe(preset="tuned")))
        self.assertEqual(
            [stage.id for stage in stages],
            ["latents", "codes", "align", "scores", "prepare", "train"],
        )
        align = flat(stages[2])
        self.assertIn("--mmsfa", align)
        self.assertIn("--stems", align)
        self.assertIn(("--lyric-timing", "1"), pairs(flat(stages[4])))

    def test_sidecars_put_lyrics_last(self):
        [(ext, text), (style_ext, style)] = sidecars("indie  pop,\nfemale", "[Verse]\nla", False)
        self.assertEqual(ext, ".txt")
        self.assertTrue(text.startswith("caption: indie pop, female\n"))
        self.assertTrue(text.endswith("lyrics:\n[Verse]\nla\n"))
        self.assertEqual((style_ext, style), (".yue2.txt", "indie pop, female\n"))


if __name__ == "__main__":
    unittest.main()
