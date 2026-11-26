import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from yue2_lora import caption
from yue2_lora.caption import (
    CaptionError,
    Heard,
    from_prose,
    heard_style,
    normalize_yue2,
    parse_prose,
)

PROSE = """\
genre: mandopop, acoustic ballad
caption: A sparse acoustic ballad with fingerpicked guitar and warm male tenor. Strings swell under the chorus, and the production stays intimate and dry.
bpm: 72
key: C major
signature: 4
"""

#: A stand-in for ace-caption: reads --src-list and writes a prose file per song.
FAKE = """\
#!{python}
import sys
args = sys.argv[1:]
src_list = args[args.index("--src-list") + 1]
for line in open(src_list, encoding="utf-8"):
    line = line.rstrip("\\n")
    if not line:
        continue
    _, out = line.split("\\t")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write({prose!r})
"""


class CaptionParsingTests(unittest.TestCase):
    def test_parse_prose_reads_the_five_fields(self):
        fields = parse_prose(PROSE)
        self.assertEqual(fields["genre"], "mandopop, acoustic ballad")
        self.assertEqual(fields["bpm"], "72")
        self.assertIn("fingerpicked guitar", fields["caption"])

    def test_normalize_strips_labels_fences_and_trailing_bpm(self):
        self.assertEqual(normalize_yue2("caption: warm pop", 0), "warm pop")
        self.assertEqual(normalize_yue2("indie rock at 120 bpm.", 96), "indie rock, 96 BPM")
        self.assertEqual(normalize_yue2("```\nsoft jazz\n```", 0), "soft jazz")

    def test_heard_style_joins_genre_and_caption_with_tempo(self):
        self.assertEqual(
            heard_style("lo-fi", "dusty drums", 84),
            "lo-fi. dusty drums, 84 BPM",
        )
        self.assertEqual(heard_style("", "just a caption", 0), "just a caption")

    def test_from_prose_rejects_an_empty_caption(self):
        with self.assertRaises(CaptionError):
            from_prose("genre: rock\nbpm: 100\n")

    def test_from_prose_builds_a_heard(self):
        heard = from_prose(PROSE)
        self.assertIsInstance(heard, Heard)
        self.assertEqual(heard.bpm, 72)
        self.assertTrue(heard.style.endswith("72 BPM"))


class CaptionRunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="y2cap-"))
        self.binary = self.root / "ace-caption"
        self.binary.write_text(FAKE.format(python=sys.executable, prose=PROSE), encoding="utf-8")
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IEXEC)
        self.moss = self.root / "moss"
        self.moss.mkdir()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_describe_returns_one_heard_per_song(self):
        a = self.root / "a.wav"
        b = self.root / "b.wav"
        a.write_bytes(b"RIFF")
        b.write_bytes(b"RIFF")
        heard = caption.describe([a, b], self.binary, self.moss)
        self.assertEqual(set(heard), {a, b})
        self.assertEqual(heard[a].genre, "mandopop, acoustic ballad")
        self.assertEqual(heard[b].bpm, 72)

    def test_styles_for_is_keyed_by_file_name(self):
        a = self.root / "song.wav"
        a.write_bytes(b"RIFF")
        styles = caption.styles_for([a], self.binary, self.moss)
        self.assertIn("song.wav", styles)
        self.assertTrue(styles["song.wav"].startswith("mandopop, acoustic ballad."))


if __name__ == "__main__":
    unittest.main()
