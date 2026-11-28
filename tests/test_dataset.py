import tempfile
import unittest
import wave
from pathlib import Path

from yue2_lora.dataset import (
    DatasetError,
    Song,
    collect_songs,
    inspect_songs,
    parse_lrc,
    parse_sidecar,
    prepare_dataset,
    require_prepared,
    songs_needing_style,
    wav_sample_rate,
    write_sidecars,
)


def make_wav(path: Path, rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = wave.open(str(path), "wb")
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(rate)
    handle.writeframes(b"\x00\x00" * 100)
    handle.close()


LRC = """\
[ti:My Song]
[ar:Artist]
[00:00.00]intro line
[00:12.34]hold the light
[01:18.90]a little longer
"""


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="y2ds-"))
        self.audio = self.root / "audio"
        self.audio.mkdir()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_sidecar_round_trip(self):
        style, lyrics, instrumental = parse_sidecar(
            "caption: indie pop\nis_instrumental: true\nlyrics:\nla la\n"
        )
        self.assertEqual(style, "indie pop")
        self.assertEqual(lyrics, "la la")
        self.assertTrue(instrumental)
        write_sidecars(self.audio, [Song("a", "a.wav", "jazz", "ba", False)])
        text = (self.audio / "a.txt").read_text(encoding="utf-8")
        self.assertIn("caption: jazz", text)
        self.assertIn("is_instrumental: false", text)
        self.assertEqual((self.audio / "a.yue2.txt").read_text(encoding="utf-8"), "jazz\n")

    def test_lrc_strips_timestamps_and_metadata(self):
        self.assertEqual(parse_lrc(LRC), "intro line\nhold the light\na little longer")
        self.assertEqual(parse_lrc("[00:01.2][00:09.5]twice\n[Verse]\nwords\n"), "twice\n[Verse]\nwords")

    def test_yue2_is_generated_from_the_existing_caption_then_txt_is_merged(self):
        make_wav(self.audio / "song.wav")
        (self.audio / "song.lrc").write_text(LRC, encoding="utf-8")
        (self.audio / "song.txt").write_text(
            "caption: lo-fi\nis_instrumental: false\nlyrics:\nold words\n", encoding="utf-8"
        )
        songs = collect_songs(self.audio)
        self.assertEqual(len(songs), 1)
        self.assertEqual(songs[0].style, "lo-fi")
        self.assertEqual(songs[0].lyrics, "intro line\nhold the light\na little longer")
        self.assertEqual(
            (self.audio / "song.yue2.txt").read_text(encoding="utf-8"), "lo-fi\n"
        )
        merged = (self.audio / "song.txt").read_text(encoding="utf-8")
        self.assertIn("caption: lo-fi", merged)
        self.assertIn("is_instrumental: false", merged)
        self.assertIn("hold the light", merged)
        self.assertNotIn("old words", merged)

    def test_style_from_the_captioner_is_written_then_merged(self):
        make_wav(self.audio / "song.wav")
        (self.audio / "song.lrc").write_text(LRC, encoding="utf-8")
        songs = collect_songs(self.audio, {"song.wav": "indie pop, warm vocal, 96 BPM"})
        self.assertEqual(len(songs), 1)
        self.assertEqual(songs[0].style, "indie pop, warm vocal, 96 BPM")
        self.assertEqual(
            (self.audio / "song.yue2.txt").read_text(encoding="utf-8"),
            "indie pop, warm vocal, 96 BPM\n",
        )
        merged = (self.audio / "song.txt").read_text(encoding="utf-8")
        self.assertIn("caption: indie pop, warm vocal, 96 BPM", merged)
        self.assertIn("hold the light", merged)

    def test_missing_style_points_at_the_captioner(self):
        make_wav(self.audio / "song.wav")
        (self.audio / "song.lrc").write_text(LRC, encoding="utf-8")
        with self.assertRaises(DatasetError) as caught:
            collect_songs(self.audio)
        self.assertIn("yue2-lora caption", str(caught.exception))

    def test_songs_needing_style_lists_songs_without_a_style(self):
        make_wav(self.audio / "a.wav")
        make_wav(self.audio / "b.wav")
        (self.audio / "a.yue2.txt").write_text("ambient\n", encoding="utf-8")
        self.assertEqual([path.name for path in songs_needing_style(self.audio)], ["b.wav"])

    def test_missing_lrc_is_reported(self):
        make_wav(self.audio / "song.wav")
        (self.audio / "song.yue2.txt").write_text("ambient\n", encoding="utf-8")
        with self.assertRaises(DatasetError) as caught:
            collect_songs(self.audio)
        self.assertIn("song.lrc", str(caught.exception))

    def test_instrumental_is_kept_from_an_existing_sidecar(self):
        make_wav(self.audio / "song.wav")
        (self.audio / "song.yue2.txt").write_text("ambient\n", encoding="utf-8")
        (self.audio / "song.txt").write_text(
            "caption: ambient\nis_instrumental: true\nlyrics:\n", encoding="utf-8"
        )
        songs = collect_songs(self.audio)
        self.assertTrue(songs[0].instrumental)
        self.assertEqual(songs[0].lyrics, "")

    def test_wav_sample_rate_is_read(self):
        make_wav(self.audio / "48.wav", 48000)
        make_wav(self.audio / "44.wav", 44100)
        self.assertEqual(wav_sample_rate(self.audio / "48.wav"), 48000)
        self.assertEqual(wav_sample_rate(self.audio / "44.wav"), 44100)
        self.assertIsNone(wav_sample_rate(self.audio / "missing.wav"))

    def test_inspect_reports_missing_files_without_writing(self):
        make_wav(self.audio / "song.wav")
        (self.audio / "song.lrc").write_text(LRC, encoding="utf-8")
        result = inspect_songs(self.audio)
        self.assertEqual(result.songs, [])
        self.assertTrue(any("song.yue2.txt" in problem for problem in result.problems))
        self.assertFalse((self.audio / "song.yue2.txt").exists())
        self.assertFalse((self.audio / "song.txt").exists())

    def test_require_prepared_points_at_the_dataset_command(self):
        make_wav(self.audio / "song.wav")
        with self.assertRaises(DatasetError) as caught:
            require_prepared(self.audio)
        self.assertIn("yue2-lora dataset", str(caught.exception))

    def test_prepare_writes_the_files_train_then_reads(self):
        make_wav(self.audio / "song.wav")
        (self.audio / "song.lrc").write_text(LRC, encoding="utf-8")
        # Pre-create .yue2.txt so we know the expected style.
        (self.audio / "song.yue2.txt").write_text("ambient\n", encoding="utf-8")
        songs, output = prepare_dataset(self.audio, normalize=False)
        self.assertEqual(output, self.audio)
        self.assertEqual(songs[0].style, "ambient")
        self.assertTrue((self.audio / "song.yue2.txt").is_file())
        self.assertTrue((self.audio / "song.txt").is_file())
        prepared = require_prepared(self.audio)
        self.assertEqual(prepared[0].lyrics, "intro line\nhold the light\na little longer")

    def test_prepare_rejects_non_48k_audio_without_normalize(self):
        make_wav(self.audio / "song.wav", 44100)
        (self.audio / "song.lrc").write_text(LRC, encoding="utf-8")
        (self.audio / "song.yue2.txt").write_text("ambient\n", encoding="utf-8")
        with self.assertRaises(DatasetError) as caught:
            prepare_dataset(self.audio, normalize=False)
        self.assertIn("normalize", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
