import os
import struct
import tempfile
import unittest
from pathlib import Path

from yue2_lora.checkpoints import (
    complete_resume_file,
    checkpoints,
    next_output_dir,
    normalize_device,
    resume_point,
)


def resume_record(payload: int = 64, metadata: int = 8) -> bytes:
    header = (
        b"Y2RSUME1"
        + struct.pack("<I", 1)
        + struct.pack("<Q", payload)
        + struct.pack("<Q", metadata)
        + b"\x00" * 8
        + struct.pack("<Q", metadata)
    )
    assert len(header) == 44
    return header + b"m" * metadata + b"p" * payload


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="y2ckpt-"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def add(self, output, step, state=True):
        directory = self.root / output / f"checkpoint-step{step}"
        directory.mkdir(parents=True)
        if state:
            (directory / "optimizer.resume").write_bytes(resume_record())
        for name in ("native-ar.safetensors", "native-nar.safetensors"):
            (directory / name).write_bytes(b"w")
        return directory

    def test_resume_uses_the_latest_complete_state(self):
        self.add("output", 500, True)
        self.add("output", 750, False)
        self.add("output-2", 900, True)
        broken = self.add("output-2", 950, True)
        data = resume_record()
        (broken / "optimizer.resume").write_bytes(data[:-1])
        step, path = resume_point(self.root)
        self.assertEqual(step, 900)
        self.assertTrue(str(path).endswith("output-2/checkpoint-step900/optimizer.resume"))
        self.assertEqual([item.step for item in checkpoints(self.root)], [950, 900, 750, 500])

    def test_new_joint_export_names_are_found(self):
        directory = self.root / "output" / "checkpoint-step200"
        directory.mkdir(parents=True)
        (directory / "optimizer.resume").write_bytes(resume_record())
        for name in (
            "anm_yue2_joint_ar_step200.safetensors",
            "anm_yue2_joint_nar_step200.safetensors",
        ):
            (directory / name).write_bytes(b"w")
        found = checkpoints(self.root)
        self.assertEqual([item.step for item in found], [200])
        self.assertTrue(found[0].files[0].name.endswith("joint_ar_step200.safetensors"))

    def test_incomplete_header_is_rejected(self):
        path = self.root / "optimizer.resume"
        path.write_bytes(resume_record()[:20])
        self.assertFalse(complete_resume_file(path))
        path.write_bytes(resume_record())
        self.assertTrue(complete_resume_file(path))

    def test_next_output_skips_existing_ones(self):
        (self.root / "output").mkdir()
        (self.root / "output-2").mkdir()
        self.assertEqual(next_output_dir(self.root).name, "output-3")

    def test_device_names_are_normalized(self):
        self.assertEqual(normalize_device("cuda:1"), "CUDA1")
        self.assertEqual(normalize_device("metal"), "MTL0")
        self.assertEqual(normalize_device("Vulkan2"), "Vulkan2")
        self.assertEqual(normalize_device("CPU"), "CPU")
        with self.assertRaises(ValueError):
            normalize_device("tpu0")


if __name__ == "__main__":
    unittest.main()
