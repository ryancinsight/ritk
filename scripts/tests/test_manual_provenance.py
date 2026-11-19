"""Integrity checks for the public MRI manual image provenance."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "docs/manual/images/dicom-metis-real-mri.png"
RESOURCE = ROOT / "docs/manual/images/dicom-metis-real-mri-resource.json"
REPLAY = ROOT / "docs/manual/images/dicom-metis-real-mri.json"


def decoded_rgba(path):
    with Image.open(path) as image:
        return image.size, image.convert("RGBA").tobytes()


def decoded_images_equal(left, right):
    return decoded_rgba(left) == decoded_rgba(right)


class ManualImageProvenanceTests(unittest.TestCase):
    def test_rgba_comparison_detects_rgb_change_with_equal_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.png"
            equal = directory / "equal.webp"
            changed = directory / "changed.webp"
            Image.new("RGBA", (1, 1), (10, 20, 30, 255)).save(source)
            Image.new("RGBA", (1, 1), (10, 20, 30, 255)).save(equal, lossless=True)
            Image.new("RGBA", (1, 1), (11, 20, 30, 255)).save(changed, lossless=True)

            self.assertTrue(decoded_images_equal(source, equal))
            self.assertEqual(decoded_rgba(source)[1][3::4], decoded_rgba(changed)[1][3::4])
            self.assertFalse(decoded_images_equal(source, changed))

    def test_output_digest_and_size_match_the_tracked_png(self):
        resource = json.loads(RESOURCE.read_text(encoding="utf-8"))
        replay = json.loads(REPLAY.read_text(encoding="utf-8"))
        image = IMAGE.read_bytes()
        output = resource["output"]
        manual_image = replay["standalone_lock_replay"]["manual_image"]

        self.assertEqual(resource["schema"], 2)
        self.assertEqual(output["image"], IMAGE.name)
        self.assertEqual(output["sha256"], hashlib.sha256(image).hexdigest())
        self.assertEqual(output["bytes"], len(image))
        self.assertEqual(output["sha256"], manual_image["sha256"])
        self.assertEqual(output["bytes"], manual_image["bytes"])

    def test_source_capture_digest_and_size_remain_distinct(self):
        resource = json.loads(RESOURCE.read_text(encoding="utf-8"))
        replay = json.loads(REPLAY.read_text(encoding="utf-8"))
        capture = resource["source_capture"]
        replay_capture = replay["standalone_lock_replay"]

        self.assertEqual(capture["image"], replay_capture["capture"])
        self.assertEqual(capture["sha256"], replay_capture["capture_sha256"])
        self.assertEqual(capture["bytes"], replay_capture["capture_bytes"])
        self.assertNotEqual(capture["sha256"], resource["output"]["sha256"])
        self.assertNotEqual(capture["bytes"], resource["output"]["bytes"])

    def test_current_windows_frames_match_their_capture_record(self):
        directory = ROOT / "docs/manual/images"
        record = json.loads(
            (directory / "dicom-native-annotations.json").read_text(encoding="utf-8")
        )
        expected = {
            "dicom-native-slices.webp",
            "dicom-native-calibration.webp",
            "dicom-native-annotations.webp",
            "dicom-native-length-edit.webp",
        }
        self.assertEqual(set(record["frames"]), expected)
        self.assertEqual(record["fixture"]["dicom_images"], 94)
        for name, frame in record["frames"].items():
            with self.subTest(image=name):
                image = (directory / name).read_bytes()
                self.assertEqual(hashlib.sha256(image).hexdigest(), frame["webp_sha256"])
                self.assertEqual(len(image), frame["webp_bytes"])
                self.assertLessEqual(len(image), 200_000)
                self.assertEqual((frame["width_px"], frame["height_px"]), (1280, 800))

    def test_current_windows_frames_match_explicit_capture(self):
        capture_root = os.environ.get("RITK_MANUAL_CAPTURE_DIR")
        if capture_root is None:
            self.skipTest("pixel equality requires an explicit RITK_MANUAL_CAPTURE_DIR")
        capture_root = Path(capture_root)
        directory = ROOT / "docs/manual/images"
        record = json.loads(
            (directory / "dicom-native-annotations.json").read_text(encoding="utf-8")
        )
        for name, frame in record["frames"].items():
            with self.subTest(image=name):
                capture = capture_root / f"{Path(name).stem}.png"
                self.assertEqual(decoded_rgba(capture)[0], (frame["width_px"], frame["height_px"]))
                self.assertTrue(decoded_images_equal(capture, directory / name))
