"""Exercise four-frame manual updates with real PNG and WebP images."""

import json
import random
import unittest
from pathlib import Path

from PIL import Image

from scripts import update_manual_images as updater
from scripts.tests.test_capture_source import CaptureFixture


class ImageFixture(CaptureFixture):
    def __init__(self):
        super().__init__()
        self.images = self.source / "docs/manual/images"
        self.images.mkdir(parents=True)
        metadata = {"fixture": self.fixture, "capture": "Native Windows RITK client framebuffer"}
        (self.images / "dicom-native-annotations.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        for index, stem in enumerate(updater.FRAME_STEMS):
            color = (10 + index, 20, 30, 255)
            Image.new("RGBA", updater.FRAME_SIZE, color).save(self.capture / f"{stem}.png")
            prior = color if stem != "dicom-native-annotations" else (99, 20, 30, 255)
            Image.new("RGBA", updater.FRAME_SIZE, prior).save(
                self.images / f"{stem}.webp", format="WEBP", lossless=True
            )

    def arguments(self):
        return [
            "--capture-root", str(self.capture),
            "--capture-record", str(self.record),
            "--source-root", str(self.source),
            "--repository", str(self.repository),
            "--atlas-root", str(self.atlas_root),
            "--base-rev", self.base,
        ]

    def outputs(self):
        return {path.name: path.read_bytes() for path in self.images.iterdir()}


class UpdateManualImagesTests(unittest.TestCase):
    def test_update_preserves_equal_bytes_and_losslessly_encodes_changed_frame(self):
        fixture = ImageFixture()
        self.addCleanup(fixture.close)
        before = fixture.outputs()
        self.assertEqual(updater.main(fixture.arguments()), 0)
        after = fixture.outputs()
        for stem in updater.FRAME_STEMS:
            name = f"{stem}.webp"
            if stem == "dicom-native-annotations":
                self.assertNotEqual(after[name], before[name])
                self.assertEqual(
                    updater.rgba(after[name], name, "WEBP"),
                    updater.rgba((fixture.capture / f"{stem}.png").read_bytes(), name, "PNG"),
                )
            else:
                self.assertEqual(after[name], before[name])
        metadata = json.loads((fixture.images / "dicom-native-annotations.json").read_text())
        self.assertEqual(metadata["comparison"]["reencoded_frames"],
                         ["dicom-native-annotations.webp"])
        self.assertEqual(metadata["comparison"]["unchanged_frames"], [
            "dicom-native-slices.webp", "dicom-native-calibration.webp",
            "dicom-native-length-edit.webp",
        ])
        self.assertEqual(metadata["image_tools"]["Pillow"]["applies_to"],
                         "comparison.reencoded_frames")
        for name, frame in metadata["frames"].items():
            self.assertLessEqual(frame["webp_bytes"], updater.FRAME_BYTE_LIMIT, name)
            self.assertNotIn("encoding", frame)

    def test_invalid_frames_never_partially_write_outputs(self):
        def frame(fixture):
            return fixture.capture / "dicom-native-length-edit.png"

        def oversized(fixture):
            pixels = random.Random(7).randbytes(4 * updater.FRAME_SIZE[0] * updater.FRAME_SIZE[1])
            image = Image.frombytes("RGBA", updater.FRAME_SIZE, pixels)
            image.save(frame(fixture), format="PNG")
            image.save(fixture.images / "dicom-native-length-edit.webp",
                       format="WEBP", lossless=True)

        cases = {
            "corrupt": lambda fixture: frame(fixture).write_bytes(b"not an image"),
            "missing": lambda fixture: frame(fixture).unlink(),
            "wrong_size": lambda fixture: Image.new("RGBA", (1, 1)).save(
                frame(fixture), format="PNG"),
            "jpeg_png": lambda fixture: Image.new("RGB", updater.FRAME_SIZE).save(
                frame(fixture), format="JPEG"),
            "png_webp": lambda fixture: Image.new("RGBA", updater.FRAME_SIZE).save(
                fixture.images / "dicom-native-length-edit.webp", format="PNG"),
            "oversized": oversized,
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                fixture = ImageFixture()
                try:
                    mutate(fixture)
                    before = fixture.outputs()
                    self.assertEqual(updater.main(fixture.arguments()), 1)
                    self.assertEqual(fixture.outputs(), before)
                finally:
                    fixture.close()


if __name__ == "__main__":
    unittest.main()
