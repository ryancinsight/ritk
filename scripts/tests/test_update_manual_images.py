"""Exercise four-frame manual updates with real PNG and WebP images."""

import json
import random
import sys
import unittest
from pathlib import Path

from PIL import Image

from scripts import capture_source
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
        for index, stem in enumerate(capture_source.CAPTURE_FRAME_STEMS):
            color = (10 + index, 20, 30, 255)
            Image.new("RGBA", capture_source.FRAME_SIZE, color).save(
                self.capture / f"{stem}.png"
            )
            prior = color if stem != "dicom-native-annotations" else (99, 20, 30, 255)
            Image.new("RGBA", capture_source.FRAME_SIZE, prior).save(
                self.images / f"{stem}.webp", format="WEBP", lossless=True
            )
        self.write_record()

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
    def test_substitution_after_admission_uses_admitted_bytes(self):
        fixture = ImageFixture()
        self.addCleanup(fixture.close)
        name = "dicom-native-slices"
        path = fixture.capture / f"{name}.png"
        admitted = path.read_bytes()

        def substitute(frame, event, argument):
            if event == "call" and frame.f_code is updater.prepare_frames.__code__:
                Image.new("RGBA", capture_source.FRAME_SIZE, (200, 20, 30, 255)).save(
                    path, format="PNG"
                )
            return substitute

        sys.settrace(substitute)
        try:
            self.assertEqual(updater.main(fixture.arguments()), 0)
        finally:
            sys.settrace(None)
        self.assertNotEqual(path.read_bytes(), admitted)
        metadata = json.loads((fixture.images / "dicom-native-annotations.json").read_text())
        self.assertEqual(metadata["frames"][f"{name}.webp"]["source_png_sha256"],
                         capture_source.sha256(admitted))
        self.assertEqual(
            capture_source.decoded_rgba(
                (fixture.images / f"{name}.webp").read_bytes(), name, "WEBP"
            )[1],
            capture_source.decoded_rgba(admitted, name, "PNG")[1],
        )

    def test_substituted_valid_frame_is_typed_error_and_never_writes(self):
        fixture = ImageFixture()
        self.addCleanup(fixture.close)
        before = fixture.outputs()
        Image.new("RGBA", capture_source.FRAME_SIZE, (200, 20, 30, 255)).save(
            fixture.capture / "dicom-native-slices.png", format="PNG"
        )
        arguments = updater.parser().parse_args(fixture.arguments())
        with self.assertRaisesRegex(
            capture_source.CaptureSourceError, "frames differ from the capture record"
        ):
            updater.update(arguments)
        self.assertEqual(fixture.outputs(), before)

    def test_update_preserves_equal_bytes_and_losslessly_encodes_changed_frame(self):
        fixture = ImageFixture()
        self.addCleanup(fixture.close)
        before = fixture.outputs()
        self.assertEqual(updater.main(fixture.arguments()), 0)
        after = fixture.outputs()
        for stem in capture_source.CAPTURE_FRAME_STEMS:
            name = f"{stem}.webp"
            if stem == "dicom-native-annotations":
                self.assertNotEqual(after[name], before[name])
                self.assertEqual(
                    capture_source.decoded_rgba(after[name], name, "WEBP"),
                    capture_source.decoded_rgba(
                        (fixture.capture / f"{stem}.png").read_bytes(), name, "PNG"
                    ),
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
            pixels = random.Random(7).randbytes(
                4 * capture_source.FRAME_SIZE[0] * capture_source.FRAME_SIZE[1]
            )
            image = Image.frombytes("RGBA", capture_source.FRAME_SIZE, pixels)
            image.save(frame(fixture), format="PNG")
            image.save(fixture.images / "dicom-native-length-edit.webp",
                       format="WEBP", lossless=True)
            fixture.write_record()

        cases = {
            "corrupt": lambda fixture: frame(fixture).write_bytes(b"not an image"),
            "missing": lambda fixture: frame(fixture).unlink(),
            "wrong_size": lambda fixture: Image.new("RGBA", (1, 1)).save(
                frame(fixture), format="PNG"),
            "jpeg_png": lambda fixture: Image.new("RGB", capture_source.FRAME_SIZE).save(
                frame(fixture), format="JPEG"),
            "png_webp": lambda fixture: Image.new("RGBA", capture_source.FRAME_SIZE).save(
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
