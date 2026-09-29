"""Integrity checks for the public MRI manual image provenance."""

import hashlib
import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "docs/manual/images/dicom-metis-real-mri.png"
APPLICATION_IMAGE = (
    ROOT / "docs/manual/images/dicom-metis-real-mri-application-window.webp"
)
COMPARISON_IMAGE = (
    ROOT / "docs/manual/images/dicom-metis-real-mri-ct-multiseries-window.webp"
)
RESOURCE = ROOT / "docs/manual/images/dicom-metis-real-mri-resource.json"
REPLAY = ROOT / "docs/manual/images/dicom-metis-real-mri.json"
APPLICATION_REPLAY = (
    ROOT / "docs/manual/images/dicom-metis-real-mri-application-window.json"
)
COMPARISON_REPLAY = (
    ROOT / "docs/manual/images/dicom-metis-real-mri-ct-multiseries-window.json"
)


def lossless_webp_dimensions(image: bytes) -> tuple[int, int]:
    if image[:4] != b"RIFF" or image[8:12] != b"WEBP" or image[12:16] != b"VP8L":
        raise ValueError("manual screenshot is not a lossless WebP image")
    chunk_size = struct.unpack_from("<I", image, 16)[0]
    payload = image[20 : 20 + chunk_size]
    if len(payload) != chunk_size or not payload or payload[0] != 0x2F:
        raise ValueError("manual screenshot has a malformed VP8L header")
    dimensions = int.from_bytes(payload[1:5], "little")
    return (dimensions & 0x3FFF) + 1, ((dimensions >> 14) & 0x3FFF) + 1


class ManualImageProvenanceTests(unittest.TestCase):
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

    def test_full_application_captures_match_live_window_records(self):
        cases = (
            (
                APPLICATION_IMAGE,
                APPLICATION_REPLAY,
                94,
                49_807_236,
                1,
                1,
                ["MPR: axial", "MPR: coronal", "MPR: sagittal", "MIP projection"],
            ),
            (
                COMPARISON_IMAGE,
                COMPARISON_REPLAY,
                503,
                265_963_652,
                2,
                2,
                ["P1  |  MR  |  T2", "P2  |  CT  |  CT"],
            ),
        )
        for (
            image_path,
            record_path,
            instance_count,
            byte_count,
            study_count,
            series_count,
            panels,
        ) in cases:
            with self.subTest(image=image_path.name):
                capture = json.loads(record_path.read_text(encoding="utf-8"))
                output = capture["output"]
                dataset = capture["dataset"]
                source_capture = capture["source_capture"]
                runtime = capture["runtime"]
                image = image_path.read_bytes()
                self.assertEqual(output["path"], image_path.relative_to(ROOT).as_posix())
                self.assertEqual(output["sha256"], hashlib.sha256(image).hexdigest())
                self.assertEqual(output["bytes"], len(image))
                self.assertLessEqual(len(image), 200_000)
                self.assertEqual(
                    (output["width"], output["height"]),
                    lossless_webp_dimensions(image),
                )
                self.assertEqual(output["encoding"], "lossless WebP")
                self.assertNotIn("decoded_rgba_byte_equal_to_source_capture", output)
                self.assertEqual(runtime["process_returncode"], 0)
                self.assertRegex(source_capture["sha256"], r"\A[0-9a-f]{64}\Z")
                self.assertGreater(source_capture["bytes"], 0)
                self.assertEqual(
                    source_capture["encoding"],
                    "PNG captured from the live native application window",
                )
                self.assertEqual(dataset["dicom_instances_read"], instance_count)
                self.assertEqual(dataset["dicom_bytes_read"], byte_count)
                self.assertEqual(dataset["study_count"], study_count)
                self.assertEqual(dataset["series_count"], series_count)
                self.assertIn(dataset["path"], runtime["command"])
                self.assertEqual(
                    runtime["window"], {"width": 1_298, "height": 847, "dpi": 120}
                )
                self.assertEqual(runtime["client"], {"width": 1_280, "height": 800})
                self.assertEqual(
                    runtime["capture_utility"]["path"],
                    "scripts/python_native_capture.py",
                )
                self.assertEqual(runtime["ritk_pull_request"], "ryancinsight/ritk#676")
                self.assertRegex(runtime["ritk_revision"], r"\A[0-9a-f]{40}\Z")
                self.assertRegex(runtime["ritk_tree"], r"\A[0-9a-f]{40}\Z")
                self.assertEqual(
                    runtime["capture_utility"]["repository"], "ryancinsight/metis"
                )
                self.assertRegex(
                    runtime["capture_utility"]["file_revision"],
                    r"\A[0-9a-f]{40}\Z",
                )
                self.assertRegex(
                    runtime["capture_utility"]["sha256"], r"\A[0-9a-f]{64}\Z"
                )
                self.assertRegex(
                    runtime["capture_utility"]["git_blob"], r"\A[0-9a-f]{40}\Z"
                )
                self.assertEqual(len(runtime["executable_sha256"]), 64)
                self.assertGreater(runtime["executable_bytes"], 0)
                expected_controls = [
                    "File",
                    "View",
                    "Tools",
                    "Window",
                    "Open Study...",
                    "Open multiple series",
                    "W/L",
                    "Pan",
                    "Zoom",
                    "Length",
                    "Angle",
                    "Crosshair",
                    "Cine",
                    "Split screen",
                    "Series bar",
                ]
                if output["panel_actions_visible"]:
                    expected_controls.extend(["Panel maximize", "Panel close"])
                expected_controls.extend(["Series preview", "Load to P1"])
                self.assertEqual(output["visible_controls"], expected_controls)
                self.assertEqual(output["toolbar_divider_count"], 4)
                self.assertEqual(output["panels"], panels)
                self.assertEqual(len(dataset["series"]), series_count)
                self.assertEqual(
                    sum(series["instances"] for series in dataset["series"]),
                    instance_count,
                )
                series_uids = [
                    series["series_instance_uid"] for series in dataset["series"]
                ]
                self.assertEqual(len(set(series_uids)), series_count)
                self.assertEqual(
                    [series["panel"] for series in dataset["series"]],
                    [f"P{index}" for index in range(1, series_count + 1)],
                )
                if series_count > 1:
                    self.assertIn("--series-instance-uid", runtime["command"])
                    self.assertIn("--compare-series-instance-uid", runtime["command"])
                    self.assertNotIn("input_events", runtime)
                    self.assertIn("two independent image panels", output["visual_scope"])
                    self.assertEqual(
                        [
                            (series["modality"], series["instances"])
                            for series in dataset["series"]
                        ],
                        [("MR", 94), ("CT", 409)],
                    )
                if output["panel_actions_visible"]:
                    self.assertEqual(
                        output["visible_annotations"],
                        [
                            "Image number and total count",
                            "Window width and centre",
                            "Source pixel dimensions",
                        ],
                    )
                    if "input_events" in runtime:
                        self.assertEqual(
                            runtime["input_events"]["keys"],
                            ["F4", "Space", "ArrowDown", "Space"],
                        )
                        self.assertEqual(
                            runtime["input_events"]["pointer"],
                            {
                                "button": "Left",
                                "down": {"x": 959, "y": 646},
                                "up": {"x": 959, "y": 646},
                            },
                        )
                        self.assertNotIn(
                            "--series-instance-uid", runtime["command"]
                        )
                        self.assertNotIn(
                            "--compare-series-instance-uid", runtime["command"]
                        )
                self.assertFalse(output["clinical_patient_identifiers_displayed"])
                self.assertIn(
                    "actual running métis native window",
                    output["visual_scope"].casefold(),
                )
                self.assertIn("left study and series preview bar", output["visual_scope"])
                self.assertIn("thumbnail image-count badges", output["visual_scope"])
                if output["panel_actions_visible"]:
                    self.assertIn(
                        "title-bar maximize and close controls", output["visual_scope"]
                    )
