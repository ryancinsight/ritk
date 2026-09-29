"""Integrity checks for the public MRI manual image provenance."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "docs/manual/images/dicom-metis-real-mri.png"
RESOURCE = ROOT / "docs/manual/images/dicom-metis-real-mri-resource.json"
REPLAY = ROOT / "docs/manual/images/dicom-metis-real-mri.json"
OBLIQUE_IMAGE = ROOT / "docs/manual/images/dicom-metis-real-mri-oblique.webp"
OBLIQUE_REPLAY = ROOT / "docs/manual/images/dicom-metis-real-mri-oblique.json"
MAX_MANUAL_IMAGE_BYTES = 200_000
OBLIQUE_IMAGE_SHA256 = "81c4c8a87a903622ea68c8bf9871e708229acd22a3231058340c8973b5a8280f"
OBLIQUE_CAPTURE_SHA256 = "1ae218828f79e02c8cd533b74e8225d86b2b44530e947cf7b001e34234d74fe0"
OBLIQUE_EXECUTABLE_SHA256 = "534f9b08d52fa4cc86e3503918327709ba7868eafac96a05247532265a8569ba"
OBLIQUE_SOURCE_TREE_SHA256 = "e2f7133aade94c2bfa856d7c3ed28fc8717a4fa27b0afd6efde9178c84678328"


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

    def test_oblique_phantom_image_matches_its_capture_record(self):
        replay = json.loads(OBLIQUE_REPLAY.read_text(encoding="utf-8"))
        image = OBLIQUE_IMAGE.read_bytes()
        source = replay["source_capture"]
        output = replay["output"]

        self.assertEqual(replay["schema"], 2)
        self.assertEqual(replay["status"], "passed")
        self.assertEqual(replay["item"], "RITK-SNAP-OBLIQUE-NATIVE-001")
        self.assertEqual(replay["dataset"]["dicom_files"], 94)
        self.assertEqual(replay["dataset"]["dicom_bytes"], 49_807_236)
        self.assertEqual(replay["dataset"]["directory_entries"], 95)
        self.assertEqual(replay["dataset"]["license_file_bytes"], 2_787)
        self.assertEqual(replay["dataset"]["license"], "CC BY 4.0")
        self.assertIn("not private patient data", replay["dataset"]["description"])
        self.assertIn("--metis-native-layout oblique", replay["runtime"]["command"])
        self.assertIn("--capture-application", replay["runtime"]["command"])
        self.assertIn("--locked --offline", replay["runtime"]["command"])
        self.assertEqual(
            replay["source_revision"]["ritk_base_commit"],
            "de32e1114df5398732eb65838e00b7c7c90b23f2",
        )
        self.assertEqual(
            replay["source_revision"]["source_tree_sha256"],
            OBLIQUE_SOURCE_TREE_SHA256,
        )
        self.assertEqual(output["image"], OBLIQUE_IMAGE.name)
        self.assertEqual(output["width"], 1280)
        self.assertEqual(output["height"], 800)
        self.assertEqual(hashlib.sha256(image).hexdigest(), OBLIQUE_IMAGE_SHA256)
        self.assertEqual(output["sha256"], OBLIQUE_IMAGE_SHA256)
        self.assertEqual(output["bytes"], len(image))
        self.assertEqual(output["executable_sha256"], OBLIQUE_EXECUTABLE_SHA256)
        self.assertEqual(output["executable_bytes"], 26_073_600)
        self.assertLessEqual(len(image), MAX_MANUAL_IMAGE_BYTES)
        self.assertEqual(source["sha256"], OBLIQUE_CAPTURE_SHA256)
        self.assertEqual(source["width"], output["width"])
        self.assertEqual(source["height"], output["height"])
        self.assertNotEqual(source["sha256"], output["sha256"])
        self.assertEqual(
            output["planes"], ["axial", "coronal", "sagittal", "oblique"]
        )
