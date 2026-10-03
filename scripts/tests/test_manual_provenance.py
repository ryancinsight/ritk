"""Integrity checks for the public MRI manual image provenance."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "docs/manual/images/dicom-metis-real-mri.png"
RESOURCE = ROOT / "docs/manual/images/dicom-metis-real-mri-resource.json"
REPLAY = ROOT / "docs/manual/images/dicom-metis-real-mri.json"


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
