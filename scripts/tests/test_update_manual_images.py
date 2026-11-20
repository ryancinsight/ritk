"""Exercise the source-bound manual image update command with real files."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "metis" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
from process_tree import run
import update_manual_images as updater


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class Fixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.source = self.root / "source"
        self.capture = self.root / "capture"
        self.record = self.root / "capture.json"
        self.images = self.source / "docs/manual/images"
        for path in (self.repository, self.source, self.capture, self.images):
            path.mkdir(parents=True, exist_ok=True)
        shared = {
            "Cargo.toml": b"[workspace]\nmembers=[]\n",
            "Cargo.lock": b"version = 4\n",
            "rust-toolchain.toml": b"[toolchain]\nchannel='stable'\n",
        }
        baseline = dict(shared, **{
            "crates/demo/src/lib.rs": b"pub fn value() -> u8 { 1 }\n",
            "crates/deleted/Cargo.toml": b"[package]\nname='deleted'\nversion='0.1.0'\n"})
        captured = dict(shared, **{
            "crates/demo/src/lib.rs": b"pub fn value() -> u8 { 2 }\n",
            "crates/new/Cargo.toml": b"[package]\nname='new'\nversion='0.1.0'\n"})
        for name, data in baseline.items():
            path = self.repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for name, data in captured.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.study = self.repository / "test_data/2_head_mri_t2/DICOM"
        self.study.mkdir(parents=True)
        for index in range(94):
            (self.study / f"{index:03}.dcm").write_bytes(b"")
        self.git("init", "-b", "main")
        self.git("add", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "crates")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.org", "commit", "-m", "Fixture")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("update-ref", "refs/remotes/origin/main", self.base)
        self.git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
        metadata = {"fixture": {"path": "test_data/2_head_mri_t2/DICOM", "dicom_images": 94}}
        (self.images / "dicom-native-annotations.json").write_text(json.dumps(metadata), encoding="utf-8")
        for index, stem in enumerate(updater.FRAME_STEMS):
            color = (10 + index, 20, 30, 255)
            Image.new("RGBA", updater.FRAME_SIZE, color).save(self.capture / f"{stem}.png")
            prior = color if stem != "dicom-native-annotations" else (99, 20, 30, 255)
            Image.new("RGBA", updater.FRAME_SIZE, prior).save(self.images / f"{stem}.webp", format="WEBP", lossless=True)
        self.write_record()

    def git(self, *arguments):
        result = run(["git", "-C", str(self.repository), *arguments], timeout=10)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        return result

    def write_record(self, **changes):
        sources = {name: digest(path) for name, path in updater.source_files(self.source).items()}
        stage = {"argv": ["cargo", "nextest", "run"], "exit": 0,
                 "environment": {"RITK_MANUAL_CAPTURE_DIR": str(self.capture),
                                 "RITK_MANUAL_STUDY": str(self.study)},
                 "source_before": sources, "source_after": dict(sources)}
        stage.update(changes)
        self.record.write_text(json.dumps([stage]), encoding="utf-8")

    def arguments(self):
        return ["--capture-root", str(self.capture), "--capture-record", str(self.record),
                "--source-root", str(self.source), "--repository", str(self.repository), "--base-rev", self.base]

    def outputs(self):
        return {path.name: path.read_bytes() for path in self.images.iterdir()}

    def close(self):
        self.temporary.cleanup()

class UpdateManualImagesTests(unittest.TestCase):
    def test_valid_update_preserves_equal_bytes_and_records_source_overrides(self):
        fixture = Fixture()
        self.addCleanup(fixture.close)
        before = fixture.outputs()
        self.assertEqual(updater.main(fixture.arguments()), 0)
        after = fixture.outputs()
        for stem in updater.FRAME_STEMS:
            name = f"{stem}.webp"
            if stem == "dicom-native-annotations":
                self.assertNotEqual(after[name], before[name])
                self.assertEqual(updater.rgba(after[name], name), updater.rgba(
                    (fixture.capture / f"{stem}.png").read_bytes(), name))
            else:
                self.assertEqual(after[name], before[name])
        metadata = json.loads((fixture.images / "dicom-native-annotations.json").read_text())
        self.assertEqual(metadata["viewer_base_revision"], fixture.base)
        self.assertEqual(len(metadata["viewer_source_overrides"]["changed"]), 1)
        self.assertEqual(len(metadata["viewer_source_overrides"]["new"]), 1)
        self.assertEqual(len(metadata["viewer_source_overrides"]["deleted"]), 1)
        self.assertEqual(metadata["comparison"]["reencoded_frames"], ["dicom-native-annotations.webp"])
        self.assertEqual(metadata["capture_record_sha256"], digest(fixture.record))

    def test_invalid_inputs_never_partially_write_outputs(self):
        for failure in ("failed", "mutated", "corrupt", "missing"):
            with self.subTest(failure=failure):
                fixture = Fixture()
                try:
                    if failure == "failed":
                        fixture.write_record(exit=1)
                    elif failure == "mutated":
                        fixture.write_record(source_after={})
                    else:
                        frame = fixture.capture / "dicom-native-length-edit.png"
                        frame.write_bytes(b"not an image") if failure == "corrupt" else frame.unlink()
                    before = fixture.outputs()
                    self.assertEqual(updater.main(fixture.arguments()), 1)
                    self.assertEqual(fixture.outputs(), before)
                finally:
                    fixture.close()


if __name__ == "__main__":
    unittest.main()
