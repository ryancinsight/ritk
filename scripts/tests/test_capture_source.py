"""Exercise manual-capture source binding with real Git and files."""

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts import capture_source


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_files(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def atlas_root():
    value = os.environ.get("RITK_MANUAL_ATLAS_ROOT")
    if value is None:
        raise RuntimeError("RITK_MANUAL_ATLAS_ROOT must name the Atlas checkout")
    return Path(value)


class CaptureFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.source = self.root / "source"
        self.capture = self.root / "capture"
        self.record = self.root / "capture.json"
        self.atlas_root = atlas_root()
        self.runner = capture_source.atlas_process_tree(self.atlas_root)
        for path in (self.repository, self.source, self.capture):
            path.mkdir(parents=True)
        shared = {
            "Cargo.toml": b"[workspace]\nmembers=[]\n",
            "Cargo.lock": b"version = 4\n",
            "rust-toolchain.toml": b"[toolchain]\nchannel='stable'\n",
        }
        self.baseline = dict(shared, **{
            ".gitattributes": b"*.rs -text\n",
            "crates/demo/src/lib.rs": b"pub fn value() -> u8 { 1 }\r\n",
            "crates/deleted/Cargo.toml": b"[package]\nname='deleted'\nversion='0.1.0'\n",
        })
        self.captured = dict(shared, **{
            "crates/demo/src/lib.rs": b"pub fn value() -> u8 { 2 }\n",
            "crates/new/Cargo.toml": b"[package]\nname='new'\nversion='0.1.0'\n",
        })
        write_files(self.repository, self.baseline)
        write_files(self.source, self.captured)
        self.study = self.repository.joinpath(*capture_source.FIXTURE_PATH.parts)
        self.study.mkdir(parents=True)
        for index in range(capture_source.FIXTURE_IMAGES):
            (self.study / f"{index:03}.dcm").write_bytes(b"")
        self.git("init", "-b", "main")
        self.git("add", ".gitattributes", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "crates")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.org",
                 "commit", "-m", "Fixture")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("update-ref", "refs/remotes/origin/main", self.base)
        self.git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
        self.fixture = {"path": capture_source.FIXTURE_PATH.as_posix(),
                        "dicom_images": capture_source.FIXTURE_IMAGES}
        self.write_record()

    def git(self, *arguments):
        return capture_source.git(self.runner, self.repository, *arguments)

    def environment(self):
        return {"RITK_MANUAL_CAPTURE_DIR": str(self.capture),
                "RITK_MANUAL_STUDY": str(self.study)}

    def source_hashes(self):
        return {name: digest(path.read_bytes())
                for name, path in capture_source.source_files(self.source).items()}

    def dataset_hashes(self):
        return {path.relative_to(self.study).as_posix(): digest(path.read_bytes())
                for path in self.study.rglob("*") if path.is_file()}

    def execution_inputs(self):
        scripts = self.atlas_root / "scripts"
        runtime = {
            name: digest((scripts / name).read_bytes())
            for name in capture_source.ATLAS_RUNTIME_FILES
        }
        atlas_config = self.atlas_root / ".cargo/config.toml"
        return {
            "toolchain": runtime["process_tree.py"],
            "cargo_config": digest(atlas_config.read_bytes()),
            "environment": runtime["windows_process.py"],
            "atlas_config": digest(atlas_config.read_bytes()),
            "atlas_runtime": runtime,
        }

    def write_record(self, **changes):
        sources = self.source_hashes()
        dataset = self.dataset_hashes()
        execution = self.execution_inputs()
        stage = {"version": capture_source.CAPTURE_RECORD_VERSION,
                 "argv": ["cargo", "nextest", "run"], "exit": 0,
                 "admission": {"status": "accepted", "errors": []},
                 "environment": self.environment(),
                 "source_before": sources, "source_after": dict(sources),
                 "dataset_before": dataset, "dataset_after": dict(dataset),
                 "execution_before": execution, "execution_after": dict(execution)}
        stage.update(changes)
        self.record.write_text(json.dumps([stage]), encoding="utf-8")

    def validate(self):
        return capture_source.validate_capture_source(
            self.capture, self.record, self.source, self.repository,
            self.atlas_root, self.base, self.fixture
        )

    def close(self):
        self.temporary.cleanup()


class CaptureSourceTests(unittest.TestCase):
    def test_missing_atlas_runtime_is_a_typed_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(capture_source.CaptureSourceError, "Atlas runtime is missing"):
                capture_source.atlas_process_tree(Path(directory))

    def test_valid_record_binds_raw_git_bytes_and_every_override(self):
        fixture = CaptureFixture()
        self.addCleanup(fixture.close)
        result = fixture.validate()
        expected = {
            "crates/deleted/Cargo.toml": {
                "base_sha256": digest(fixture.baseline["crates/deleted/Cargo.toml"]),
                "capture_sha256": None},
            "crates/demo/src/lib.rs": {
                "base_sha256": digest(fixture.baseline["crates/demo/src/lib.rs"]),
                "capture_sha256": digest(fixture.captured["crates/demo/src/lib.rs"])},
            "crates/new/Cargo.toml": {
                "base_sha256": None,
                "capture_sha256": digest(fixture.captured["crates/new/Cargo.toml"])},
        }
        self.assertEqual(result["base_revision"], fixture.base)
        self.assertEqual(result["source_overrides"], expected)
        self.assertEqual(result["capture_record_sha256"], digest(fixture.record.read_bytes()))
        self.assertEqual(result["source_file_count"], len(fixture.source_hashes()))

    def test_invalid_record_cases_are_rejected_without_source_changes(self):
        fixture = CaptureFixture()
        self.addCleanup(fixture.close)
        environment = fixture.environment()
        root_type, study_type = dict(environment), dict(environment)
        root_type["RITK_MANUAL_CAPTURE_DIR"] = 7
        study_type["RITK_MANUAL_STUDY"] = 7
        sources = fixture.source_hashes()
        missing, typed, noncanonical = dict(sources), dict(sources), dict(sources)
        missing.pop("Cargo.lock")
        typed["Cargo.lock"] = 7
        noncanonical["Cargo.lock"] = "A" * 64
        cases = {
            "failed": {"exit": 1}, "false_exit": {"exit": False},
            "version": {"version": 2},
            "rejected": {"admission": {"status": "rejected", "errors": ["frame"]}},
            "empty_argv": {"argv": []}, "mutated": {"source_after": {}},
            "dataset_mutated": {"dataset_after": {}},
            "execution_mutated": {"execution_after": {}},
            "execution_typed_equal": {
                "execution_before": {"RUSTC": False},
                "execution_after": {"RUSTC": 0},
            },
            "execution_bool": {
                "execution_before": {"RUSTC": False},
                "execution_after": {"RUSTC": False},
            },
            "execution_incomplete": {
                "execution_before": {"toolchain": "rustc"},
                "execution_after": {"toolchain": "rustc"},
            },
            "missing": {"source_before": missing, "source_after": missing},
            "typed": {"source_before": typed, "source_after": typed},
            "noncanonical": {"source_before": noncanonical, "source_after": noncanonical},
            "root_type": {"environment": root_type}, "study_type": {"environment": study_type},
        }
        before = {name: path.read_bytes()
                  for name, path in capture_source.source_files(fixture.source).items()}
        for name, changes in cases.items():
            with self.subTest(case=name):
                fixture.write_record(**changes)
                with self.assertRaises(capture_source.CaptureSourceError):
                    fixture.validate()
                self.assertEqual(
                    {key: path.read_bytes()
                     for key, path in capture_source.source_files(fixture.source).items()},
                    before,
                )

    def test_frame_decoder_rejects_corrupt_format_and_dimensions(self):
        def image_bytes(size, format_name):
            output = io.BytesIO()
            mode = "RGB" if format_name == "JPEG" else "RGBA"
            Image.new(mode, size).save(output, format=format_name)
            return output.getvalue()

        valid = image_bytes(capture_source.FRAME_SIZE, "PNG")
        size, pixels = capture_source.decoded_rgba(valid, "frame.png", "PNG")
        self.assertEqual(size, capture_source.FRAME_SIZE)
        self.assertEqual(len(pixels), size[0] * size[1] * 4)
        cases = {
            "corrupt": b"not an image",
            "format": image_bytes(capture_source.FRAME_SIZE, "JPEG"),
            "dimensions": image_bytes((1, 1), "PNG"),
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(capture_source.CaptureSourceError):
                    capture_source.decoded_rgba(data, "frame.png", "PNG")

    def test_fixture_identity_requires_regular_dicom_files(self):
        for case in ("metadata", "directory"):
            with self.subTest(case=case):
                fixture = CaptureFixture()
                try:
                    if case == "metadata":
                        fixture.fixture["path"] = "test_data/other"
                    else:
                        (fixture.study / "000.dcm").unlink()
                        (fixture.study / "directory.dcm").mkdir()
                    with self.assertRaises(capture_source.CaptureSourceError):
                        fixture.validate()
                finally:
                    fixture.close()


if __name__ == "__main__":
    unittest.main()
