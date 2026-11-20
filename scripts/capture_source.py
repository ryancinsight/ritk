"""Bind a manual capture to its exact RITK source and public fixture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath


FIXTURE_PATH = PurePosixPath("test_data/2_head_mri_t2/DICOM")
FIXTURE_IMAGES = 94
GIT_TIMEOUT_SECONDS = 30


class CaptureSourceError(RuntimeError):
    """Report capture provenance that cannot be established."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atlas_module(scripts: Path, name: str):
    path = scripts / f"{name}.py"
    if not path.is_file():
        raise CaptureSourceError(f"Atlas runtime is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CaptureSourceError(f"cannot load Atlas runtime: {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def atlas_process_tree(atlas_root: Path):
    """Load the exact bounded runner from an explicit Atlas checkout."""
    scripts = atlas_root.resolve() / "scripts"
    loaded = None
    for name in ("darwin_process", "windows_process", "process_tree"):
        loaded = _atlas_module(scripts, name)
    return loaded


def atlas_build_source(atlas_root: Path):
    """Load Atlas build-source helpers from the explicit checkout."""
    scripts = atlas_root.resolve() / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        return _atlas_module(scripts, "atlas_build_source")
    finally:
        sys.path.pop(0)


def git(runner, repository: Path, *arguments: str):
    result = runner.run(
        ["git", "--no-replace-objects", "-C", str(repository), *arguments],
        timeout=GIT_TIMEOUT_SECONDS,
    )
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    if result.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise CaptureSourceError(f"git {' '.join(arguments)} failed: {detail}")
    return subprocess.CompletedProcess(result.args, result.returncode, stdout, stderr)


def is_capture_input(path: PurePosixPath) -> bool:
    return (
        (len(path.parts) > 1 and path.parts[0] == "crates" and path.suffix == ".rs")
        or path.name == "Cargo.toml"
        or path.as_posix() in {"Cargo.lock", "rust-toolchain.toml"}
    )


def source_files(source_root: Path) -> dict[str, Path]:
    files = {}
    for path in source_root.rglob("*"):
        relative = PurePosixPath(path.relative_to(source_root).as_posix())
        if path.is_file() and is_capture_input(relative):
            files[relative.as_posix()] = path
    return files


def load_stage(record_bytes: bytes, capture_root: Path) -> tuple[dict, dict[str, str]]:
    try:
        records = json.loads(record_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureSourceError("capture record is not valid UTF-8 JSON") from error
    if not isinstance(records, list):
        raise CaptureSourceError("capture record must contain a JSON array")
    matching = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("environment"), dict):
            continue
        recorded_root = record["environment"].get("RITK_MANUAL_CAPTURE_DIR")
        if isinstance(recorded_root, str) and Path(recorded_root).resolve() == capture_root.resolve():
            matching.append(record)
    if not matching:
        raise CaptureSourceError("capture record has no stage for the capture root")
    stage = matching[-1]
    if type(stage.get("exit")) is not int or stage["exit"] != 0:
        raise CaptureSourceError("capture stage did not complete successfully")
    command = stage.get("argv")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
        raise CaptureSourceError("capture stage has no command argument record")
    before, after = stage.get("source_before"), stage.get("source_after")
    if not isinstance(before, dict) or not before or before != after:
        raise CaptureSourceError("capture source hashes are missing or changed during capture")
    return stage, before


def verify_source(source_root: Path, expected: dict[str, str]) -> None:
    actual = source_files(source_root)
    if set(actual) != set(expected):
        raise CaptureSourceError("capture source coverage differs from the source tree")
    for name, path in actual.items():
        if sha256(path.read_bytes()) != expected[name]:
            raise CaptureSourceError(f"source changed after capture: {name}")


def base_inputs(runner, repository: Path, revision: str) -> dict[str, str]:
    listing = git(runner, repository, "ls-tree", "-r", "--name-only", revision).stdout.splitlines()
    names = [name for name in listing if is_capture_input(PurePosixPath(name))]
    request = "".join(f"{revision}:{name}\n" for name in names).encode()
    result = runner.run(
        ["git", "--no-replace-objects", "-C", str(repository), "cat-file", "--batch"],
        input=request,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise CaptureSourceError(
            f"git cat-file --batch failed: {result.stderr.decode('utf-8').strip()}"
        )
    stream = memoryview(result.stdout)
    digests = {}
    offset = 0
    for name in names:
        line_end = result.stdout.find(b"\n", offset)
        header = bytes(stream[offset:line_end]).split()
        if line_end < 0 or len(header) != 3 or header[1] != b"blob":
            raise CaptureSourceError(f"git returned no blob for base input: {name}")
        offset = line_end + 1
        size = int(header[2])
        data = bytes(stream[offset:offset + size])
        offset += size
        if len(data) != size or bytes(stream[offset:offset + 1]) != b"\n":
            raise CaptureSourceError(f"git returned a truncated base input: {name}")
        offset += 1
        digests[name] = sha256(data)
    if offset != len(stream):
        raise CaptureSourceError("git returned unexpected base input data")
    return digests


def resolve_base(runner, repository: Path, base_rev: str) -> str:
    base = git(runner, repository, "rev-parse", "--verify", f"{base_rev}^{{commit}}").stdout.strip()
    default = git(runner, repository, "rev-parse", "--verify", "refs/remotes/origin/HEAD^{commit}").stdout.strip()
    if git(runner, repository, "merge-base", base, default).stdout.strip() != base:
        raise CaptureSourceError("base revision is not an ancestor of fetched origin/HEAD")
    return base


def source_overrides(runner, repository: Path, base: str, captured: dict[str, str]) -> dict:
    baseline = base_inputs(runner, repository, base)
    return {
        path: {"base_sha256": baseline.get(path), "capture_sha256": captured.get(path)}
        for path in sorted(baseline.keys() | captured.keys())
        if baseline.get(path) != captured.get(path)
    }


def verify_fixture(repository: Path, fixture: object, study: object) -> None:
    expected = {"path": FIXTURE_PATH.as_posix(), "dicom_images": FIXTURE_IMAGES}
    fixture_path = repository.joinpath(*FIXTURE_PATH.parts).resolve()
    if fixture != expected:
        raise CaptureSourceError("manual metadata does not declare the public MRI fixture")
    if not isinstance(study, str) or Path(study).resolve() != fixture_path:
        raise CaptureSourceError("capture study is not the public MRI fixture")
    images = [path for path in fixture_path.rglob("*") if path.is_file() and path.suffix.lower() == ".dcm"]
    if len(images) != FIXTURE_IMAGES:
        raise CaptureSourceError("public MRI fixture does not contain 94 .dcm files")


def validate_capture_source(
    capture_root: Path,
    capture_record: Path,
    source_root: Path,
    repository: Path,
    atlas_root: Path,
    base_rev: str,
    fixture: object,
) -> dict:
    """Validate capture source, fixture, and Git baseline before image comparison."""
    record_bytes = capture_record.read_bytes()
    stage, captured = load_stage(record_bytes, capture_root)
    verify_source(source_root, captured)
    verify_fixture(repository, fixture, stage["environment"].get("RITK_MANUAL_STUDY"))
    runner = atlas_process_tree(atlas_root)
    base = resolve_base(runner, repository, base_rev)
    return {
        "stage": stage,
        "base_revision": base,
        "source_overrides": source_overrides(runner, repository, base, captured),
        "capture_record_sha256": sha256(record_bytes),
        "source_map_sha256": sha256(
            json.dumps(captured, sort_keys=True, separators=(",", ":")).encode()
        ),
        "source_file_count": len(captured),
    }
