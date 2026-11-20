"""Update source-bound AsterSlice manual images from a completed capture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path, PurePosixPath

from PIL import Image, __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "metis" / "scripts"))
from process_tree import run  # noqa: E402


FRAME_STEMS = ("dicom-native-slices", "dicom-native-calibration",
               "dicom-native-annotations", "dicom-native-length-edit")
FRAME_SIZE = (1280, 800)
FRAME_BYTE_LIMIT = 200_000
GIT_TIMEOUT_SECONDS = 30


class UpdateError(RuntimeError):
    """Report an invalid capture without changing manual assets."""

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def git(repository: Path, *arguments: str, allow_failure: bool = False):
    result = run(
        ["git", "--no-replace-objects", "-C", str(repository), *arguments],
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if result.returncode != 0 and not allow_failure:
        detail = result.stderr.strip() or result.stdout.strip()
        raise UpdateError(f"git {' '.join(arguments)} failed: {detail}")
    return result

def capture_input(path: PurePosixPath) -> bool:
    return (
        (len(path.parts) > 1 and path.parts[0] == "crates" and path.suffix == ".rs")
        or path.name == "Cargo.toml"
        or path.as_posix() in {"Cargo.lock", "rust-toolchain.toml"}
    )

def source_files(source_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in source_root.rglob("*"):
        if path.is_file():
            relative = PurePosixPath(path.relative_to(source_root).as_posix())
            if capture_input(relative):
                files[relative.as_posix()] = path
    return files

def load_stage(record_bytes: bytes, capture_root: Path) -> tuple[dict, dict[str, str]]:
    try:
        records = json.loads(record_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpdateError("capture record is not valid UTF-8 JSON") from error
    if not isinstance(records, list):
        raise UpdateError("capture record must contain a JSON array")
    matching = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("environment"), dict):
            continue
        recorded_root = record["environment"].get("RITK_MANUAL_CAPTURE_DIR")
        if recorded_root and Path(recorded_root).resolve() == capture_root.resolve():
            matching.append(record)
    if not matching:
        raise UpdateError("capture record has no stage for --capture-root")
    stage = matching[-1]
    if stage.get("exit") != 0:
        raise UpdateError("capture stage did not complete successfully")
    if not isinstance(stage.get("argv"), list) or not all(isinstance(value, str) for value in stage["argv"]):
        raise UpdateError("capture stage has no command argument record")
    before, after = stage.get("source_before"), stage.get("source_after")
    if not isinstance(before, dict) or not before or before != after:
        raise UpdateError("capture source hashes are missing or changed during capture")
    if not all(
        isinstance(path, str)
        and isinstance(digest, str)
        and len(digest) == 64
        and digest == digest.lower()
        and all(character in "0123456789abcdef" for character in digest)
        for path, digest in before.items()
    ):
        raise UpdateError("capture source map contains an invalid SHA-256 entry")
    return stage, before

def verify_source(source_root: Path, expected: dict[str, str]) -> None:
    actual = source_files(source_root)
    if set(actual) != set(expected):
        missing = sorted(set(actual) - set(expected))
        stale = sorted(set(expected) - set(actual))
        raise UpdateError(f"capture source coverage differs: unrecorded={missing}, missing={stale}")
    for name, digest in expected.items():
        if sha256(actual[name].read_bytes()) != digest:
            raise UpdateError(f"source changed after capture: {name}")

def base_inputs(repository: Path, revision: str) -> dict[str, str]:
    listing = git(repository, "ls-tree", "-r", "--name-only", revision).stdout.splitlines()
    inputs = {}
    for name in listing:
        if capture_input(PurePosixPath(name)):
            blob = git(repository, "show", f"{revision}:{name}").stdout.encode()
            inputs[name] = sha256(blob)
    return inputs

def resolve_base(repository: Path, base_rev: str) -> str:
    base = git(repository, "rev-parse", "--verify", f"{base_rev}^{{commit}}").stdout.strip()
    default = git(
        repository, "rev-parse", "--verify", "refs/remotes/origin/HEAD^{commit}"
    ).stdout.strip()
    ancestry = git(repository, "merge-base", "--is-ancestor", base, default, allow_failure=True)
    if ancestry.returncode != 0:
        raise UpdateError("--base-rev is not an ancestor of fetched origin/HEAD")
    return base

def source_overrides(repository: Path, base: str, captured: dict[str, str]) -> dict:
    baseline = base_inputs(repository, base)
    changed = [
        {"path": path, "base_sha256": baseline[path], "capture_sha256": captured[path]}
        for path in sorted(captured.keys() & baseline.keys())
        if captured[path] != baseline[path]
    ]
    new = [
        {"path": path, "capture_sha256": captured[path]}
        for path in sorted(captured.keys() - baseline.keys())
    ]
    deleted = [
        {"path": path, "base_sha256": baseline[path]}
        for path in sorted(baseline.keys() - captured.keys())
    ]
    return {"changed": changed, "new": new, "deleted": deleted}

def rgba(data: bytes, name: str) -> tuple[tuple[int, int], bytes]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.size, image.convert("RGBA").tobytes()
    except OSError as error:
        raise UpdateError(f"cannot decode image: {name}") from error

def encode_webp(size: tuple[int, int], pixels: bytes) -> bytes:
    image = Image.frombytes("RGBA", size, pixels)
    output = io.BytesIO()
    image.save(output, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    return output.getvalue()

def prepare_frames(capture_root: Path, image_root: Path) -> tuple[dict, dict[str, bytes]]:
    records, writes, unchanged, reencoded = {}, {}, [], []
    for stem in FRAME_STEMS:
        png_path = capture_root / f"{stem}.png"
        webp_path = image_root / f"{stem}.webp"
        if not png_path.is_file() or not webp_path.is_file():
            raise UpdateError(f"required frame is missing: {stem}")
        png, webp = png_path.read_bytes(), webp_path.read_bytes()
        png_size, png_pixels = rgba(png, png_path.name)
        webp_size, webp_pixels = rgba(webp, webp_path.name)
        if png_size != FRAME_SIZE:
            raise UpdateError(f"capture frame has wrong dimensions: {png_path.name}")
        if webp_size == png_size and webp_pixels == png_pixels:
            encoded, action = webp, "retained_existing_bytes"
            unchanged.append(webp_path.name)
        else:
            encoded, action = encode_webp(png_size, png_pixels), "encoded_lossless"
            if rgba(encoded, webp_path.name) != (png_size, png_pixels):
                raise UpdateError(f"encoded frame does not preserve RGBA pixels: {webp_path.name}")
            writes[webp_path.name] = encoded
            reencoded.append(webp_path.name)
        if len(encoded) > FRAME_BYTE_LIMIT:
            raise UpdateError(f"frame exceeds {FRAME_BYTE_LIMIT} bytes: {webp_path.name}")
        encoding = {
            "tool": "Pillow",
            "version": pillow_version,
            "action": action,
            "decoded_rgba_equal": True,
        }
        if action == "encoded_lossless":
            encoding.update(format="WEBP", lossless=True, quality=100, method=6, exact=True)
        records[webp_path.name] = {
            "width_px": png_size[0],
            "height_px": png_size[1],
            "webp_bytes": len(encoded),
            "webp_sha256": sha256(encoded),
            "source_png_bytes": len(png),
            "source_png_sha256": sha256(png),
            "encoding": encoding,
        }
    comparison = {
        "method": "Pillow Image.convert('RGBA').tobytes() byte equality",
        "unchanged_frames": unchanged,
        "reencoded_frames": reencoded,
    }
    return {"frames": records, "comparison": comparison}, writes

def update(args) -> None:
    capture_root, source_root = args.capture_root.resolve(), args.source_root.resolve()
    repository, record_path = args.repository.resolve(), args.capture_record.resolve()
    image_root = source_root / "docs" / "manual" / "images"
    metadata_path = image_root / "dicom-native-annotations.json"
    existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    record_bytes = record_path.read_bytes()
    stage, captured = load_stage(record_bytes, capture_root)
    verify_source(source_root, captured)
    fixture = existing.get("fixture")
    if not isinstance(fixture, dict) or fixture.get("dicom_images") != 94:
        raise UpdateError("manual metadata does not declare the 94-image MRI fixture")
    fixture_path = (repository / str(fixture.get("path", ""))).resolve()
    study = stage["environment"].get("RITK_MANUAL_STUDY")
    if not study or Path(study).resolve() != fixture_path:
        raise UpdateError("capture study is not the authoritative manual fixture")
    if len([path for path in fixture_path.rglob("*") if path.suffix.lower() == ".dcm"]) != 94:
        raise UpdateError("authoritative manual fixture does not contain 94 .dcm files")
    base = resolve_base(repository, args.base_rev)
    prepared, writes = prepare_frames(capture_root, image_root)
    coverage_bytes = json.dumps(captured, sort_keys=True, separators=(",", ":")).encode()
    metadata = {
        "viewer_base_revision": base,
        "viewer_source_overrides": source_overrides(repository, base, captured),
        "capture_record_sha256": sha256(record_bytes),
        "capture_input_coverage": {
            "description": "All crates/**/*.rs, all Cargo.toml files, Cargo.lock, and rust-toolchain.toml used by the capture",
            "file_count": len(captured),
            "source_map_sha256": sha256(coverage_bytes),
        },
        "fixture": fixture,
        "capture": existing.get("capture", "Native Windows RITK client framebuffer"),
        "capture_command": stage.get("argv"),
        "comparison": prepared["comparison"],
        "frames": prepared["frames"],
    }
    for name, data in writes.items():
        (image_root / name).write_bytes(data)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Validate a completed AsterSlice capture and update its four manual WebP frames.")
    value.add_argument("--capture-root", required=True, type=Path, help="Directory containing the four fresh 1280x800 PNG captures")
    value.add_argument("--capture-record", required=True, type=Path, help="Completed capture-checks JSON emitted by the bounded capture runner")
    value.add_argument("--source-root", required=True, type=Path, help="Captured RITK source tree and manual output root")
    value.add_argument("--repository", required=True, type=Path, help="Authoritative fetched RITK Git repository and fixture root")
    value.add_argument("--base-rev", required=True, help="Git base commit used to describe captured source overrides")
    return value

def main(arguments: list[str] | None = None) -> int:
    try:
        update(parser().parse_args(arguments))
    except (UpdateError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
