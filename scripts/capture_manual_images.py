"""Run the bounded AsterSlice capture and record its exact inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import capture_source  # noqa: E402


CAPTURE_TIMEOUT_SECONDS = 600
OUTPUT_BYTE_LIMIT = 16 * 1024 * 1024
FRAME_STEMS = (
    "dicom-native-slices",
    "dicom-native-calibration",
    "dicom-native-annotations",
    "dicom-native-length-edit",
)


class CaptureCommandError(RuntimeError):
    """Report a capture that cannot produce trustworthy evidence."""


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): capture_source.sha256(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def source_hashes(source_root: Path) -> dict[str, str]:
    return {
        name: capture_source.sha256(path.read_bytes())
        for name, path in capture_source.source_files(source_root).items()
    }


def capture_command(config: Path, source_root: Path) -> list[str]:
    return [
        "cargo", "--config", str(config), "nextest", "run",
        "--manifest-path", str(source_root / "Cargo.toml"), "--locked",
        "-p", "ritk-snap", "--features", "eframe-shell",
        "-E", "test(native_manual_annotation_frames)", "--profile", "default",
    ]


def checked_version(runner, command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = runner.run(
        command, cwd=cwd, env=environment, timeout=30,
        stdout_limit=64 * 1024, stderr_limit=64 * 1024,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
        raise CaptureCommandError(f"{' '.join(command)} failed: {detail}")
    return (result.stdout or b"").decode("utf-8", errors="strict").strip()


def validate_paths(args) -> tuple[Path, Path, Path, Path, Path]:
    atlas_root = args.atlas_root.resolve()
    source_root = args.source_root.resolve()
    repository = args.repository.resolve()
    capture_root = args.capture_root.resolve()
    record = args.capture_record.resolve()
    for name, path in (("Atlas root", atlas_root), ("source root", source_root),
                       ("repository", repository)):
        if not path.is_dir():
            raise CaptureCommandError(f"{name} is not a directory: {path}")
    if not (source_root / "Cargo.toml").is_file():
        raise CaptureCommandError(f"source root has no Cargo.toml: {source_root}")
    if not (repository / ".git").exists():
        raise CaptureCommandError(f"repository has no Git metadata: {repository}")
    if capture_root.exists():
        raise CaptureCommandError(f"capture root already exists: {capture_root}")
    if record == capture_root or capture_root in record.parents:
        raise CaptureCommandError("capture record must be outside the capture root")
    streams = (record.with_suffix(record.suffix + ".stdout"),
               record.with_suffix(record.suffix + ".stderr"))
    if record.exists() or any(path.exists() for path in streams):
        raise CaptureCommandError(f"capture record output already exists: {record}")
    return atlas_root, source_root, repository, capture_root, record


def run_capture(args) -> None:
    atlas_root, source_root, repository, capture_root, record = validate_paths(args)
    study = repository.joinpath(*capture_source.FIXTURE_PATH.parts).resolve()
    fixture = {"path": capture_source.FIXTURE_PATH.as_posix(),
               "dicom_images": capture_source.FIXTURE_IMAGES}
    capture_source.verify_fixture(repository, fixture, str(study))
    before = source_hashes(source_root)
    if not before:
        raise CaptureCommandError("capture source input set is empty")
    dataset_before = file_hashes(study)
    toolchain_path = source_root / "rust-toolchain.toml"
    with toolchain_path.open("rb") as stream:
        toolchain = tomllib.load(stream).get("toolchain", {}).get("channel")
    if not isinstance(toolchain, str) or not toolchain:
        raise CaptureCommandError(f"toolchain channel is missing: {toolchain_path}")
    runner = capture_source.atlas_process_tree(atlas_root)
    builder = capture_source.atlas_build_source(atlas_root)
    try:
        config_text = builder.cargo_config_without_patches(atlas_root / ".cargo/config.toml")
    except RuntimeError as error:
        raise CaptureCommandError(str(error)) from error
    environment = os.environ.copy()
    environment.update({
        "CARGO_TARGET_DIR": str(atlas_root / "target"),
        "CARGO_TERM_COLOR": "never",
        "RUSTUP_TOOLCHAIN": toolchain,
        "RITK_MANUAL_CAPTURE_DIR": str(capture_root),
        "RITK_MANUAL_STUDY": str(study),
    })
    with tempfile.TemporaryDirectory() as neutral_name:
        neutral = Path(neutral_name)
        config = neutral / "config.toml"
        config.write_text(config_text, encoding="utf-8", newline="\n")
        compiler = {
            "cargo": checked_version(runner, ["cargo", "--version", "--verbose"], neutral, environment),
            "rustc": checked_version(runner, ["rustc", "--version", "--verbose"], neutral, environment),
            "toolchain": toolchain,
            "toolchain_sha256": capture_source.sha256(toolchain_path.read_bytes()),
        }
        capture_root.mkdir(parents=True)
        record.parent.mkdir(parents=True, exist_ok=True)
        command = capture_command(config, source_root)
        started = datetime.now(timezone.utc).isoformat()
        clock = time.monotonic()
        failure = None
        try:
            result = runner.run(
                command, cwd=neutral, env=environment,
                timeout=CAPTURE_TIMEOUT_SECONDS,
                stdout_limit=OUTPUT_BYTE_LIMIT, stderr_limit=OUTPUT_BYTE_LIMIT,
            )
            stdout, stderr, exit_code = result.stdout or b"", result.stderr or b"", result.returncode
        except (runner.ProcessTreeTimeout, runner.ProcessTreeOutputLimit) as error:
            stdout, stderr, exit_code = error.stdout or b"", error.stderr or b"", None
            failure = f"{type(error).__name__}: {error}"
        finished = datetime.now(timezone.utc).isoformat()
        seconds = time.monotonic() - clock
        stdout_path = record.with_suffix(record.suffix + ".stdout")
        stderr_path = record.with_suffix(record.suffix + ".stderr")
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        after = source_hashes(source_root)
        dataset_after = file_hashes(study)
        captures = file_hashes(capture_root)
        runtime = {
            name: capture_source.sha256((atlas_root / "scripts" / name).read_bytes())
            for name in ("process_tree.py", "windows_process.py", "darwin_process.py",
                         "atlas_build_source.py")
        }
        stage = {
            "argv": command,
            "cwd": str(neutral),
            "environment": {name: environment[name] for name in (
                "CARGO_TARGET_DIR", "CARGO_TERM_COLOR", "RUSTUP_TOOLCHAIN",
                "RITK_MANUAL_CAPTURE_DIR", "RITK_MANUAL_STUDY",
            )},
            "started_utc": started, "finished_utc": finished,
            "seconds": seconds, "exit": exit_code,
            "source_before": before, "source_after": after,
            "dataset_before": dataset_before, "dataset_after": dataset_after,
            "capture_files": captures, "compiler": compiler,
            "config_sha256": capture_source.sha256(config.read_bytes()),
            "atlas_config_sha256": capture_source.sha256(
                (atlas_root / ".cargo/config.toml").read_bytes()
            ),
            "atlas_runtime": runtime,
            "stdout": {"path": str(stdout_path), "bytes": len(stdout),
                       "sha256": capture_source.sha256(stdout)},
            "stderr": {"path": str(stderr_path), "bytes": len(stderr),
                       "sha256": capture_source.sha256(stderr)},
        }
        if failure is not None:
            stage["failure"] = failure
        record.write_text(json.dumps([stage], indent=2) + "\n", encoding="utf-8", newline="\n")
    expected = {f"{stem}.png" for stem in FRAME_STEMS}
    if exit_code != 0:
        raise CaptureCommandError(f"capture command failed; inspect {record}")
    if before != after or dataset_before != dataset_after:
        raise CaptureCommandError(f"capture inputs changed; inspect {record}")
    if set(captures) != expected:
        raise CaptureCommandError(f"capture output set is incomplete; inspect {record}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the bounded native AsterSlice manual capture and write its source record."
    )
    value.add_argument("--atlas-root", required=True, type=Path)
    value.add_argument("--source-root", required=True, type=Path)
    value.add_argument("--repository", required=True, type=Path)
    value.add_argument("--capture-root", required=True, type=Path)
    value.add_argument("--capture-record", required=True, type=Path)
    return value


def main(arguments: list[str] | None = None) -> int:
    try:
        run_capture(parser().parse_args(arguments))
    except (CaptureCommandError, capture_source.CaptureSourceError, OSError,
            UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
