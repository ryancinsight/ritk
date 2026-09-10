"""Verify synthetic DICOM and patient-coordinate fusion workflows."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def linked(path):
    """Include Windows junctions on Python 3.11, which has no Path.is_junction."""
    return path.is_symlink() or (path.exists() and bool(
        getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="compiled dicom_workflow example")
    parser.add_argument("--update-goldens", action="store_true", help="replace the reviewed manual images with this run")
    parser.add_argument("--native-binary", type=Path, help="also open and capture the real desktop viewer")
    arguments = parser.parse_args()
    destination = ROOT / "scratch" / "viewer"
    images = [f"{axis}{suffix}.png" for axis in ("depth", "row", "column", "fusion", "orientation")
              for suffix in ("", "-grid")]
    for path in (destination.parent, destination):
        if linked(path):
            raise ValueError(f"refusing linked output directory: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in [*images, "window.png", "workflow.json"]:
        path = destination / name
        if linked(path) or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError(f"refusing linked output file: {path}")
    # Invalidate previous evidence before resolving or executing the new binary.
    (destination / "workflow.json").write_text('{"schema":1,"status":"failed"}\n', encoding="utf-8")
    # A headless run must not retain a native capture from an earlier run.
    (destination / "window.png").unlink(missing_ok=True)
    binary = arguments.binary.resolve(strict=True)
    # A fresh directory prevents old images from becoming evidence for this run.
    # Keep only the fixed, small artifact set; temporary study bytes are removed
    # by TemporaryDirectory, so successive invocations cannot grow storage.
    with tempfile.TemporaryDirectory(prefix="ritk-viewer-") as temporary:
        output = Path(temporary)
        result = subprocess.run([str(binary), str(output)], capture_output=True,
                                encoding="utf-8", timeout=60, check=True)
        report = json.loads((output / "workflow.json").read_text(encoding="utf-8"))
        if arguments.native_binary:
            native = arguments.native_binary.resolve(strict=True)
            window = subprocess.run([str(native), str(output / "study"), "--capture", str(output / "window.png")],
                                    capture_output=True, encoding="utf-8", timeout=60, check=True)
            # Exercise actual failure propagation without publishing a screenshot
            # of the empty viewer as evidence of successful study opening.
            rejected = subprocess.run([str(native), str(output / "absent.dcm"), "--capture", str(output / "rejected.png")],
                                      capture_output=True, encoding="utf-8", timeout=60, check=False)
            if rejected.returncode == 0 or (output / "rejected.png").exists() or "initial study did not load" not in rejected.stderr:
                raise ValueError("native invalid-study capture did not reject explicitly")
            images.append("window.png")
            with native.open("rb") as executable:
                native_hash = hashlib.file_digest(executable, "sha256").hexdigest()
            report["native"] = {"binary_sha256": native_hash, "stdout": window.stdout,
                                "stderr": window.stderr, "invalid_study_exit": rejected.returncode}
        report["sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                            for name in images}
        with binary.open("rb") as executable:
            report["binary_sha256"] = hashlib.file_digest(executable, "sha256").hexdigest()
        report["status"] = "passed"
        report["stdout"] = result.stdout
        report["stderr"] = result.stderr
        for name in images:
            (destination / name).write_bytes((output / name).read_bytes())
        golden_root = ROOT / "docs" / "manual" / "images"
        for path in (ROOT / "docs", golden_root.parent, golden_root):
            if linked(path):
                raise ValueError(f"refusing linked golden directory: {path}")
        golden_root.mkdir(parents=True, exist_ok=True)
        for axis in ("depth", "row", "column", "fusion", "orientation"):
            golden = golden_root / f"dicom-{axis}.png"
            if linked(golden) or (golden.exists() and golden.stat().st_nlink != 1):
                raise ValueError(f"refusing linked golden file: {golden}")
            actual = (output / f"{axis}-grid.png").read_bytes()
            if arguments.update_goldens:
                golden.write_bytes(actual)
            elif golden.read_bytes() != actual:
                raise ValueError(f"render differs from reviewed manual image: {golden}")
        (destination / "workflow.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(destination / "workflow.json")


if __name__ == "__main__":
    main()
