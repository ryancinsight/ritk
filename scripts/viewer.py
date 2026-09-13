"""Verify synthetic and saved-study DICOM workflows through the viewer hosts."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_CAPTURE_PIXELS = 16 * 1024 * 1024
# Four 8-bit channels per bounded pixel plus room for PNG chunk overhead.
MAX_CAPTURE_BYTES = MAX_CAPTURE_PIXELS * 4 + 1024 * 1024


def _paeth(left, above, upper_left):
    """Return the PNG Paeth predictor for one byte."""
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def png_summary(path):
    """Decode a bounded 8-bit PNG and return width, height, and non-black pixels.

    The visual smoke only emits PNGs from the RITK capture path. Keeping this
    decoder in the harness avoids an unpinned imaging dependency while still
    checking pixel values instead of asserting that a file merely exists.
    """
    content = path.read_bytes()
    if len(content) > MAX_CAPTURE_BYTES:
        raise ValueError(f"PNG capture exceeds the bounded image size: {path}")
    if not content.startswith(PNG_SIGNATURE):
        raise ValueError(f"capture is not a PNG: {path}")
    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    saw_iend = False
    while offset < len(content):
        if len(content) - offset < 12:
            raise ValueError(f"truncated PNG chunk header: {path}")
        length = struct.unpack_from(">I", content, offset)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + length
        crc_end = chunk_end + 4
        if chunk_end < chunk_start or crc_end > len(content):
            raise ValueError(f"PNG chunk exceeds capture bounds: {path}")
        chunk_type = content[offset + 4:offset + 8]
        payload = content[chunk_start:chunk_end]
        expected_crc = struct.unpack_from(">I", content, chunk_end)[0]
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"PNG chunk checksum mismatch: {path}")
        if chunk_type == b"IHDR":
            if len(payload) != 13 or width is not None:
                raise ValueError(f"invalid PNG header: {path}")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (width == 0 or height == 0 or width * height > MAX_CAPTURE_PIXELS
                    or bit_depth != 8 or color_type not in (2, 6)):
                raise ValueError(f"unsupported PNG format: {path}")
            if compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError(f"unsupported PNG encoding: {path}")
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            if payload:
                raise ValueError(f"invalid PNG end chunk: {path}")
            saw_iend = True
            offset = crc_end
            break
        offset = crc_end
    if width is None or not saw_iend or not compressed:
        raise ValueError(f"incomplete PNG capture: {path}")
    channels = 4 if color_type == 6 else 3
    row_bytes = width * channels
    expected_size = (row_bytes + 1) * height
    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise ValueError(f"invalid PNG image stream: {path}") from error
    if len(raw) != expected_size:
        raise ValueError(f"PNG image stream length does not match dimensions: {path}")
    previous = bytearray(row_bytes)
    non_black = 0
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        encoded = raw[cursor:cursor + row_bytes]
        cursor += row_bytes
        row = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth(left, above, upper_left)
            else:
                raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
            row[index] = (value + predictor) & 0xFF
        non_black += sum(
            1
            for pixel in range(0, row_bytes, channels)
            if any(row[pixel + channel] != 0 for channel in range(3))
        )
        previous = row
    return {"width": width, "height": height, "non_black_pixels": non_black}


def linked(path):
    """Include Windows junctions on Python 3.11, which has no Path.is_junction."""
    return path.is_symlink() or (path.exists() and bool(
        getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="compiled dicom_workflow example")
    parser.add_argument("--update-goldens", action="store_true", help="replace the reviewed manual images with this run")
    parser.add_argument("--native-binary", type=Path, help="also open and capture the real desktop viewer")
    parser.add_argument("--metis-native", action="store_true",
                        help="run the native binary through the Windows Métis host")
    parser.add_argument("--real-study", type=Path,
                        help="also capture a caller-supplied saved DICOM file or directory")
    parser.add_argument("--real-series-uid", metavar="UID",
                        help="select this SeriesInstanceUID for --real-study")
    arguments = parser.parse_args()
    if arguments.metis_native and not arguments.native_binary:
        parser.error("--metis-native requires --native-binary")
    if arguments.real_study and not arguments.native_binary:
        parser.error("--real-study requires --native-binary")
    if arguments.real_series_uid and not arguments.real_study:
        parser.error("--real-series-uid requires --real-study")
    destination = ROOT / "scratch" / "viewer"
    images = [f"{axis}{suffix}.png" for axis in ("depth", "row", "column", "fusion", "orientation")
              for suffix in ("", "-grid")]
    images.extend(
        f"color-{axis}{suffix}.png"
        for axis in ("depth", "row", "column")
        for suffix in ("", "-grid")
    )
    images.extend(("grayscale.png", "grayscale-grid.png"))
    for path in (destination.parent, destination):
        if linked(path):
            raise ValueError(f"refusing linked output directory: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    real_capture_name = None
    if arguments.real_study:
        real_capture_name = "real-metis-frame.png" if arguments.metis_native else "real-window.png"
    cleanup_names = [*images, "window.png", "metis-frame.png", "rejected.png", "metis-rejected.png", "workflow.json"]
    if real_capture_name:
        cleanup_names.append(real_capture_name)
    for name in cleanup_names:
        path = destination / name
        if linked(path) or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError(f"refusing linked output file: {path}")
    # Invalidate previous evidence before resolving or executing the new binary.
    (destination / "workflow.json").write_text('{"schema":1,"status":"failed"}\n', encoding="utf-8")
    # A run without a native host must not retain captures from an earlier run.
    (destination / "window.png").unlink(missing_ok=True)
    (destination / "metis-frame.png").unlink(missing_ok=True)
    if real_capture_name:
        (destination / real_capture_name).unlink(missing_ok=True)
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
            native_args = [str(native), str(output / "study")]
            if arguments.metis_native:
                native_args.extend(("--metis-native", "--capture", str(output / "metis-frame.png")))
            else:
                native_args.extend(("--capture", str(output / "window.png")))
            window = subprocess.run(native_args, capture_output=True, encoding="utf-8",
                                    timeout=60, check=True)
            # Exercise actual failure propagation without publishing a screenshot
            # of the empty viewer as evidence of successful study opening.
            rejected_name = "metis-rejected.png" if arguments.metis_native else "rejected.png"
            rejected_args = [str(native), str(output / "absent.dcm")]
            if arguments.metis_native:
                rejected_args.extend(("--metis-native", "--capture", str(output / rejected_name)))
            else:
                rejected_args.extend(("--capture", str(output / rejected_name)))
            rejected = subprocess.run(rejected_args, capture_output=True, encoding="utf-8",
                                      timeout=60, check=False)
            error_marker = "open initial RITK study" if arguments.metis_native else "initial study did not load"
            if rejected.returncode == 0 or (output / rejected_name).exists() or error_marker not in rejected.stderr:
                raise ValueError("native invalid-study capture did not reject explicitly")
            with native.open("rb") as executable:
                native_hash = hashlib.file_digest(executable, "sha256").hexdigest()
            report_key = "metis_native" if arguments.metis_native else "native"
            capture_name = "metis-frame.png" if arguments.metis_native else "window.png"
            metrics = png_summary(output / capture_name)
            if metrics["non_black_pixels"] == 0:
                raise ValueError(f"native capture contains no visible pixels: {capture_name}")
            images.append(capture_name)
            report[report_key] = {"binary_sha256": native_hash, "stdout": window.stdout,
                                  "stderr": window.stderr, "invalid_study_exit": rejected.returncode,
                                  "capture": capture_name, **metrics}
            if arguments.real_study:
                study = arguments.real_study.resolve(strict=True)
                real_args = [str(native), str(study)]
                if arguments.real_series_uid:
                    real_args.extend(("--series-instance-uid", arguments.real_series_uid))
                if arguments.metis_native:
                    real_args.extend(("--metis-native", "--capture-application", "--capture",
                                      str(output / real_capture_name)))
                else:
                    real_args.extend(("--capture", str(output / real_capture_name)))
                subprocess.run(real_args, capture_output=True, encoding="utf-8",
                               timeout=60, check=True)
                real_metrics = png_summary(output / real_capture_name)
                if real_metrics["non_black_pixels"] == 0:
                    raise ValueError(f"saved-study capture contains no visible pixels: {real_capture_name}")
                report["real_study"] = {
                    "input_kind": "directory" if study.is_dir() else "file",
                    "series_selected": bool(arguments.real_series_uid),
                    "capture": real_capture_name,
                    **real_metrics,
                }
                images.append(real_capture_name)
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
        for axis in ("depth", "row", "column"):
            golden = golden_root / f"dicom-color-{axis}.png"
            if linked(golden) or (golden.exists() and golden.stat().st_nlink != 1):
                raise ValueError(f"refusing linked golden file: {golden}")
            actual = (output / f"color-{axis}-grid.png").read_bytes()
            if arguments.update_goldens:
                golden.write_bytes(actual)
            elif golden.read_bytes() != actual:
                raise ValueError(f"render differs from reviewed manual image: {golden}")
        golden = golden_root / "dicom-grayscale.png"
        if linked(golden) or (golden.exists() and golden.stat().st_nlink != 1):
            raise ValueError(f"refusing linked golden file: {golden}")
        actual = (output / "grayscale-grid.png").read_bytes()
        if arguments.update_goldens:
            golden.write_bytes(actual)
        elif golden.read_bytes() != actual:
            raise ValueError(f"render differs from reviewed manual image: {golden}")
        if arguments.metis_native:
            golden = golden_root / "dicom-metis-native.png"
            if linked(golden) or (golden.exists() and golden.stat().st_nlink != 1):
                raise ValueError(f"refusing linked golden file: {golden}")
            actual = (output / "metis-frame.png").read_bytes()
            if arguments.update_goldens:
                golden.write_bytes(actual)
            elif golden.read_bytes() != actual:
                raise ValueError(f"native render differs from reviewed manual image: {golden}")
        (destination / "workflow.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(destination / "workflow.json")


if __name__ == "__main__":
    main()
