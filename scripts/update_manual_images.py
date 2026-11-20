"""Update four AsterSlice manual images from a source-bound capture."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image, __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import capture_source  # noqa: E402


FRAME_BYTE_LIMIT = 200_000


class ImageUpdateError(RuntimeError):
    """Report invalid manual image input before any output write."""


def encode_webp(size: tuple[int, int], pixels: bytes) -> bytes:
    image = Image.frombytes("RGBA", size, pixels)
    output = io.BytesIO()
    image.save(output, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    return output.getvalue()


def prepare_frames(
    frames: tuple[tuple[str, bytes, tuple[int, int], bytes], ...],
    image_root: Path,
) -> tuple[dict, dict[str, bytes]]:
    records, writes, unchanged, reencoded = {}, {}, [], []
    for png_name, png, png_size, png_pixels in frames:
        stem = Path(png_name).stem
        webp_path = image_root / f"{stem}.webp"
        if not webp_path.is_file():
            raise ImageUpdateError(f"required frame is missing: {stem}")
        webp = webp_path.read_bytes()
        webp_size, webp_pixels = capture_source.decoded_rgba(webp, webp_path.name, "WEBP")
        if webp_size == png_size and webp_pixels == png_pixels:
            encoded = webp
            unchanged.append(webp_path.name)
        else:
            encoded = encode_webp(png_size, png_pixels)
            if capture_source.decoded_rgba(
                encoded, webp_path.name, "WEBP"
            ) != (png_size, png_pixels):
                raise ImageUpdateError(
                    f"encoded frame does not preserve RGBA pixels: {webp_path.name}"
                )
            writes[webp_path.name] = encoded
            reencoded.append(webp_path.name)
        if len(encoded) > FRAME_BYTE_LIMIT:
            raise ImageUpdateError(
                f"frame exceeds {FRAME_BYTE_LIMIT} bytes: {webp_path.name}"
            )
        records[webp_path.name] = {
            "width_px": png_size[0],
            "height_px": png_size[1],
            "webp_bytes": len(encoded),
            "webp_sha256": capture_source.sha256(encoded),
            "source_png_bytes": len(png),
            "source_png_sha256": capture_source.sha256(png),
        }
    comparison = {
        "method": "Pillow Image.convert('RGBA').tobytes() byte equality",
        "unchanged_frames": unchanged,
        "reencoded_frames": reencoded,
    }
    return {"frames": records, "comparison": comparison}, writes


def update(args) -> None:
    capture_root = args.capture_root.resolve()
    source_root = args.source_root.resolve()
    repository = args.repository.resolve()
    image_root = source_root / "docs" / "manual" / "images"
    metadata_path = image_root / "dicom-native-annotations.json"
    existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    binding = capture_source.validate_capture_source(
        capture_root,
        args.capture_record.resolve(),
        source_root,
        repository,
        args.atlas_root.resolve(),
        args.base_rev,
        existing.get("fixture"),
    )
    prepared, writes = prepare_frames(binding["capture_frames"], image_root)
    metadata = {
        "viewer_base_revision": binding["base_revision"],
        "viewer_source_overrides": binding["source_overrides"],
        "capture_record_sha256": binding["capture_record_sha256"],
        "capture_input_coverage": {
            "description": "All crates/**/*.rs, all Cargo.toml files, Cargo.lock, and rust-toolchain.toml used by the capture",
            "file_count": binding["source_file_count"],
            "source_map_sha256": binding["source_map_sha256"],
        },
        "fixture": existing["fixture"],
        "capture": existing.get("capture", "Native Windows RITK client framebuffer"),
        "capture_command": binding["stage"]["argv"],
        "image_tools": {
            "Pillow": {
                "version": pillow_version,
                "applies_to": "comparison.reencoded_frames",
                "format": "WEBP",
                "lossless": True,
                "quality": 100,
                "method": 6,
                "exact": True,
            }
        },
        "comparison": prepared["comparison"],
        "frames": prepared["frames"],
    }
    for name, data in writes.items():
        (image_root / name).write_bytes(data)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Validate a completed AsterSlice capture and update its four manual WebP frames."
    )
    value.add_argument("--capture-root", required=True, type=Path,
                       help="directory containing the four fresh 1280x800 PNG captures")
    value.add_argument("--capture-record", required=True, type=Path,
                       help="capture-checks JSON emitted by the bounded capture runner")
    value.add_argument("--source-root", required=True, type=Path,
                       help="captured RITK source tree and manual output root")
    value.add_argument("--repository", required=True, type=Path,
                       help="authoritative fetched RITK Git repository and fixture root")
    value.add_argument("--atlas-root", required=True, type=Path,
                       help="Atlas checkout supplying scripts/process_tree.py and OS runtime modules")
    value.add_argument("--base-rev", required=True,
                       help="Git base commit used to describe captured source overrides")
    return value


def main(arguments: list[str] | None = None) -> int:
    try:
        update(parser().parse_args(arguments))
    except (
        capture_source.CaptureSourceError,
        ImageUpdateError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
