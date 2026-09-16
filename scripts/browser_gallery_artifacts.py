"""PNG evidence writers for the browser gallery."""

from __future__ import annotations

import hashlib
import pathlib
import struct
from typing import Any

from browser_protocol import ROOT, WebDriverClient, _safe_path


def _write_png(
    content: bytes,
    directory: pathlib.Path,
    filename: str,
    scope: str,
) -> dict[str, Any]:
    """Store one driver-validated PNG with traceable content metadata."""
    path = _safe_path(directory / filename, directory=directory)
    path.write_bytes(content)
    width, height = struct.unpack(">II", content[16:24])
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "width": width,
        "height": height,
        "bytes": len(content),
        "scope": scope,
    }


def _write_gallery_screenshots(client: WebDriverClient, directory: pathlib.Path) -> dict[str, Any]:
    """Store the restored viewport and an unclipped orthogonal-view capture."""
    window = _write_png(client.screenshot(), directory, "gallery-slices.png", "window")
    views_element = client.find(".gallery-views")
    views = _write_png(
        client.element_screenshot(views_element),
        directory,
        "gallery-slice-controls.png",
        "element",
    )
    return {"window": window, "orthogonal_views": views}
