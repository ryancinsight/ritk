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
    responsive = client.execute(
        "return document.getElementById('responsive-views')?.hidden === false;"
    )
    views_element = client.find(".responsive-views" if responsive else ".gallery-views")
    views = _write_png(
        client.element_screenshot(views_element),
        directory,
        "gallery-slice-controls.png",
        "element",
    )
    return {"window": window, "orthogonal_views": views}


def set_menu_open(client: WebDriverClient, menu: str, opened: bool) -> None:
    visible = client.execute("return document.getElementById(arguments[0]).matches(':popover-open');", [menu])
    if visible != opened:
        client.click(client.find(f'[popovertarget="{menu}"]'))


def capture_menu_controls(client: WebDriverClient, directory: pathlib.Path, menu: str,
                          controls: str, filename: str) -> dict[str, Any]:
    """Save the reading workspace and its controls, then restore the workspace."""
    set_menu_open(client, menu, False)
    window = _write_png(client.screenshot(), directory, f"{filename}.png", "window")
    set_menu_open(client, menu, True)
    try:
        png = client.element_screenshot(client.find(controls))
        detail = _write_png(png, directory, f"{filename}-controls.png", "element")
    finally:
        set_menu_open(client, menu, False)
    return {"window": window, "controls": detail}
