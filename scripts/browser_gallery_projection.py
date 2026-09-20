"""Validate the consumer-owned display-only scalar projection canvas."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping

from browser_protocol import ROOT, BrowserRuntimeError, WebDriverClient, _safe_path


PROJECTION_PIXELS = """
const done = arguments[arguments.length - 1];
const canvas = document.getElementById(arguments[0]);
if (!(canvas instanceof HTMLCanvasElement)) {
  done({diagnostic: 'projection canvas is missing'});
} else if (canvas.width > 4096 || canvas.height > 4096) {
  done({diagnostic: 'projection canvas exceeds the capture bound'});
} else {
  const context = canvas.getContext('2d');
  if (context === null) {
    done({diagnostic: 'projection canvas has no 2d context'});
  } else {
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let nonBlack = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if (pixels[index] || pixels[index + 1] || pixels[index + 2]) ++nonBlack;
    }
    crypto.subtle.digest('SHA-256', pixels).then((hash) => done({
      width: canvas.width,
      height: canvas.height,
      non_black_pixels: nonBlack,
      rgba_sha256: Array.from(new Uint8Array(hash), (byte) =>
        byte.toString(16).padStart(2, '0')).join(''),
    }), (error) => done({diagnostic: String(error)}));
  }
}
"""


PROJECTION_ATTRIBUTES = """
const id = arguments[0];
const names = arguments[1];
const canvas = document.getElementById(id);
if (!(canvas instanceof HTMLCanvasElement)) return {diagnostic: 'projection canvas is missing'};
const attributes = {};
for (const name of names) attributes[name] = canvas.getAttribute(name);
return {attributes};
"""


def capture_projection_gallery(
    client: WebDriverClient,
    output_directory: pathlib.Path,
    oracle: Mapping[str, Any],
    *,
    statistic: str,
) -> Mapping[str, Any]:
    """Capture real scalar projection pixels and prove the surface is display-only.

    The generic host owns exact dimensions and screenshot capture. This consumer
    hook adds a value-semantic pixel read from the loaded study, validates the
    RITK projection metadata, and checks that only the three orthogonal canvases
    retain input listener guards.
    """
    if statistic not in {"mip", "minip", "average"}:
        raise BrowserRuntimeError(f"unsupported projection statistic: {statistic!r}")
    if not isinstance(output_directory, pathlib.Path):
        raise BrowserRuntimeError("projection output directory must be a pathlib.Path")
    directory = _safe_path(output_directory, directory=ROOT / "output")
    directory.mkdir(parents=True, exist_ok=True)
    canvas_id = "ritk-snap-projection"
    expected = oracle.get(canvas_id)
    if not isinstance(expected, Mapping):
        raise BrowserRuntimeError(f"projection oracle omitted {canvas_id!r}")
    attributes = expected.get("attributes")
    if not isinstance(attributes, Mapping):
        raise BrowserRuntimeError("projection oracle omitted its attributes")
    expected_label = {"mip": "MIP", "minip": "MinIP", "average": "Average"}[statistic]
    required_attributes = {
        "data-ritk-role": "projection",
        "data-ritk-load-state": "ready",
        "data-ritk-frame-state": "presented",
        "data-ritk-projection-statistic": expected_label,
    }
    for name, value in required_attributes.items():
        if attributes.get(name) != value:
            raise BrowserRuntimeError(
                f"projection oracle attribute {name!r} must be {value!r}"
            )
    for key in ("width", "height"):
        if type(expected.get(key)) is not int or not 1 <= expected[key] <= 4096:
            raise BrowserRuntimeError(f"projection oracle has invalid {key}")
    actual = client.execute_async(PROJECTION_PIXELS, [canvas_id])
    if not isinstance(actual, Mapping) or "diagnostic" in actual:
        detail = actual.get("diagnostic") if isinstance(actual, Mapping) else actual
        raise BrowserRuntimeError(f"projection pixels could not be read: {detail!r}")
    wanted_dimensions = {key: expected[key] for key in ("width", "height")}
    if {key: actual.get(key) for key in wanted_dimensions} != wanted_dimensions:
        raise BrowserRuntimeError(
            f"projection dimensions differ: expected {wanted_dimensions}, found {actual}"
        )
    non_black = actual.get("non_black_pixels")
    if type(non_black) is not int or non_black <= 0:
        raise BrowserRuntimeError("projection canvas contains no visible study pixels")
    observed = client.execute(PROJECTION_ATTRIBUTES, [canvas_id, list(attributes)])
    if not isinstance(observed, Mapping) or observed.get("attributes") != dict(attributes):
        raise BrowserRuntimeError(
            f"projection metadata differs: expected {dict(attributes)}, found {observed}"
        )
    sample = client.execute("return window.metisGallery.sample();")
    if not isinstance(sample, Mapping):
        raise BrowserRuntimeError("gallery sample is not an object")
    if sample.get("projection_mode") != statistic:
        raise BrowserRuntimeError(
            f"gallery selected {sample.get('projection_mode')!r}, expected {statistic!r}"
        )
    if sample.get("projection_statistic") != expected_label:
        raise BrowserRuntimeError("gallery sample omitted the selected projection label")
    # Each input-enabled Metis surface retains seven provider guards. A
    # four-canvas projection workflow must therefore report exactly 3 * 7.
    if sample.get("consumer_listeners") != 21:
        raise BrowserRuntimeError(
            f"projection surface changed the interactive listener budget: {sample.get('consumer_listeners')!r}"
        )
    evidence = {
        "schema": 1,
        "canvas_id": canvas_id,
        "statistic": statistic,
        "label": expected_label,
        "dimensions": wanted_dimensions,
        "non_black_pixels": non_black,
        "rgba_sha256": actual.get("rgba_sha256"),
        "attributes": dict(attributes),
        "consumer_listeners": sample["consumer_listeners"],
        "display_only": True,
    }
    if not isinstance(evidence["rgba_sha256"], str) or len(evidence["rgba_sha256"]) != 64:
        raise BrowserRuntimeError("projection pixel evidence omitted its SHA-256 digest")
    path = _safe_path(directory / "projection.json", directory=directory)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"projection": evidence, "artifact": path.relative_to(ROOT).as_posix()}
