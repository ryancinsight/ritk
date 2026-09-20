"""Exercise the RITK saved-study browser gallery and record exact evidence."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import pathlib
import sys
from typing import Any, Mapping, Sequence


def _configure_metis_scripts() -> pathlib.Path | None:
    """Make the selected Metis host scripts importable for this consumer."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--metis-root", type=pathlib.Path)
    known, _ = bootstrap.parse_known_args()
    raw_root = known.metis_root or os.environ.get("METIS_ROOT")
    if raw_root is None:
        return None
    metis_root = raw_root.resolve()
    scripts = metis_root / "scripts"
    if not scripts.is_dir() or scripts.is_symlink():
        raise SystemExit(f"Metis scripts directory is unavailable: {scripts}")
    text = str(scripts)
    if text not in sys.path:
        sys.path.insert(0, text)
    return metis_root


_METIS_ROOT = _configure_metis_scripts()

from browser_canvas import settle_canvas_input
from browser_gallery_actions import _arrow_batch, _keyboard_action
from browser_gallery_artifacts import _write_gallery_screenshots
from browser_gallery_cine import capture_cine_gallery, finalize_cine_teardown
from browser_gallery_tools import capture_tool_gallery
from browser_gallery_window import capture_window_preset_gallery
from browser_gallery_projection import capture_projection_gallery
from browser_gallery_trace import (
    ARROW_BATCH_SIZE,
    AXES,
    _cleanup_event_trace,
    _consume_events,
    _install_event_trace,
    _probe_invalid_slice_api,
    _snapshot,
    _validate_expected_counts,
    _validate_transition,
)
from browser_protocol import ROOT, BrowserRuntimeError, WebDriverClient, _safe_path


def capture_slice_gallery(
    client: WebDriverClient,
    output_directory: pathlib.Path,
    *,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Exercise all consumer gallery sliders and capture trusted evidence.

    The caller must load its study or media and wait for its canvases to report
    presented frames. ``expected_counts`` is the caller's independent
    per-axis shape oracle, keyed by ``axial``, ``coronal`` and ``sagittal``.
    """
    if not isinstance(output_directory, pathlib.Path):
        raise BrowserRuntimeError("gallery output directory must be a pathlib.Path")
    directory = _safe_path(output_directory, directory=ROOT / "output")
    directory.mkdir(parents=True, exist_ok=True)
    counts = _validate_expected_counts(expected_counts)
    initial = _snapshot(client, counts)
    invalid_api_probes = _probe_invalid_slice_api(client, counts, initial)
    initial_indices = {axis: initial[axis]["index"] for axis in AXES}
    actions: list[dict[str, Any]] = []
    expected_listener_count = _install_event_trace(client)
    trace_installed = True
    actions_released = False
    released_listeners = 0
    current = initial
    try:
        for axis in AXES:
            slider_id = f"slice-{axis}"
            slider_element = client.find(f"#{slider_id}")

            before = current
            client.click(slider_element)
            settle_canvas_input(client)
            current = _snapshot(client, counts)
            click_changed = before[axis]["index"] != current[axis]["index"]
            click_required = ["click"] + (["input", "change"] if click_changed else [])
            click_events = _consume_events(client, axis, click_required)
            _validate_transition(before, current, axis, None)
            actions.append(
                {
                    "axis": axis,
                    "action": "native-click",
                    "from_index": before[axis]["index"],
                    "to_index": current[axis]["index"],
                    "generation": current[axis]["generation"],
                    "rgba_sha256": current[axis]["rgba_sha256"],
                    "events": click_events,
                }
            )

            current = _keyboard_action(
                client, counts, actions, axis, "Home", "home", current, 0
            )
            current = _keyboard_action(
                client, counts, actions, axis, "End", "end", current, counts[axis] - 1
            )

            slider_width = current[axis]["slider_width"]
            horizontal_extent = max(1, math.floor(slider_width / 2.0) - 3)
            before = current
            client.pointer_drag(
                slider_element,
                (horizontal_extent, 0),
                (-horizontal_extent, 0),
                source_id=f"metis-gallery-{axis}-pointer",
            )
            settle_canvas_input(client)
            current = _snapshot(client, counts)
            drag_events = _consume_events(
                client,
                axis,
                ("pointerdown", "pointermove", "pointerup", "input", "change"),
            )
            _validate_transition(before, current, axis, 0)
            actions.append(
                {
                    "axis": axis,
                    "action": "pointer-drag-end-to-home",
                    "start": [horizontal_extent, 0],
                    "end": [-horizontal_extent, 0],
                    "from_index": before[axis]["index"],
                    "to_index": current[axis]["index"],
                    "generation": current[axis]["generation"],
                    "rgba_sha256": current[axis]["rgba_sha256"],
                    "events": drag_events,
                }
            )

            current = _keyboard_action(
                client, counts, actions, axis, "Home", "restore-home", current, 0
            )
            remaining = initial_indices[axis]
            while remaining:
                batch = min(remaining, ARROW_BATCH_SIZE)
                current = _arrow_batch(client, counts, actions, axis, current, batch)
                remaining -= batch
            if current[axis]["index"] != initial_indices[axis]:
                raise BrowserRuntimeError(f"{axis} slider did not restore its initial slice")

        restored = _snapshot(client, counts)
        for axis in AXES:
            if (
                restored[axis]["index"] != initial[axis]["index"]
                or restored[axis]["rgba_sha256"] != initial[axis]["rgba_sha256"]
            ):
                raise BrowserRuntimeError(f"{axis} gallery frame did not restore exactly")
            # Different indices can legitimately contain identical RGBA planes,
            # for example empty boundary slices.  Generation proves every index
            # transition rendered; the digest set independently proves the full
            # traversal displayed more than one anatomical image.
            observed_digests = {initial[axis]["rgba_sha256"]}
            observed_digests.update(
                action["rgba_sha256"] for action in actions if action["axis"] == axis
            )
            if len(observed_digests) < 2:
                raise BrowserRuntimeError(f"{axis} slider traversal exposed no changed RGBA frame")
        client.release_actions()
        actions_released = True
        released_listeners = _cleanup_event_trace(client, expected_listener_count)
        trace_installed = False
        screenshots = _write_gallery_screenshots(client, directory)
        evidence = {
            "schema": 1,
            "consumer": "ritk-snap",
            "expected_counts": counts,
            "invalid_api_probes": invalid_api_probes,
            "initial": initial,
            "actions": actions,
            "restored": restored,
            "screenshots": screenshots,
            "cleanup": {
                "active_input_sources_released": True,
                "diagnostic_listener_count": released_listeners,
                "diagnostic_listeners_released": True,
            },
        }
        source_root = pathlib.Path(__file__).resolve().parent
        evidence["sources"] = {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in (
                "browser_gallery.py",
                "browser_gallery_actions.py",
                "browser_gallery_artifacts.py",
                "browser_gallery_projection.py",
                "browser_gallery_trace.py",
            )
        }
        evidence_path = _safe_path(directory / "gallery-slices.json", directory=directory)
        evidence["artifact"] = evidence_path.relative_to(ROOT).as_posix()
        encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if len(encoded.encode("utf-8")) > 512 * 1024:
            raise BrowserRuntimeError("gallery slice evidence exceeds the 512 KiB trace bound")
        evidence_path.write_text(encoded, encoding="utf-8", newline="\n")
        return evidence
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if trace_installed:
            try:
                _cleanup_event_trace(client, expected_listener_count)
            except BrowserRuntimeError as cleanup_error:
                cleanup_errors.append(("gallery slider event listener cleanup", cleanup_error))
        if not actions_released:
            try:
                client.release_actions()
            except BrowserRuntimeError as cleanup_error:
                cleanup_errors.append(("browser input release", cleanup_error))
        if cleanup_errors:
            if primary_error is not None:
                for operation, cleanup_error in cleanup_errors:
                    primary_error.add_note(f"{operation} also failed: {cleanup_error}")
            else:
                operation, cleanup_error = cleanup_errors[0]
                for later_operation, later_error in cleanup_errors[1:]:
                    cleanup_error.add_note(f"{later_operation} also failed: {later_error}")
                cleanup_error.add_note(f"failed operation: {operation}")
                raise cleanup_error


def _capture_consumer_controls(
    client: WebDriverClient,
    output: pathlib.Path,
    oracle: Mapping[str, Any],
    canvas_ids: Sequence[str],
    *,
    window_presets: bool = False,
    cine_controls: bool = False,
    tool_controls: bool = False,
    projection: str | None = None,
) -> Mapping[str, Any]:
    """Run the RITK slice contract after the generic Metis transfer contract."""
    expected_ids = tuple(f"ritk-snap-{axis}" for axis in AXES)
    if projection is not None:
        expected_ids += ("ritk-snap-projection",)
    if tuple(canvas_ids) != expected_ids:
        raise BrowserRuntimeError(
            f"RITK gallery requires canvas IDs {expected_ids!r}; received {tuple(canvas_ids)!r}"
        )
    expected_counts: dict[str, int] = {}
    for axis, canvas_id in zip(AXES, expected_ids):
        view = oracle.get(canvas_id)
        if not isinstance(view, Mapping):
            raise BrowserRuntimeError(f"RITK oracle omitted {canvas_id!r}")
        attributes = view.get("attributes")
        if not isinstance(attributes, Mapping):
            raise BrowserRuntimeError(f"RITK oracle omitted attributes for {canvas_id!r}")
        raw_count = attributes.get("data-ritk-slice-count")
        if not isinstance(raw_count, str) or not raw_count.isdecimal():
            raise BrowserRuntimeError(f"RITK oracle omitted the {axis} slice count")
        expected_counts[axis] = int(raw_count)
    client.set_window_rect(1440, 1200)
    slices = capture_slice_gallery(
        client,
        output / "slices",
        expected_counts=expected_counts,
    )
    if not window_presets and not cine_controls and not tool_controls and projection is None:
        return slices
    result: dict[str, Any] = {"slices": slices}
    if window_presets:
        result["window_level"] = capture_window_preset_gallery(client, output / "window-level")
    if projection is not None:
        result.update(capture_projection_gallery(client, output / "projection", oracle, statistic=projection))
    if cine_controls:
        if tool_controls:
            result["cine"] = capture_cine_gallery(
                client,
                output / "cine",
                stop_viewer=False,
            )
        else:
            result["cine"] = capture_cine_gallery(client, output / "cine")
    if tool_controls:
        result["tools"] = capture_tool_gallery(client, output / "tools")
        if cine_controls:
            result["cine"] = finalize_cine_teardown(client, result["cine"])
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the Metis host parser with the RITK consumer hook."""
    if _METIS_ROOT is None:
        raise SystemExit("--metis-root or METIS_ROOT is required")
    from browser_drop import build_parser as build_host_parser

    parser = build_host_parser()
    parser.add_argument(
        "--metis-root",
        type=pathlib.Path,
        default=_METIS_ROOT,
        help="Metis checkout containing the generic browser host and generated assets",
    )
    parser.add_argument(
        "--window-presets",
        action="store_true",
        help="exercise the RITK modality window/level preset control after slice navigation",
    )
    parser.add_argument(
        "--cine-controls",
        action="store_true",
        help="exercise the RITK Play/Pause and bounded FPS controls after slice navigation",
    )
    parser.add_argument(
        "--tool-controls",
        action="store_true",
        help="exercise every RITK diagnostic interaction tool after slice navigation",
    )
    parser.add_argument(
        "--projection",
        choices=("mip", "minip", "average"),
        help="validate the selected display-only browser scalar projection",
    )
    return parser


def main() -> None:
    """Run the generic host workflow with the RITK slice-control hook."""
    from browser_drop import run as run_host

    args = build_parser().parse_args()
    consumer_capture = None
    if args.lifecycle_cycles == 1:
        consumer_capture = functools.partial(
            _capture_consumer_controls,
            window_presets=args.window_presets,
            cine_controls=args.cine_controls,
            tool_controls=args.tool_controls,
            projection=args.projection,
        )
    run_host(args, consumer_capture=consumer_capture)


if __name__ == "__main__":
    main()
