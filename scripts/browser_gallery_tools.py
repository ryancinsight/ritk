"""Exercise the RITK browser diagnostic-tool palette on a saved study."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any, Mapping

from browser_canvas import settle_canvas_input
from browser_gallery_artifacts import _write_png
from browser_gallery_trace import AXES
from browser_protocol import ROOT, BrowserRuntimeError, WebDriverClient, _safe_path

TOOL_TIMEOUT_MS = 60_000
TOOL_COUNT_LIMIT = 32
ANNOTATION_KINDS = {"length", "angle", "roi-rect", "roi-ellipse", "hu-point"}
ANNOTATION_TOOL_KINDS = {
    3: "length",
    4: "angle",
    5: "roi-rect",
    6: "roi-ellipse",
    8: "hu-point",
}

from browser_gallery_tool_trace import (
    CLEANUP_TOOL_TRACE_SCRIPT, FOCUS_TOOL_CANVAS_SCRIPT, INSTALL_TOOL_TRACE_SCRIPT,
    INVALID_TOOL_API_PROBE_SCRIPT, READ_TOOL_TRACE_SCRIPT, STOPPED_TOOL_STATE_SCRIPT,
    TOOL_EVENT_TYPES, TOOL_SNAPSHOT_SCRIPT, WAIT_TOOL_FRAME_SCRIPT, WAIT_TOOL_STATE_SCRIPT,
)


def _decimal(value: Any, label: str, *, lower: int = 0) -> int:
    """Parse one bounded unsigned DOM integer."""
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise BrowserRuntimeError(f"{label} is not an unsigned decimal integer: {value!r}")
    parsed = int(value)
    if not lower <= parsed <= (1 << 53) - 1:
        raise BrowserRuntimeError(f"{label} is outside its bound: {parsed}")
    return parsed


def _annotation_state(canvas: Mapping[str, Any], axis: str) -> dict[str, Any]:
    """Validate one input-sensitive completed-annotation projection."""
    count = _decimal(canvas.get("annotation_count"), f"{axis} annotation count", lower=0)
    if count > TOOL_COUNT_LIMIT:
        raise BrowserRuntimeError(f"{axis} annotation count exceeds its bound: {count}")
    kind = canvas.get("last_annotation_kind")
    value = canvas.get("last_annotation_value")
    if not isinstance(kind, str) or not isinstance(value, str):
        raise BrowserRuntimeError(f"{axis} annotation summary is malformed")
    if count == 0:
        if kind or value:
            raise BrowserRuntimeError(f"{axis} empty annotation state carries a summary")
        return {"count": count, "kind": "", "value": None}
    if kind not in ANNOTATION_KINDS or not value:
        raise BrowserRuntimeError(f"{axis} annotation summary has an unknown kind or value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise BrowserRuntimeError(f"{axis} annotation value is not numeric: {value!r}") from error
    if not math.isfinite(numeric):
        raise BrowserRuntimeError(f"{axis} annotation value is not finite: {value!r}")
    return {"count": count, "kind": kind, "value": numeric}


def _tool_replay_order(buttons: list[Mapping[str, Any]]) -> tuple[int, ...]:
    """Replay measurements before tools that can move the viewport."""
    indexes = tuple(range(len(buttons)))
    annotation_indexes = tuple(index for index in indexes if index in ANNOTATION_TOOL_KINDS)
    other_indexes = tuple(index for index in indexes if index not in ANNOTATION_TOOL_KINDS)
    return annotation_indexes + other_indexes


def _snapshot(client: WebDriverClient) -> dict[str, Any]:
    """Read the shared active-tool state and palette controls."""
    result = client.execute(TOOL_SNAPSHOT_SCRIPT, [list(AXES)])
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"tool state could not be sampled: {detail!r}")
    canvases = result.get("canvases")
    controls = result.get("controls")
    if not isinstance(canvases, list) or len(canvases) != len(AXES) or not isinstance(controls, dict):
        raise BrowserRuntimeError("tool state returned an unexpected shape")
    buttons = controls.get("buttons")
    if not isinstance(buttons, list) or not 1 <= len(buttons) <= TOOL_COUNT_LIMIT:
        raise BrowserRuntimeError("tool palette has an invalid button count")
    parsed_buttons = []
    selected = []
    for position, button in enumerate(buttons):
        if not isinstance(button, dict) or button.get("index") != str(position):
            raise BrowserRuntimeError("tool palette indexes are not contiguous")
        if not isinstance(button.get("label"), str) or not button["label"]:
            raise BrowserRuntimeError("tool palette contains an empty label")
        if button.get("pressed") not in ("true", "false") or type(button.get("disabled")) is not bool:
            raise BrowserRuntimeError("tool palette button state is malformed")
        if button["pressed"] == "true":
            selected.append(position)
        parsed_buttons.append({"index": position, "label": button["label"],
                               "pressed": button["pressed"] == "true", "disabled": button["disabled"]})
    if len(selected) != 1:
        raise BrowserRuntimeError("tool palette does not identify exactly one active tool")
    validated = []
    for position, canvas in enumerate(canvases):
        axis = AXES[position]
        if not isinstance(canvas, dict) or canvas.get("axis") != axis:
            raise BrowserRuntimeError(f"tool canvas state for {axis!r} is malformed")
        if canvas.get("load_state") != "ready" or canvas.get("frame_state") != "presented":
            raise BrowserRuntimeError(f"tool canvas {axis!r} is not presenting a study")
        index = _decimal(canvas.get("active_tool_index"), f"{axis} active tool", lower=0)
        if index >= len(parsed_buttons) or not isinstance(canvas.get("active_tool"), str) or not canvas["active_tool"]:
            raise BrowserRuntimeError(f"{axis} active tool is outside the palette")
        if canvas["active_tool"] != parsed_buttons[index]["label"]:
            raise BrowserRuntimeError(f"{axis} active tool label disagrees with the palette")
        generation = _decimal(canvas.get("frame_generation"), f"{axis} frame generation", lower=1)
        annotation = _annotation_state(canvas, axis)
        validated.append({"axis": axis, "active_tool_index": index,
                          "active_tool": canvas["active_tool"], "generation": generation,
                          "annotation": annotation})
    active_indexes = {canvas["active_tool_index"] for canvas in validated}
    active_names = {canvas["active_tool"] for canvas in validated}
    if len(active_indexes) != 1 or len(active_names) != 1 or selected[0] not in active_indexes:
        raise BrowserRuntimeError("active tool semantics diverged across the three canvases")
    annotation_states = [canvas["annotation"] for canvas in validated]
    if any(state != annotation_states[0] for state in annotation_states[1:]):
        raise BrowserRuntimeError("annotation semantics diverged across the three canvases")
    output = controls.get("output")
    if output != f"Active tool: {validated[0]['active_tool']}":
        raise BrowserRuntimeError("active tool output does not match canvas semantics")
    return {"canvases": validated, "controls": {"buttons": parsed_buttons, "output": output}}


def _wait_for_tool(client: WebDriverClient, index: int, label: str) -> None:
    """Wait for all canvases and the palette to publish one tool selection."""
    result = client.execute_async(WAIT_TOOL_STATE_SCRIPT, [index, label, TOOL_TIMEOUT_MS])
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise BrowserRuntimeError(f"browser tool {index} did not settle")


def _wait_for_frame(client: WebDriverClient, previous: Mapping[str, Any]) -> None:
    """Wait for all three canvases to present a new gesture frame."""
    canvases = previous.get("canvases") if isinstance(previous, Mapping) else None
    if not isinstance(canvases, list) or len(canvases) != len(AXES):
        raise BrowserRuntimeError("previous tool snapshot omitted canvas generations")
    generations = [canvas["generation"] for canvas in canvases]
    result = client.execute_async(WAIT_TOOL_FRAME_SCRIPT, [generations, TOOL_TIMEOUT_MS])
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise BrowserRuntimeError("tool gesture did not present a new frame on every canvas")


def _invalid_api_probes(client: WebDriverClient) -> list[dict[str, Any]]:
    """Require malformed tool indexes to reject before viewer mutation."""
    result = client.execute_async(INVALID_TOOL_API_PROBE_SCRIPT)
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"invalid browser tool probes failed: {detail!r}")
    probes = result.get("probes")
    if not isinstance(probes, list) or len(probes) != 7:
        raise BrowserRuntimeError("invalid browser tool probes returned an unexpected count")
    for probe in probes:
        if (
            not isinstance(probe, dict)
            or probe.get("rejected") is not True
            or not isinstance(probe.get("value"), str)
            or not isinstance(probe.get("error"), str)
            or not probe["error"]
        ):
            raise BrowserRuntimeError(f"consumer accepted an invalid browser tool index: {probe!r}")
    return probes


def _install_trace(client: WebDriverClient) -> int:
    """Install bounded trusted observers on the toolbar and axial canvas."""
    result = client.execute(INSTALL_TOOL_TRACE_SCRIPT, [list(TOOL_EVENT_TYPES), 128])
    expected = 8
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("listener_count") != expected:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"tool event trace could not be installed: {detail!r}")
    return expected


def _consume_trace(client: WebDriverClient) -> list[dict[str, Any]]:
    """Read one bounded trusted-event batch."""
    result = client.execute(READ_TOOL_TRACE_SCRIPT)
    if not isinstance(result, dict) or result.get("overflow") is not False or not isinstance(result.get("events"), list):
        raise BrowserRuntimeError("tool event trace is malformed or overflowed")
    events = result["events"]
    if not events:
        raise BrowserRuntimeError("tool action emitted no browser events")
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in TOOL_EVENT_TYPES or event.get("trusted") is not True:
            raise BrowserRuntimeError(f"tool action emitted an untrusted event: {event!r}")
    return events


def _cleanup_trace(client: WebDriverClient, expected: int) -> None:
    """Remove diagnostic listeners exactly once."""
    result = client.execute(CLEANUP_TOOL_TRACE_SCRIPT)
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("listener_count") != expected:
        raise BrowserRuntimeError("tool event listeners were not released exactly once")


def _require_events(events: list[dict[str, Any]], *, target: str, types: set[str], tool_index: int | None = None) -> dict[str, Any]:
    """Validate trusted events for one toolbar, keyboard, or pointer action."""
    relevant = [event for event in events if event.get("target") == target]
    if not relevant or not types.issubset({event["type"] for event in relevant}):
        raise BrowserRuntimeError(f"tool trace omitted {sorted(types)} on {target}")
    if tool_index is not None and not any(event.get("tool_index") == str(tool_index) for event in relevant):
        raise BrowserRuntimeError(f"tool trace omitted palette index {tool_index}")
    return {"event_count": len(relevant), "event_types": dict(sorted(Counter(event["type"] for event in relevant).items())),
            "target": target, "all_trusted": True}


def _click_at(
    client: WebDriverClient,
    canvas: str,
    position: tuple[int, int],
    *,
    source_id: str,
) -> None:
    """Deliver one trusted click at a bounded canvas offset."""
    client.pointer_drag(canvas, position, position, source_id=source_id)
    settle_canvas_input(client)


def _gesture(client: WebDriverClient, canvas: str, index: int) -> str:
    """Perform one input-sensitive gesture for each stable tool index."""
    if index == 0:
        client.pointer_drag(canvas, (64, 64), (128, 96), source_id="ritk-tool-pan-0")
        return "pointer-drag"
    if index in (1, 2):
        client.pointer_drag(canvas, (128, 128), (128, 192), source_id=f"ritk-tool-drag-{index}")
        return "pointer-drag"
    if index == 3:
        _click_at(client, canvas, (64, 64), source_id="ritk-tool-length-0")
        _click_at(client, canvas, (192, 192), source_id="ritk-tool-length-1")
        return "two-click-measurement"
    if index == 4:
        for click_index, position in enumerate(((64, 64), (192, 192), (256, 64))):
            _click_at(client, canvas, position, source_id=f"ritk-tool-angle-{click_index}")
        return "three-click-measurement"
    if index in (5, 6):
        client.pointer_drag(canvas, (64, 64), (192, 192), source_id=f"ritk-tool-roi-{index}")
        return "roi-drag"
    _click_at(client, canvas, (128, 128), source_id=f"ritk-tool-click-{index}")
    return "trusted-click"


def _write_tool_screenshots(client: WebDriverClient, directory: pathlib.Path) -> dict[str, Any]:
    """Store the real three-plane viewport and diagnostic palette."""
    return {
        "window": _write_png(client.screenshot(), directory, "gallery-tools.png", "window"),
        "controls": _write_png(
            client.element_screenshot(client.find(".tool-controls")),
            directory,
            "gallery-tools-controls.png",
            "element",
        ),
    }


def capture_tool_gallery(client: WebDriverClient, output_directory: pathlib.Path) -> dict[str, Any]:
    """Exercise every RITK browser tool with trusted input and save evidence."""
    if not isinstance(output_directory, pathlib.Path):
        raise BrowserRuntimeError("tool output directory must be a pathlib.Path")
    directory = _safe_path(output_directory, directory=ROOT / "output")
    directory.mkdir(parents=True, exist_ok=True)
    initial = _snapshot(client)
    invalid = _invalid_api_probes(client)
    if _snapshot(client) != initial:
        raise BrowserRuntimeError("invalid tool probes mutated the viewer")
    expected_listener_count = _install_trace(client)
    trace_installed = True
    actions_released = False
    try:
        canvas = client.find("#ritk-snap-axial")
        actions: list[dict[str, Any]] = []
        buttons = initial["controls"]["buttons"]
        for index in _tool_replay_order(buttons):
            label = buttons[index]["label"]
            before = _snapshot(client)
            client.click(client.find(f'#tool-buttons button[data-tool-index="{index}"]'))
            _wait_for_tool(client, index, label)
            selected = _snapshot(client)
            selection_events = _require_events(_consume_trace(client), target="toolbar", types={"click"}, tool_index=index)
            if selected["controls"]["buttons"][index]["pressed"] is not True:
                raise BrowserRuntimeError(f"tool palette did not select {label!r}")
            gesture = _gesture(client, canvas, index)
            _wait_for_frame(client, selected)
            after = _snapshot(client)
            pointer_events = _require_events(
                _consume_trace(client), target="canvas", types={"pointerdown", "pointerup"}
            )
            if any(after_canvas["generation"] <= selected_canvas["generation"]
                   for after_canvas, selected_canvas in zip(after["canvases"], selected["canvases"])):
                raise BrowserRuntimeError(f"{label} gesture did not change every presented frame")
            before_annotation = before["canvases"][0]["annotation"]
            after_annotation = after["canvases"][0]["annotation"]
            expected_kind = ANNOTATION_TOOL_KINDS.get(index)
            if expected_kind is None:
                if after_annotation != before_annotation:
                    raise BrowserRuntimeError(f"{label} changed completed annotation state")
            else:
                if (after_annotation["count"] != before_annotation["count"] + 1
                        or after_annotation["kind"] != expected_kind
                        or after_annotation["value"] is None):
                    raise BrowserRuntimeError(
                        f"{label} did not publish {expected_kind} annotation result"
                    )
                if index in (3, 4, 5, 6) and after_annotation["value"] <= 0.0:
                    raise BrowserRuntimeError(f"{label} published a non-positive result")
            actions.append({"index": index, "label": label, "selection": selection_events,
                            "gesture": gesture, "pointer": pointer_events,
                            "generations": [canvas_state["generation"] for canvas_state in after["canvases"]],
                            "annotation_before": before_annotation,
                            "annotation_after": after_annotation})

        if client.execute(FOCUS_TOOL_CANVAS_SCRIPT) is not True:
            raise BrowserRuntimeError("axial canvas did not accept keyboard focus")
        before_keyboard = _snapshot(client)
        client.key_press("p", source_id="ritk-tool-keyboard")
        _wait_for_tool(client, 0, "Pan")
        _wait_for_frame(client, before_keyboard)
        keyboard = _snapshot(client)
        keyboard_events = _consume_trace(client)
        keyboard_evidence = _require_events(keyboard_events, target="canvas", types={"keydown", "keyup"})
        if not any(event.get("key") == "p" and event.get("code") == "KeyP" for event in keyboard_events):
            raise BrowserRuntimeError("tool keyboard trace omitted KeyP metadata")
        client.release_actions()
        actions_released = True
        _cleanup_trace(client, expected_listener_count)
        trace_installed = False
        samples = {"before_stop": client.execute("return window.metisGallery.sample();")}
        screenshots = _write_tool_screenshots(client, directory)
        client.execute("window.metisGallery.stop(); return true;")
        samples["after_stop"] = client.execute("return window.metisGallery.sample();")
        stopped = client.execute(STOPPED_TOOL_STATE_SCRIPT)
        if (
            not isinstance(samples["after_stop"], dict)
            or samples["after_stop"].get("mounted") is not False
            or samples["after_stop"].get("consumer_listeners") != 0
            or not isinstance(stopped, dict)
            or stopped.get("button_count") != len(actions)
            or stopped.get("disabled") is not True
            or stopped.get("output") != "No study"
        ):
            raise BrowserRuntimeError(f"tool viewer did not release its stopped state: {samples!r}, {stopped!r}")
        evidence = {
            "schema": 1,
            "consumer": "ritk-snap",
            "initial": initial,
            "invalid_api_probes": invalid,
            "actions": actions,
            "keyboard": {"active_tool": keyboard["canvases"][0]["active_tool"], "events": keyboard_evidence},
            "screenshots": screenshots,
            "samples": samples,
            "stopped": stopped,
            "cleanup": {
                "active_input_sources_released": True,
                "diagnostic_listener_count": expected_listener_count,
                "diagnostic_listeners_released": True,
                "consumer_canvas_listeners": 0,
            },
        }
        source_root = pathlib.Path(__file__).resolve().parent
        evidence["sources"] = {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in (
                "browser_gallery_tools.py",
                "browser_gallery_tool_trace.py",
                "browser_gallery.py",
            )
        }
        evidence_path = _safe_path(directory / "gallery-tools.json", directory=directory)
        evidence["artifact"] = evidence_path.relative_to(ROOT).as_posix()
        encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if len(encoded.encode("utf-8")) > 512 * 1024:
            raise BrowserRuntimeError("tool evidence exceeds the 512 KiB trace bound")
        evidence_path.write_text(encoded, encoding="utf-8", newline="\n")
        return evidence
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if trace_installed:
            try:
                _cleanup_trace(client, expected_listener_count)
            except BrowserRuntimeError as error:
                cleanup_errors.append(("tool event listener cleanup", error))
        if not actions_released:
            try:
                client.release_actions()
            except BrowserRuntimeError as error:
                cleanup_errors.append(("browser input release", error))
        try:
            client.execute("window.metisGallery.stop(); return true;")
        except BrowserRuntimeError as error:
            cleanup_errors.append(("viewer teardown", error))
        if cleanup_errors:
            if primary_error is not None:
                for operation, error in cleanup_errors:
                    primary_error.add_note(f"{operation} also failed: {error}")
            else:
                operation, error = cleanup_errors[0]
                for later_operation, later_error in cleanup_errors[1:]:
                    error.add_note(f"{later_operation} also failed: {later_error}")
                error.add_note(f"failed operation: {operation}")
                raise error
