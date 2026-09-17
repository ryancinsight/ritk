"""Exercise the RITK browser window/level preset control."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Mapping

from browser_canvas import settle_canvas_input
from browser_gallery_artifacts import _write_png
from browser_protocol import ROOT, BrowserRuntimeError, WebDriverClient, _safe_path


AXES = ("axial", "coronal", "sagittal")
WINDOW_PRESET_EVENT_TYPES = ("click", "keydown", "keyup", "input", "change")
MAX_PRESET_COUNT = 64
WINDOW_PRESET_TIMEOUT_MS = 60_000

WINDOW_PRESET_SNAPSHOT_SCRIPT = """
const done = arguments[arguments.length - 1];
const axes = arguments[0];
const hex = (bytes) => Array.from(new Uint8Array(bytes),
  (byte) => byte.toString(16).padStart(2, "0")).join("");
const read = async (axis) => {
  const canvas = document.getElementById(`ritk-snap-${axis}`);
  if (!(canvas instanceof HTMLCanvasElement)) return {axis, error: "canvas missing"};
  const pixels = canvas.getContext("2d", {willReadFrequently: true})
    .getImageData(0, 0, canvas.width, canvas.height).data;
  const digest = await crypto.subtle.digest("SHA-256", pixels);
  return {
    axis,
    center: canvas.getAttribute("data-ritk-window-center"),
    width: canvas.getAttribute("data-ritk-window-width"),
    preset_index: canvas.getAttribute("data-ritk-window-preset-index"),
    frame_generation: canvas.getAttribute("data-ritk-frame-generation"),
    frame_state: canvas.getAttribute("data-ritk-frame-state"),
    rgba_sha256: hex(digest),
  };
};
Promise.all(axes.map(read)).then((canvases) => {
  const select = document.getElementById("window-preset");
  const output = document.getElementById("window-level");
  if (!(select instanceof HTMLSelectElement) || !(output instanceof HTMLOutputElement)) {
    done({ok: false, error: "window preset controls are missing"});
    return;
  }
  done({ok: true, canvases, select: {
    disabled: select.disabled,
    value: select.value,
    option_count: select.options.length,
    options: Array.from(select.options, (option) => ({
      value: option.value,
      label: option.textContent || "",
    })),
    output: output.textContent || "",
  }});
}, (error) => done({ok: false, error: String(error && error.message || error)}));
"""

WAIT_FOR_PRESET_SCRIPT = """
const done = arguments[arguments.length - 1];
const expected = String(arguments[0]);
const previous = arguments[1];
const limit = arguments[2];
const ready = () => {
  const canvases = ["axial", "coronal", "sagittal"].map((axis) =>
    document.getElementById(`ritk-snap-${axis}`));
  return canvases.every((canvas, index) => canvas instanceof HTMLCanvasElement &&
    canvas.getAttribute("data-ritk-window-preset-index") === expected &&
    Number(canvas.getAttribute("data-ritk-frame-generation")) > previous[index] &&
    canvas.getAttribute("data-ritk-frame-state") === "presented");
};
if (ready()) { done({ok: true}); return; }
let settled = false;
const observer = new MutationObserver(() => {
  if (settled || !ready()) return;
  settled = true;
  window.clearTimeout(timer);
  observer.disconnect();
  done({ok: true});
});
observer.observe(document, {subtree: true, attributes: true});
const timer = window.setTimeout(() => {
  if (settled) return;
  settled = true;
  observer.disconnect();
  done({ok: false});
}, limit);
"""

INSTALL_PRESET_TRACE_SCRIPT = """
const types = arguments[0];
const maxEvents = arguments[1];
if (window.__ritkWindowPresetTrace) return {ok: false, error: "trace already installed"};
const select = document.getElementById("window-preset");
if (!(select instanceof HTMLSelectElement)) return {ok: false, error: "select missing"};
const events = [];
const registrations = types.map((type) => {
  const listener = (event) => {
    if (events.length >= maxEvents) return;
    events.push({
      type: event.type,
      trusted: event.isTrusted === true,
      key: typeof event.key === "string" ? event.key : null,
      value: select.value,
    });
  };
  select.addEventListener(type, listener, {capture: true, passive: true});
  return {type, listener};
});
window.__ritkWindowPresetTrace = {select, events, registrations, maxEvents};
return {ok: true, listener_count: registrations.length};
"""

READ_PRESET_TRACE_SCRIPT = """
const trace = window.__ritkWindowPresetTrace;
if (!trace) return null;
const overflow = trace.events.length >= trace.maxEvents;
const events = trace.events.splice(0);
return {events, overflow};
"""

CLEANUP_PRESET_TRACE_SCRIPT = """
const trace = window.__ritkWindowPresetTrace;
if (!trace) return {ok: true, listener_count: 0};
for (const registration of trace.registrations) {
  trace.select.removeEventListener(registration.type, registration.listener, true);
}
const listenerCount = trace.registrations.length;
delete window.__ritkWindowPresetTrace;
return {ok: true, listener_count: listenerCount};
"""

FOCUS_PRESET_SCRIPT = """
const select = document.getElementById("window-preset");
if (!(select instanceof HTMLSelectElement)) return false;
select.focus();
return document.activeElement === select;
"""

INVALID_PRESET_API_PROBE_SCRIPT = """
const done = arguments[arguments.length - 1];
const invalid = [
  ["NaN", NaN], ["Infinity", Infinity], ["-Infinity", -Infinity],
  ["-1", -1], ["0.5", 0.5], ["4294967296", 4294967296],
];
import("./consumer/ritk_snap.js").then(({set_web_window_preset}) => {
  const probes = invalid.map(([value, argument]) => {
    let rejected = false;
    let error = null;
    try { set_web_window_preset(argument); }
    catch (failure) {
      rejected = true;
      error = String(failure && failure.message || failure);
    }
    return {value, rejected, error};
  });
  done({ok: true, probes});
}, (error) => done({ok: false, error: String(error && error.message || error)}));
"""


def _decimal(value: Any, label: str, *, lower: int = 0) -> int:
    """Parse one bounded unsigned DOM integer."""
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise BrowserRuntimeError(f"{label} is not an unsigned decimal integer: {value!r}")
    parsed = int(value)
    if parsed < lower or parsed > (1 << 53) - 1:
        raise BrowserRuntimeError(f"{label} is outside its bound: {parsed}")
    return parsed


def _snapshot(client: WebDriverClient) -> dict[str, Any]:
    """Read every window/level semantic and exact canvas digest."""
    result = client.execute_async(WINDOW_PRESET_SNAPSHOT_SCRIPT, [list(AXES)])
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"window preset state could not be sampled: {detail!r}")
    canvases = result.get("canvases")
    select = result.get("select")
    if not isinstance(canvases, list) or len(canvases) != len(AXES) or not isinstance(select, dict):
        raise BrowserRuntimeError("window preset state returned an unexpected shape")
    if (
        type(select.get("disabled")) is not bool
        or not isinstance(select.get("value"), str)
        or type(select.get("option_count")) is not int
        or not 1 <= select["option_count"] <= MAX_PRESET_COUNT
        or not isinstance(select.get("options"), list)
        or len(select["options"]) != select["option_count"]
        or not isinstance(select.get("output"), str)
    ):
        raise BrowserRuntimeError("window preset select state is malformed")
    options = []
    for position, option in enumerate(select["options"]):
        if (
            not isinstance(option, dict)
            or option.get("value") != str(position)
            or not isinstance(option.get("label"), str)
            or not option["label"]
        ):
            raise BrowserRuntimeError("window preset options are not a contiguous label table")
        options.append({"value": option["value"], "label": option["label"]})
    if select["value"] not in {"", *(option["value"] for option in options)}:
        raise BrowserRuntimeError("window preset select value is outside its option table")
    validated: dict[str, dict[str, Any]] = {}
    for position, canvas in enumerate(canvases):
        axis = AXES[position]
        if not isinstance(canvas, dict) or canvas.get("axis") != axis:
            raise BrowserRuntimeError(f"window preset canvas state for {axis!r} is malformed")
        center = canvas.get("center")
        width = canvas.get("width")
        if not isinstance(center, str) or not isinstance(width, str):
            raise BrowserRuntimeError(f"{axis} did not publish window/level values")
        try:
            parsed_center = float(center)
            parsed_width = float(width)
        except ValueError as error:
            raise BrowserRuntimeError(f"{axis} window/level values are not numeric") from error
        if not math.isfinite(parsed_center) or not math.isfinite(parsed_width) or parsed_width <= 0:
            raise BrowserRuntimeError(f"{axis} published invalid window/level values")
        preset_index = canvas.get("preset_index")
        if preset_index not in ("", *[option["value"] for option in options]):
            raise BrowserRuntimeError(f"{axis} preset index is outside its option table")
        generation = _decimal(canvas.get("frame_generation"), f"{axis} frame generation", lower=1)
        digest = canvas.get("rgba_sha256")
        if (
            canvas.get("frame_state") != "presented"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BrowserRuntimeError(f"{axis} did not expose a presented RGBA frame")
        validated[axis] = {
            "center": center,
            "width": width,
            "preset_index": preset_index,
            "generation": generation,
            "rgba_sha256": digest,
        }
    centers = {state["center"] for state in validated.values()}
    widths = {state["width"] for state in validated.values()}
    if len(centers) != 1 or len(widths) != 1:
        raise BrowserRuntimeError("window/level semantics diverged across the three canvases")
    return {"canvases": validated, "select": {"disabled": select["disabled"],
        "value": select["value"], "options": options, "output": select["output"]}}


def _wait_for_preset(client: WebDriverClient, target: int, previous: Mapping[str, Mapping[str, Any]]) -> None:
    """Wait for all three canvases to present the selected preset."""
    generations = [previous[axis]["generation"] for axis in AXES]
    result = client.execute_async(WAIT_FOR_PRESET_SCRIPT, [target, generations, WINDOW_PRESET_TIMEOUT_MS])
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise BrowserRuntimeError(f"window preset {target} did not present on every canvas")


def _install_trace(client: WebDriverClient) -> int:
    """Install one bounded trusted-event observer on the select control."""
    result = client.execute(INSTALL_PRESET_TRACE_SCRIPT, [list(WINDOW_PRESET_EVENT_TYPES), 128])
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("listener_count") != len(WINDOW_PRESET_EVENT_TYPES):
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"window preset event trace could not be installed: {detail!r}")
    return len(WINDOW_PRESET_EVENT_TYPES)


def _consume_trace(client: WebDriverClient) -> dict[str, Any]:
    """Validate that selection used trusted browser input delivery."""
    result = client.execute(READ_PRESET_TRACE_SCRIPT)
    if not isinstance(result, dict) or result.get("overflow") is not False or not isinstance(result.get("events"), list):
        raise BrowserRuntimeError("window preset event trace is malformed or overflowed")
    events = result["events"]
    if not events:
        raise BrowserRuntimeError("window preset selection emitted no browser events")
    counts: dict[str, int] = {}
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("type") not in WINDOW_PRESET_EVENT_TYPES
            or event.get("trusted") is not True
            or not isinstance(event.get("value"), str)
        ):
            raise BrowserRuntimeError("window preset selection emitted an untrusted event")
        event_type = event["type"]
        counts[event_type] = counts.get(event_type, 0) + 1
    missing = set(("keydown", "keyup", "input", "change")) - set(counts)
    if missing:
        raise BrowserRuntimeError(f"window preset selection omitted {sorted(missing)!r}")
    return {"event_count": len(events), "event_types": dict(sorted(counts.items())),
        "all_trusted": True, "target_id": "window-preset"}


def _cleanup_trace(client: WebDriverClient, expected_count: int) -> None:
    """Remove the select observer exactly once."""
    result = client.execute(CLEANUP_PRESET_TRACE_SCRIPT)
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("listener_count") != expected_count:
        raise BrowserRuntimeError("window preset event listeners were not released exactly once")


def _focus_preset(client: WebDriverClient) -> None:
    """Refocus the DOM select after closing Chromium's native popup."""
    if client.execute(FOCUS_PRESET_SCRIPT) is not True:
        raise BrowserRuntimeError("window preset select could not be focused")


def _invalid_probes(client: WebDriverClient) -> list[dict[str, Any]]:
    """Require malformed preset indices to reject before touching the viewer."""
    result = client.execute_async(INVALID_PRESET_API_PROBE_SCRIPT)
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"window preset rejection probes failed: {detail!r}")
    probes = result.get("probes")
    expected = {"NaN", "Infinity", "-Infinity", "-1", "0.5", "4294967296"}
    if not isinstance(probes, list) or len(probes) != len(expected):
        raise BrowserRuntimeError("window preset rejection probes returned an unexpected count")
    for probe in probes:
        if (
            not isinstance(probe, dict)
            or probe.get("value") not in expected
            or probe.get("rejected") is not True
            or not isinstance(probe.get("error"), str)
            or not probe["error"]
        ):
            raise BrowserRuntimeError(f"window preset API accepted an invalid index: {probe!r}")
    return probes


def _write_window_screenshots(client: WebDriverClient, directory: pathlib.Path) -> dict[str, Any]:
    """Store the real three-plane viewport and visible display controls."""
    window = _write_png(client.screenshot(), directory, "gallery-window-level.png", "window")
    controls = _write_png(
        client.element_screenshot(client.find(".presentation-controls")),
        directory,
        "gallery-window-level-controls.png",
        "element",
    )
    return {"window": window, "controls": controls}


def capture_window_preset_gallery(
    client: WebDriverClient,
    output_directory: pathlib.Path,
) -> dict[str, Any]:
    """Apply one real modality preset and capture semantic and pixel evidence."""
    if not isinstance(output_directory, pathlib.Path):
        raise BrowserRuntimeError("window preset output directory must be a pathlib.Path")
    directory = _safe_path(output_directory, directory=ROOT / "output")
    directory.mkdir(parents=True, exist_ok=True)
    initial = _snapshot(client)
    select = initial["select"]
    if select["disabled"] or len(select["options"]) < 2:
        raise BrowserRuntimeError("window preset control was not enabled after study load")
    if not select["output"].startswith("Center ") or " · Width " not in select["output"]:
        raise BrowserRuntimeError("window preset control did not publish its effective values")
    invalid_probes = _invalid_probes(client)
    after_invalid = _snapshot(client)
    if after_invalid != initial:
        raise BrowserRuntimeError("invalid window preset probes mutated the viewer")
    target = 0 if select["value"] != "0" else 1
    if target >= len(select["options"]):
        raise BrowserRuntimeError("window preset table has no distinct target for the replay")
    expected_listener_count = _install_trace(client)
    trace_installed = True
    actions_released = False
    try:
        select_element = client.find("#window-preset")
        client.click(select_element)
        client.key_press("Escape", source_id="ritk-window-preset-keyboard")
        _focus_preset(client)
        client.key_press("Home", source_id="ritk-window-preset-keyboard")
        for _ in range(target):
            client.key_press("ArrowDown", source_id="ritk-window-preset-keyboard")
        client.key_press("Enter", source_id="ritk-window-preset-keyboard")
        settle_canvas_input(client)
        _wait_for_preset(client, target, initial["canvases"])
        selected = _snapshot(client)
        if selected["select"]["value"] != str(target):
            raise BrowserRuntimeError("window preset select did not retain the requested option")
        if not selected["select"]["output"].startswith("Center ") or " · Width " not in selected["select"]["output"]:
            raise BrowserRuntimeError("window preset control lost its effective values")
        for axis in AXES:
            before = initial["canvases"][axis]
            after = selected["canvases"][axis]
            if after["generation"] <= before["generation"]:
                raise BrowserRuntimeError(f"{axis} window preset did not advance frame generation")
            if after["preset_index"] != str(target):
                raise BrowserRuntimeError(f"{axis} did not publish the selected window preset")
            if after["rgba_sha256"] == before["rgba_sha256"]:
                raise BrowserRuntimeError(f"{axis} window preset did not change rendered pixels")
        events = _consume_trace(client)
        client.release_actions()
        actions_released = True
        _cleanup_trace(client, expected_listener_count)
        trace_installed = False
        screenshots = _write_window_screenshots(client, directory)
        evidence = {
            "schema": 1,
            "consumer": "ritk-snap",
            "initial": initial,
            "invalid_api_probes": invalid_probes,
            "selection": {"index": target, "label": select["options"][target]["label"],
                "events": events, "after": selected},
            "screenshots": screenshots,
            "cleanup": {"active_input_sources_released": True,
                "diagnostic_listener_count": expected_listener_count,
                "diagnostic_listeners_released": True},
        }
        source = pathlib.Path(__file__).resolve()
        evidence["sources"] = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
        evidence_path = _safe_path(directory / "gallery-window-level.json", directory=directory)
        evidence["artifact"] = evidence_path.relative_to(ROOT).as_posix()
        encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if len(encoded.encode("utf-8")) > 512 * 1024:
            raise BrowserRuntimeError("window preset evidence exceeds the 512 KiB trace bound")
        evidence_path.write_text(encoded, encoding="utf-8", newline="\n")
        return evidence
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if trace_installed:
            try:
                _cleanup_trace(client, expected_listener_count)
            except BrowserRuntimeError as error:
                cleanup_errors.append(("window preset event listener cleanup", error))
        if not actions_released:
            try:
                client.release_actions()
            except BrowserRuntimeError as error:
                cleanup_errors.append(("browser input release", error))
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
