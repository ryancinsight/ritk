"""Exercise the RITK browser cine controls on a loaded saved study."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Mapping

from browser_canvas import settle_canvas_input
from browser_gallery_artifacts import _write_png
from browser_gallery_trace import AXES
from browser_protocol import ROOT, BrowserRuntimeError, WebDriverClient, _safe_path

CINE_EVENT_TYPES = ("click", "keydown", "keyup", "input", "change")
CINE_RATE_TARGET = 24
CINE_TIMEOUT_MS = 60_000
MAX_CINE_RATE = 60

CINE_SNAPSHOT_SCRIPT = """
const axes = arguments[0];
const readCanvas = (axis) => {
  const canvas = document.getElementById(`ritk-snap-${axis}`);
  if (!(canvas instanceof HTMLCanvasElement)) return {axis, error: "canvas missing"};
  return {
    axis,
    load_state: canvas.getAttribute("data-ritk-load-state"),
    frame_state: canvas.getAttribute("data-ritk-frame-state"),
    cine_enabled: canvas.getAttribute("data-ritk-cine-enabled"),
    cine_fps: canvas.getAttribute("data-ritk-cine-fps"),
    slice_index: canvas.getAttribute("data-ritk-slice-index"),
    frame_generation: canvas.getAttribute("data-ritk-frame-generation"),
  };
};
const button = document.getElementById("cine-toggle");
const rate = document.getElementById("cine-rate");
const output = document.getElementById("cine-rate-value");
if (!(button instanceof HTMLButtonElement) ||
    !(rate instanceof HTMLInputElement) ||
    !(output instanceof HTMLOutputElement)) {
  return {ok: false, error: "cine controls are missing"};
}
return {ok: true, canvases: axes.map(readCanvas), controls: {
  button_disabled: button.disabled,
  button_text: button.textContent || "",
  button_pressed: button.getAttribute("aria-pressed"),
  rate_disabled: rate.disabled,
  rate_min: rate.min,
  rate_max: rate.max,
  rate_step: rate.step,
  rate_value: rate.value,
  output: output.textContent || "",
}};
"""

WAIT_CINE_STATE_SCRIPT = """
const done = arguments[arguments.length - 1];
const enabled = arguments[0];
const rate = arguments[1];
const limit = arguments[2];
const read = () => {
  const canvas = document.getElementById("ritk-snap-axial");
  const button = document.getElementById("cine-toggle");
  const input = document.getElementById("cine-rate");
  return {
    canvas: canvas instanceof HTMLCanvasElement ? {
      enabled: canvas.getAttribute("data-ritk-cine-enabled"),
      index: canvas.getAttribute("data-ritk-slice-index"),
      generation: canvas.getAttribute("data-ritk-frame-generation"),
    } : null,
    controls: {
      button_pressed: button instanceof HTMLButtonElement ? button.getAttribute("aria-pressed") : null,
      rate: input instanceof HTMLInputElement ? input.value : null,
    },
  };
};
const ready = () => {
  const canvas = document.getElementById("ritk-snap-axial");
  const button = document.getElementById("cine-toggle");
  const input = document.getElementById("cine-rate");
  return canvas instanceof HTMLCanvasElement &&
    button instanceof HTMLButtonElement && input instanceof HTMLInputElement &&
    canvas.getAttribute("data-ritk-cine-enabled") === String(enabled) &&
    (rate === null || canvas.getAttribute("data-ritk-cine-fps") === String(rate)) &&
    button.getAttribute("aria-pressed") === String(enabled) &&
    input.disabled === false;
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
observer.observe(document, {subtree: true, childList: true, attributes: true,
  characterData: true});
const timer = window.setTimeout(() => {
  if (settled) return;
  settled = true;
  observer.disconnect();
  done({ok: false, current: read(), status: document.getElementById("gallery-status")?.textContent || ""});
}, limit);
"""

WAIT_CINE_FRAME_SCRIPT = """
const done = arguments[arguments.length - 1];
const previous = arguments[0];
const limit = arguments[1];
const read = () => ["axial", "coronal", "sagittal"].map((axis) => {
  const canvas = document.getElementById(`ritk-snap-${axis}`);
  return canvas instanceof HTMLCanvasElement ? {
    axis,
    enabled: canvas.getAttribute("data-ritk-cine-enabled"),
    index: canvas.getAttribute("data-ritk-slice-index"),
    generation: Number(canvas.getAttribute("data-ritk-frame-generation")),
  } : null;
});
const status = () => document.getElementById("gallery-status")?.textContent || "";
const ready = () => {
  const current = read();
  return current.every((state, position) => state && state.enabled === "true" &&
    Number.isSafeInteger(state.generation) &&
    state.generation >= previous[position].generation) &&
    current.some((state, position) => state.generation > previous[position].generation &&
      state.index !== previous[position].index);
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
  done({ok: false, current: read(), status: status()});
}, limit);
"""

INVALID_CINE_API_PROBE_SCRIPT = """
const done = arguments[arguments.length - 1];
const invalid = [
  ["NaN", NaN], ["Infinity", Infinity], ["-Infinity", -Infinity],
  ["0", 0], ["0.5", 0.5], ["61", 61], ["4294967296", 4294967296],
];
import("./consumer/ritk_snap.js").then(({set_web_cine_rate}) => {
  const probes = invalid.map(([value, argument]) => {
    let rejected = false;
    let error = null;
    try { set_web_cine_rate(argument); }
    catch (failure) {
      rejected = true;
      error = String(failure && failure.message || failure);
    }
    return {value, rejected, error};
  });
  done({ok: true, probes});
}, (error) => done({ok: false, error: String(error && error.message || error)}));
"""

INSTALL_CINE_TRACE_SCRIPT = """
const ids = ["cine-toggle", "cine-rate"];
const types = arguments[0];
const maxEvents = arguments[1];
if (window.__ritkCineTrace) return {ok: false, error: "trace already installed"};
const events = [];
const registrations = [];
for (const id of ids) {
  const element = document.getElementById(id);
  if (!element) return {ok: false, error: `control ${id} is missing`};
  for (const type of types) {
    const listener = (event) => {
      if (events.length >= maxEvents) return;
      events.push({type: event.type, target_id: event.target && event.target.id,
        trusted: event.isTrusted === true, key: event.key || null,
        value: typeof element.value === "string" ? element.value : null});
    };
    element.addEventListener(type, listener, {capture: true, passive: true});
    registrations.push({element, type, listener});
  }
}
window.__ritkCineTrace = {events, registrations, maxEvents};
return {ok: true, listener_count: registrations.length};
"""

READ_CINE_TRACE_SCRIPT = """
const trace = window.__ritkCineTrace;
if (!trace) return null;
const overflow = trace.events.length >= trace.maxEvents;
const events = trace.events.splice(0);
return {events, overflow};
"""

CLEANUP_CINE_TRACE_SCRIPT = """
const trace = window.__ritkCineTrace;
if (!trace) return {ok: true, listener_count: 0};
for (const registration of trace.registrations) {
  registration.element.removeEventListener(registration.type, registration.listener, true);
}
const listenerCount = trace.registrations.length;
delete window.__ritkCineTrace;
return {ok: true, listener_count: listenerCount};
"""

STOPPED_STATE_SCRIPT = """
const button = document.getElementById("cine-toggle");
const rate = document.getElementById("cine-rate");
const output = document.getElementById("cine-rate-value");
if (!(button instanceof HTMLButtonElement) ||
    !(rate instanceof HTMLInputElement) ||
    !(output instanceof HTMLOutputElement)) return null;
return {
  button_disabled: button.disabled,
  button_pressed: button.getAttribute("aria-pressed"),
  button_text: button.textContent || "",
  rate_disabled: rate.disabled,
  rate_value: rate.value,
  output: output.textContent || "",
};
"""

def _decimal(value: Any, label: str, *, lower: int = 0) -> int:
    """Parse one bounded unsigned DOM integer."""
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise BrowserRuntimeError(f"{label} is not an unsigned decimal integer: {value!r}")
    parsed = int(value)
    if not lower <= parsed <= (1 << 53) - 1:
        raise BrowserRuntimeError(f"{label} is outside its bound: {parsed}")
    return parsed

def _snapshot(client: WebDriverClient) -> dict[str, Any]:
    """Read cine semantics and control state from the live gallery."""
    result = client.execute(CINE_SNAPSHOT_SCRIPT, [list(AXES)])
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"cine state could not be sampled: {detail!r}")
    canvases = result.get("canvases")
    controls = result.get("controls")
    if not isinstance(canvases, list) or len(canvases) != len(AXES) or not isinstance(controls, dict):
        raise BrowserRuntimeError("cine state returned an unexpected shape")
    if (
        controls.get("rate_min") != "1"
        or controls.get("rate_max") != str(MAX_CINE_RATE)
        or controls.get("rate_step") != "1"
        or not isinstance(controls.get("rate_value"), str)
        or not isinstance(controls.get("output"), str)
        or type(controls.get("button_disabled")) is not bool
        or type(controls.get("rate_disabled")) is not bool
    ):
        raise BrowserRuntimeError("cine controls have invalid bounds or state")
    validated = []
    for position, canvas in enumerate(canvases):
        axis = AXES[position]
        if not isinstance(canvas, dict) or canvas.get("axis") != axis:
            raise BrowserRuntimeError(f"cine canvas state for {axis!r} is malformed")
        if canvas.get("load_state") != "ready" or canvas.get("frame_state") != "presented":
            raise BrowserRuntimeError(f"cine canvas {axis!r} is not presenting a study")
        enabled = canvas.get("cine_enabled")
        if enabled not in ("true", "false"):
            raise BrowserRuntimeError(f"cine canvas {axis!r} omitted enabled semantics")
        rate = _decimal(canvas.get("cine_fps"), f"{axis} cine rate", lower=1)
        if rate > MAX_CINE_RATE:
            raise BrowserRuntimeError(f"{axis} cine rate exceeds {MAX_CINE_RATE} FPS")
        index = _decimal(canvas.get("slice_index"), f"{axis} slice index")
        generation = _decimal(canvas.get("frame_generation"), f"{axis} frame generation", lower=1)
        validated.append({"axis": axis, "cine_enabled": enabled == "true", "cine_fps": rate,
                          "slice_index": index, "frame_generation": generation})
    rates = {canvas["cine_fps"] for canvas in validated}
    enabled_states = {canvas["cine_enabled"] for canvas in validated}
    if len(rates) != 1 or len(enabled_states) != 1:
        raise BrowserRuntimeError("cine semantics diverged across the three canvases")
    rate = validated[0]["cine_fps"]
    enabled = validated[0]["cine_enabled"]
    if controls["rate_value"] != str(rate) or controls["output"] != f"{rate} FPS":
        raise BrowserRuntimeError("cine rate control does not match canvas semantics")
    if controls["button_pressed"] != str(enabled).lower():
        raise BrowserRuntimeError("cine button state does not match canvas semantics")
    if controls["button_text"] != ("Pause" if enabled else "Play"):
        raise BrowserRuntimeError("cine button label does not match playback state")
    return {"canvases": validated, "controls": controls}


def _wait_for_state(client: WebDriverClient, enabled: bool, rate: int | None = None) -> None:
    """Wait for Rust-published cine state through a DOM mutation observer."""
    result = client.execute_async(WAIT_CINE_STATE_SCRIPT, [enabled, rate, CINE_TIMEOUT_MS])
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise BrowserRuntimeError(f"cine state did not settle at enabled={enabled}, rate={rate}")


def _wait_for_frame(client: WebDriverClient, previous: Mapping[str, Any]) -> None:
    """Wait for one active-axis frame transition after playback starts."""
    states = previous.get("canvases") if isinstance(previous, Mapping) else None
    if not isinstance(states, list) or len(states) != len(AXES):
        raise BrowserRuntimeError("previous cine snapshot omitted canvas states")
    result = client.execute_async(WAIT_CINE_FRAME_SCRIPT, [states, CINE_TIMEOUT_MS])
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result if isinstance(result, dict) else {"result": result}
        raise BrowserRuntimeError(f"cine playback did not advance a presented slice: {json.dumps(detail, sort_keys=True)}")


def _invalid_api_probes(client: WebDriverClient) -> list[dict[str, Any]]:
    """Require malformed rates to reject at the typed WASM boundary."""
    result = client.execute_async(INVALID_CINE_API_PROBE_SCRIPT)
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"invalid cine rate probes failed: {detail!r}")
    probes = result.get("probes")
    if not isinstance(probes, list) or len(probes) != 7:
        raise BrowserRuntimeError("invalid cine rate probes returned an unexpected count")
    for probe in probes:
        if (
            not isinstance(probe, dict)
            or probe.get("rejected") is not True
            or not isinstance(probe.get("value"), str)
            or not isinstance(probe.get("error"), str)
            or not probe["error"]
        ):
            raise BrowserRuntimeError(f"consumer accepted an invalid cine rate: {probe!r}")
    return probes


def _install_trace(client: WebDriverClient) -> int:
    """Install bounded trusted-event observers on the two cine controls."""
    result = client.execute(INSTALL_CINE_TRACE_SCRIPT, [list(CINE_EVENT_TYPES), 128])
    expected = 2 * len(CINE_EVENT_TYPES)
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("listener_count") != expected:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"cine event trace could not be installed: {detail!r}")
    return expected


def _consume_trace(client: WebDriverClient) -> dict[str, Any]:
    """Validate trusted button and range delivery."""
    result = client.execute(READ_CINE_TRACE_SCRIPT)
    if not isinstance(result, dict) or result.get("overflow") is not False:
        raise BrowserRuntimeError("cine event trace is malformed or overflowed")
    events = result.get("events")
    if not isinstance(events, list) or not events:
        raise BrowserRuntimeError("cine controls emitted no browser events")
    counts: Counter[str] = Counter()
    targets = set()
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("type") not in CINE_EVENT_TYPES
            or event.get("trusted") is not True
            or event.get("target_id") not in {"cine-toggle", "cine-rate"}
        ):
            raise BrowserRuntimeError(f"cine control event was not trusted: {event!r}")
        counts[event["type"]] += 1
        targets.add(event["target_id"])
    if "cine-toggle" not in targets or "cine-rate" not in targets:
        raise BrowserRuntimeError("cine trace omitted one of the controls")
    return {"event_count": len(events), "event_types": dict(sorted(counts.items())),
            "targets": sorted(targets), "all_trusted": True}


def _cleanup_trace(client: WebDriverClient, expected: int) -> None:
    """Remove the diagnostic listeners exactly once."""
    result = client.execute(CLEANUP_CINE_TRACE_SCRIPT)
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("listener_count") != expected:
        raise BrowserRuntimeError("cine event listeners were not released exactly once")


def _set_rate(client: WebDriverClient, target: int) -> None:
    """Set one exact rate through trusted range-keyboard input."""
    if not 1 <= target <= MAX_CINE_RATE:
        raise BrowserRuntimeError(f"cine target rate is outside 1..={MAX_CINE_RATE}: {target}")
    client.click(client.find("#cine-rate"))
    client.key_press("Home", source_id="ritk-cine-rate-keyboard")
    steps = target - 1
    if steps:
        key = "\ue014"
        actions = []
        for _ in range(steps):
            actions.extend(({"type": "keyDown", "value": key}, {"type": "keyUp", "value": key}))
        client.perform_actions([{"type": "key", "id": "ritk-cine-rate-batch", "actions": actions}])
    client.key_press("Enter", source_id="ritk-cine-rate-keyboard")
    settle_canvas_input(client)


def _stopped_state(client: WebDriverClient) -> dict[str, Any]:
    """Read the consumer controls after the RITK viewer has stopped."""
    result = client.execute(STOPPED_STATE_SCRIPT)
    if not isinstance(result, dict):
        raise BrowserRuntimeError("stopped cine controls returned an unexpected shape")
    if (
        result.get("button_disabled") is not True
        or result.get("rate_disabled") is not True
        or result.get("button_pressed") != "false"
        or result.get("button_text") != "Play"
        or result.get("rate_value") != "12"
        or result.get("output") != "12 FPS"
    ):
        raise BrowserRuntimeError(f"stopped cine controls retained active state: {result!r}")
    return result

def capture_cine_gallery(client: WebDriverClient, output_directory: pathlib.Path) -> dict[str, Any]:
    """Drive Play/Pause and FPS controls and record exact browser evidence."""
    if not isinstance(output_directory, pathlib.Path):
        raise BrowserRuntimeError("cine output directory must be a pathlib.Path")
    directory = _safe_path(output_directory, directory=ROOT / "output")
    directory.mkdir(parents=True, exist_ok=True)
    initial = _snapshot(client)
    if initial["controls"]["button_disabled"] or initial["controls"]["rate_disabled"]:
        raise BrowserRuntimeError("cine controls remained disabled after study load")
    invalid = _invalid_api_probes(client)
    if _snapshot(client) != initial:
        raise BrowserRuntimeError("invalid cine rate probes mutated the viewer")
    expected_listener_count = _install_trace(client)
    trace_installed = True
    actions_released = False
    try:
        client.click(client.find("#cine-toggle"))
        _wait_for_state(client, True)
        _wait_for_frame(client, initial)
        playing = _snapshot(client)
        _set_rate(client, CINE_RATE_TARGET)
        _wait_for_state(client, True, CINE_RATE_TARGET)
        rated = _snapshot(client)
        _wait_for_frame(client, playing)
        client.click(client.find("#cine-toggle"))
        _wait_for_state(client, False, CINE_RATE_TARGET)
        paused = _snapshot(client)
        settle_canvas_input(client)
        stable = _snapshot(client)
        if stable["canvases"] != paused["canvases"]:
            raise BrowserRuntimeError("paused cine advanced a slice during the settle window")
        events = _consume_trace(client)
        client.release_actions()
        actions_released = True
        _cleanup_trace(client, expected_listener_count)
        trace_installed = False
        screenshots = {
            "window": _write_png(client.screenshot(), directory, "gallery-cine.png", "window"),
            "controls": _write_png(
                client.element_screenshot(client.find(".cine-controls")),
                directory,
                "gallery-cine-controls.png",
                "element",
            ),
        }
        sample_before_stop = client.execute("return window.metisGallery.sample();")
        client.execute("window.metisGallery.stop(); return true;")
        sample_after_stop = client.execute("return window.metisGallery.sample();")
        if not isinstance(sample_after_stop, dict) or sample_after_stop.get("mounted") is not False or sample_after_stop.get("consumer_listeners") != 0:
            raise BrowserRuntimeError(f"RITK viewer did not release listeners after stop: {sample_after_stop!r}")
        stopped = _stopped_state(client)
        evidence = {
            "schema": 1,
            "consumer": "ritk-snap",
            "initial": initial,
            "invalid_api_probes": invalid,
            "play": playing,
            "rate": rated,
            "pause": paused,
            "paused_stable": stable,
            "events": events,
            "screenshots": screenshots,
            "samples": {"before_stop": sample_before_stop, "after_stop": sample_after_stop},
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
            "browser_gallery_cine.py": hashlib.sha256((source_root / "browser_gallery_cine.py").read_bytes()).hexdigest(),
            "browser_gallery.py": hashlib.sha256((source_root / "browser_gallery.py").read_bytes()).hexdigest(),
        }
        evidence_path = _safe_path(directory / "gallery-cine.json", directory=directory)
        evidence["artifact"] = evidence_path.relative_to(ROOT).as_posix()
        encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if len(encoded.encode("utf-8")) > 512 * 1024:
            raise BrowserRuntimeError("cine evidence exceeds the 512 KiB trace bound")
        evidence_path.write_text(encoded, encoding="utf-8", newline="\n")
        return evidence
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if trace_installed:
            try:
                _cleanup_trace(client, expected_listener_count)
            except BrowserRuntimeError as error:
                cleanup_errors.append(("cine event listener cleanup", error))
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
