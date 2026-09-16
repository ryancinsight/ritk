"""Browser gallery state, event-trace and rejection contracts."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from browser_canvas import settle_canvas_input
from browser_protocol import BrowserRuntimeError, WebDriverClient


AXES = ("axial", "coronal", "sagittal")
MAX_SLICE_COUNT = 4096
MAX_SLIDER_EVENTS = 256
ARROW_BATCH_SIZE = 16
SLIDER_EVENT_TYPES = (
    "click",
    "pointerdown",
    "pointermove",
    "pointerup",
    "keydown",
    "keyup",
    "input",
    "change",
)

GALLERY_SNAPSHOT_SCRIPT = """
const done = arguments[arguments.length - 1];
const axes = arguments[0];
const hex = (bytes) => Array.from(new Uint8Array(bytes),
  (byte) => byte.toString(16).padStart(2, "0")).join("");
Promise.all(axes.map(async (axis) => {
  const canvas = document.getElementById(`ritk-snap-${axis}`);
  const slider = document.getElementById(`slice-${axis}`);
  const output = document.getElementById(`position-${axis}`);
  if (!canvas || canvas.tagName.toLowerCase() !== "canvas" ||
      !slider || slider.tagName.toLowerCase() !== "input" ||
      slider.type !== "range" || !output) {
    return {axis, error: "gallery controls were not found"};
  }
  const pixels = canvas.getContext("2d", {willReadFrequently: true})
    .getImageData(0, 0, canvas.width, canvas.height).data;
  const digest = await crypto.subtle.digest("SHA-256", pixels);
  const rect = slider.getBoundingClientRect();
  return {
    axis,
    canvas: {
      width: canvas.width,
      height: canvas.height,
      slice_index: canvas.getAttribute("data-ritk-slice-index"),
      slice_count: canvas.getAttribute("data-ritk-slice-count"),
      frame_generation: canvas.getAttribute("data-ritk-frame-generation"),
      frame_state: canvas.getAttribute("data-ritk-frame-state"),
      rgba_sha256: hex(digest),
    },
    slider: {
      min: slider.min,
      max: slider.max,
      step: slider.step,
      value: slider.value,
      disabled: slider.disabled,
      width: rect.width,
      height: rect.height,
    },
    output: output.textContent,
  };
})).then((states) => done({ok: true, states}),
  (error) => done({ok: false, error: String(error && error.message || error)}));
"""

INSTALL_SLIDER_TRACE_SCRIPT = """
const ids = arguments[0];
const types = arguments[1];
const maxEvents = arguments[2];
if (window.__metisGallerySliderTraceState) {
  return {ok: false, error: "gallery slider trace is already installed"};
}
const events = Object.fromEntries(ids.map((id) => [id, []]));
const overflow = Object.fromEntries(ids.map((id) => [id, false]));
const registrations = [];
for (const id of ids) {
  const slider = document.getElementById(id);
  if (!slider || slider.tagName.toLowerCase() !== "input" || slider.type !== "range") {
    return {ok: false, error: `range input ${id} was not found`};
  }
  for (const type of types) {
    const listener = (event) => {
      const records = events[id];
      if (records.length >= maxEvents) {
        overflow[id] = true;
        return;
      }
      records.push({
        type: event.type,
        is_trusted: event.isTrusted === true,
        target_id: event.target && typeof event.target.id === "string"
          ? event.target.id : null,
        key: typeof event.key === "string" ? event.key : null,
        value: slider.value,
      });
    };
    slider.addEventListener(type, listener, {capture: true, passive: true});
    registrations.push({slider, type, listener});
  }
}
window.__metisGallerySliderTraceState = {events, overflow, registrations};
return {ok: true, listener_count: registrations.length};
"""

READ_SLIDER_TRACE_SCRIPT = """
const state = window.__metisGallerySliderTraceState;
const id = arguments[0];
if (!state || !Object.prototype.hasOwnProperty.call(state.events, id)) return null;
const events = state.events[id];
state.events[id] = [];
const overflow = state.overflow[id] === true;
state.overflow[id] = false;
return {events, overflow};
"""

CLEANUP_SLIDER_TRACE_SCRIPT = """
const state = window.__metisGallerySliderTraceState;
if (!state) return {ok: true, listener_count: 0};
for (const registration of state.registrations) {
  registration.slider.removeEventListener(
    registration.type, registration.listener, true);
}
const listenerCount = state.registrations.length;
delete window.__metisGallerySliderTraceState;
return {ok: true, listener_count: listenerCount};
"""

INVALID_SLICE_API_PROBE_SCRIPT = """
const done = arguments[arguments.length - 1];
const axes = arguments[0];
const invalid = [
  ["NaN", NaN],
  ["Infinity", Infinity],
  ["-Infinity", -Infinity],
  ["-1", -1],
  ["0.5", 0.5],
  ["4294967296", 4294967296],
];
import("./consumer/ritk_snap.js").then(({select_web_slice}) => {
  const probes = [];
  for (const [axis, axisName] of axes.entries()) {
    for (const [value, argument] of invalid) {
      let rejected = false;
      let error = null;
      try {
        select_web_slice(axis, argument);
      } catch (failure) {
        rejected = true;
        error = String(failure && failure.message || failure);
      }
      probes.push({axis: axisName, value, rejected, error});
    }
  }
  done({ok: true, probes});
}, (error) => done({ok: false, error: String(error && error.message || error)}));
"""


def _validate_expected_counts(expected_counts: Mapping[str, int]) -> dict[str, int]:
    """Validate the caller's independent slice-count oracle."""
    if not isinstance(expected_counts, Mapping) or set(expected_counts) != set(AXES):
        raise BrowserRuntimeError(f"expected slice counts must contain exactly {AXES!r}")
    validated = {}
    for axis in AXES:
        count = expected_counts[axis]
        if type(count) is not int or not 2 <= count <= MAX_SLICE_COUNT:
            raise BrowserRuntimeError(
                f"expected {axis} slice count must be between 2 and {MAX_SLICE_COUNT}"
            )
        validated[axis] = count
    return validated


def _integer(value: Any, label: str, *, lower: int, upper: int) -> int:
    """Parse one bounded decimal DOM value without accepting loose coercions."""
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise BrowserRuntimeError(f"{label} is not an unsigned decimal integer: {value!r}")
    parsed = int(value)
    if not lower <= parsed <= upper:
        raise BrowserRuntimeError(f"{label} is outside [{lower}, {upper}]: {parsed}")
    return parsed


def _snapshot(client: WebDriverClient, expected_counts: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """Read exact range state and canvas RGBA digests for every anatomical axis."""
    result = client.execute_async(GALLERY_SNAPSHOT_SCRIPT, [list(AXES)])
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"gallery state could not be sampled: {detail!r}")
    states = result.get("states")
    if not isinstance(states, list) or len(states) != len(AXES):
        raise BrowserRuntimeError("gallery state returned an unexpected anatomical-axis count")
    snapshot: dict[str, dict[str, Any]] = {}
    for position, state in enumerate(states):
        axis = AXES[position]
        if not isinstance(state, dict) or state.get("axis") != axis or state.get("error"):
            raise BrowserRuntimeError(f"gallery state for {axis!r} is malformed: {state!r}")
        canvas = state.get("canvas")
        slider = state.get("slider")
        if not isinstance(canvas, dict) or not isinstance(slider, dict):
            raise BrowserRuntimeError(f"gallery state for {axis!r} omitted canvas or slider state")
        count = expected_counts[axis]
        index = _integer(canvas.get("slice_index"), f"{axis} canvas slice index", lower=0, upper=count - 1)
        actual_count = _integer(
            canvas.get("slice_count"), f"{axis} canvas slice count", lower=2, upper=MAX_SLICE_COUNT
        )
        generation = _integer(
            canvas.get("frame_generation"),
            f"{axis} frame generation",
            lower=1,
            upper=(1 << 53) - 1,
        )
        width = canvas.get("width")
        height = canvas.get("height")
        digest = canvas.get("rgba_sha256")
        if actual_count != count:
            raise BrowserRuntimeError(
                f"{axis} reports {actual_count} slices; independent oracle expects {count}"
            )
        if (
            type(width) is not int
            or type(height) is not int
            or not 1 <= width <= 4096
            or not 1 <= height <= 4096
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or canvas.get("frame_state") != "presented"
        ):
            raise BrowserRuntimeError(f"{axis} canvas did not expose a presented bounded RGBA frame")
        slider_width = slider.get("width")
        slider_height = slider.get("height")
        if (
            slider.get("min") != "0"
            or slider.get("max") != str(count - 1)
            or slider.get("step") != "1"
            or slider.get("value") != str(index)
            or slider.get("disabled") is not False
            or not isinstance(slider_width, (int, float))
            or not isinstance(slider_height, (int, float))
            or not math.isfinite(float(slider_width))
            or not math.isfinite(float(slider_height))
            or not 8.0 <= float(slider_width) <= 8192.0
            or not 1.0 <= float(slider_height) <= 1024.0
        ):
            raise BrowserRuntimeError(f"{axis} range control does not match its canvas slice state")
        expected_output = f"{index + 1} / {count}"
        if state.get("output") != expected_output:
            raise BrowserRuntimeError(
                f"{axis} position output is {state.get('output')!r}; expected {expected_output!r}"
            )
        snapshot[axis] = {
            "index": index,
            "count": count,
            "generation": generation,
            "rgba_sha256": digest,
            "width": width,
            "height": height,
            "slider_width": float(slider_width),
            "slider_height": float(slider_height),
            "output": expected_output,
        }
    return snapshot


def _install_event_trace(client: WebDriverClient) -> int:
    """Install bounded capture-phase observers on all three range controls."""
    ids = [f"slice-{axis}" for axis in AXES]
    result = client.execute(
        INSTALL_SLIDER_TRACE_SCRIPT,
        [ids, list(SLIDER_EVENT_TYPES), MAX_SLIDER_EVENTS],
    )
    expected = len(ids) * len(SLIDER_EVENT_TYPES)
    if (
        not isinstance(result, dict)
        or result.get("ok") is not True
        or result.get("listener_count") != expected
    ):
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"gallery slider event trace could not be installed: {detail!r}")
    return expected


def _consume_events(
    client: WebDriverClient,
    axis: str,
    required_types: Sequence[str],
    *,
    expected_key: str | None = None,
) -> dict[str, Any]:
    """Validate every trusted browser event and return bounded summary evidence."""
    slider_id = f"slice-{axis}"
    result = client.execute(READ_SLIDER_TRACE_SCRIPT, [slider_id])
    if (
        not isinstance(result, dict)
        or result.get("overflow") is not False
        or not isinstance(result.get("events"), list)
    ):
        raise BrowserRuntimeError(f"gallery event trace for {slider_id!r} is malformed or overflowed")
    events = result["events"]
    if not 1 <= len(events) <= MAX_SLIDER_EVENTS:
        raise BrowserRuntimeError(f"gallery event trace for {slider_id!r} is empty or unbounded")
    counts: Counter[str] = Counter()
    values = []
    for event in events:
        if not isinstance(event, dict):
            raise BrowserRuntimeError(f"gallery event trace for {slider_id!r} contains a non-object")
        event_type = event.get("type")
        if (
            event_type not in SLIDER_EVENT_TYPES
            or event.get("is_trusted") is not True
            or event.get("target_id") != slider_id
        ):
            raise BrowserRuntimeError(f"gallery event trace for {slider_id!r} contains untrusted delivery")
        if event_type in ("keydown", "keyup") and event.get("key") != expected_key:
            raise BrowserRuntimeError(
                f"gallery keyboard event for {slider_id!r} reported {event.get('key')!r}; "
                f"expected {expected_key!r}"
            )
        value = event.get("value")
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise BrowserRuntimeError(f"gallery event for {slider_id!r} has invalid range value")
        counts[event_type] += 1
        values.append(int(value))
    missing = set(required_types) - set(counts)
    if missing:
        raise BrowserRuntimeError(
            f"gallery event trace for {slider_id!r} omitted {sorted(missing)!r}"
        )
    return {
        "event_count": len(events),
        "event_types": dict(sorted(counts.items())),
        "value_min": min(values),
        "value_max": max(values),
        "all_trusted": True,
        "target_id": slider_id,
        "key": expected_key,
    }


def _cleanup_event_trace(client: WebDriverClient, expected_listener_count: int) -> int:
    """Remove every observer and report the exact released-listener count."""
    result = client.execute(CLEANUP_SLIDER_TRACE_SCRIPT)
    if (
        not isinstance(result, dict)
        or result.get("ok") is not True
        or result.get("listener_count") != expected_listener_count
    ):
        raise BrowserRuntimeError("gallery slider event listeners were not released exactly once")
    return expected_listener_count


def _probe_invalid_slice_api(
    client: WebDriverClient,
    expected_counts: Mapping[str, int],
    before: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require the consumer API to reject non-integral and out-of-range indices."""
    result = client.execute_async(INVALID_SLICE_API_PROBE_SCRIPT, [list(AXES)])
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else result
        raise BrowserRuntimeError(f"invalid slice API probes could not execute: {detail!r}")
    probes = result.get("probes")
    expected_probe_count = len(AXES) * 6
    if not isinstance(probes, list) or len(probes) != expected_probe_count:
        raise BrowserRuntimeError("invalid slice API probes returned an unexpected result count")
    for probe in probes:
        if (
            not isinstance(probe, dict)
            or probe.get("axis") not in AXES
            or probe.get("value") not in {"NaN", "Infinity", "-Infinity", "-1", "0.5", "4294967296"}
            or probe.get("rejected") is not True
            or not isinstance(probe.get("error"), str)
            or not probe["error"]
        ):
            raise BrowserRuntimeError(f"consumer accepted or misreported an invalid slice index: {probe!r}")
    settle_canvas_input(client)
    after = _snapshot(client, expected_counts)
    for axis in AXES:
        if (
            after[axis]["index"] != before[axis]["index"]
            or after[axis]["generation"] != before[axis]["generation"]
            or after[axis]["rgba_sha256"] != before[axis]["rgba_sha256"]
        ):
            raise BrowserRuntimeError(f"invalid slice API probes mutated the {axis} canvas")
    return probes


def _validate_transition(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    axis: str,
    expected_index: int | None,
) -> None:
    """Require target repaint and preservation of the other anatomical planes."""
    if expected_index is not None and after[axis]["index"] != expected_index:
        raise BrowserRuntimeError(
            f"{axis} slider selected {after[axis]['index']}; expected {expected_index}"
        )
    changed_index = after[axis]["index"] != before[axis]["index"]
    if changed_index and after[axis]["generation"] <= before[axis]["generation"]:
        raise BrowserRuntimeError(f"{axis} slice changed without a new frame generation")
    if not changed_index and (
        after[axis]["generation"] != before[axis]["generation"]
        or after[axis]["rgba_sha256"] != before[axis]["rgba_sha256"]
    ):
        raise BrowserRuntimeError(f"{axis} no-op selection repainted or changed its RGBA frame")
    for other_axis in AXES:
        if other_axis == axis:
            continue
        if (
            after[other_axis]["index"] != before[other_axis]["index"]
            or after[other_axis]["rgba_sha256"] != before[other_axis]["rgba_sha256"]
        ):
            raise BrowserRuntimeError(f"{axis} slider changed the {other_axis} anatomical plane")
