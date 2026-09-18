"""Browser JavaScript contracts for the RITK diagnostic-tool gallery."""

TOOL_EVENT_TYPES = ("click", "keydown", "keyup", "pointerdown", "pointermove", "pointerup")

TOOL_SNAPSHOT_SCRIPT = """
const axes = arguments[0];
const readCanvas = (axis) => {
  const canvas = document.getElementById(`ritk-snap-${axis}`);
  if (!(canvas instanceof HTMLCanvasElement)) return {axis, error: "canvas missing"};
  return {
    axis,
    load_state: canvas.getAttribute("data-ritk-load-state"),
    frame_state: canvas.getAttribute("data-ritk-frame-state"),
    slice_index: canvas.getAttribute("data-ritk-slice-index"),
    frame_generation: canvas.getAttribute("data-ritk-frame-generation"),
    active_tool_index: canvas.getAttribute("data-ritk-active-tool-index"),
    active_tool: canvas.getAttribute("data-ritk-active-tool"),
  };
};
const toolbar = document.getElementById("tool-buttons");
const output = document.getElementById("active-tool");
if (!(toolbar instanceof HTMLDivElement) || !(output instanceof HTMLOutputElement)) {
  return {ok: false, error: "tool controls are missing"};
}
return {ok: true, canvases: axes.map(readCanvas), controls: {
  output: output.textContent || "",
  buttons: Array.from(toolbar.querySelectorAll("button"), (button) => ({
    index: button.dataset.toolIndex || "",
    label: button.textContent || "",
    pressed: button.getAttribute("aria-pressed"),
    disabled: button.disabled,
  })),
}};
"""

WAIT_TOOL_STATE_SCRIPT = """
const done = arguments[arguments.length - 1];
const index = String(arguments[0]);
const label = arguments[1];
const limit = arguments[2];
const ready = () => {
  const canvases = ["axial", "coronal", "sagittal"].map((axis) =>
    document.getElementById(`ritk-snap-${axis}`));
  const toolbar = document.getElementById("tool-buttons");
  const output = document.getElementById("active-tool");
  return canvases.every((canvas) => canvas instanceof HTMLCanvasElement &&
    canvas.getAttribute("data-ritk-active-tool-index") === index &&
    canvas.getAttribute("data-ritk-active-tool") === label) &&
    toolbar instanceof HTMLDivElement && output instanceof HTMLOutputElement &&
    Array.from(toolbar.querySelectorAll("button")).some((button) =>
      button.dataset.toolIndex === index && button.getAttribute("aria-pressed") === "true") &&
    output.textContent === `Active tool: ${label}`;
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
  done({ok: false});
}, limit);
"""

WAIT_TOOL_FRAME_SCRIPT = """
const done = arguments[arguments.length - 1];
const previous = arguments[0];
const limit = arguments[1];
const read = () => ["axial", "coronal", "sagittal"].map((axis) => {
  const canvas = document.getElementById(`ritk-snap-${axis}`);
  return canvas instanceof HTMLCanvasElement ? {
    axis,
    generation: Number(canvas.getAttribute("data-ritk-frame-generation")),
    frame: canvas.getAttribute("data-ritk-frame-state"),
  } : null;
});
const ready = () => {
  const current = read();
  return current.every((state, position) => state && state.frame === "presented" &&
    Number.isSafeInteger(state.generation) && state.generation > previous[position]);
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

INVALID_TOOL_API_PROBE_SCRIPT = """
const done = arguments[arguments.length - 1];
const invalid = [
  ["NaN", NaN], ["Infinity", Infinity], ["-Infinity", -Infinity],
  ["-1", -1], ["0.5", 0.5], ["4294967296", 4294967296],
];
import("./consumer/ritk_snap.js").then(({select_web_tool, web_tool_count}) => {
  let count;
  try { count = web_tool_count(); } catch (error) {
    done({ok: false, error: String(error && error.message || error)});
    return;
  }
  invalid.push(["tool-count", count]);
  const probes = invalid.map(([value, argument]) => {
    let rejected = false;
    let error = null;
    try { select_web_tool(argument); }
    catch (failure) {
      rejected = true;
      error = String(failure && failure.message || failure);
    }
    return {value, rejected, error};
  });
  done({ok: true, count, probes});
}, (error) => done({ok: false, error: String(error && error.message || error)}));
"""

INSTALL_TOOL_TRACE_SCRIPT = """
const types = arguments[0];
const maxEvents = arguments[1];
if (window.__ritkToolTrace) return {ok: false, error: "tool trace already installed"};
const toolbar = document.getElementById("tool-buttons");
const canvas = document.getElementById("ritk-snap-axial");
if (!(toolbar instanceof HTMLDivElement) || !(canvas instanceof HTMLCanvasElement)) {
  return {ok: false, error: "tool trace targets are missing"};
}
const events = [];
const registrations = [];
const add = (element, target) => {
  for (const type of types) {
    if (target === "toolbar" && type.startsWith("pointer")) continue;
    if (target === "canvas" && type === "click") continue;
    const listener = (event) => {
      if (events.length >= maxEvents) return;
      const button = event.target instanceof HTMLButtonElement ? event.target : null;
      events.push({
        type: event.type,
        target: target,
        target_id: event.target && event.target.id ? event.target.id : null,
        tool_index: button ? button.dataset.toolIndex || null : null,
        trusted: event.isTrusted === true,
        key: typeof event.key === "string" ? event.key : null,
        code: typeof event.code === "string" ? event.code : null,
      });
    };
    element.addEventListener(type, listener, {capture: true, passive: true});
    registrations.push({element, type, listener});
  }
};
add(toolbar, "toolbar");
add(canvas, "canvas");
window.__ritkToolTrace = {events, registrations, maxEvents};
return {ok: true, listener_count: registrations.length};
"""

READ_TOOL_TRACE_SCRIPT = """
const trace = window.__ritkToolTrace;
if (!trace) return null;
const overflow = trace.events.length >= trace.maxEvents;
const events = trace.events.splice(0);
return {events, overflow};
"""

CLEANUP_TOOL_TRACE_SCRIPT = """
const trace = window.__ritkToolTrace;
if (!trace) return {ok: true, listener_count: 0};
for (const registration of trace.registrations) {
  registration.element.removeEventListener(registration.type, registration.listener, true);
}
const listenerCount = trace.registrations.length;
delete window.__ritkToolTrace;
return {ok: true, listener_count: listenerCount};
"""

FOCUS_TOOL_CANVAS_SCRIPT = """
const canvas = document.getElementById("ritk-snap-axial");
if (!(canvas instanceof HTMLCanvasElement)) return false;
canvas.focus();
return document.activeElement === canvas;
"""

STOPPED_TOOL_STATE_SCRIPT = """
const toolbar = document.getElementById("tool-buttons");
const output = document.getElementById("active-tool");
if (!(toolbar instanceof HTMLDivElement) || !(output instanceof HTMLOutputElement)) return null;
const buttons = Array.from(toolbar.querySelectorAll("button"));
return {button_count: buttons.length, disabled: buttons.every((button) => button.disabled),
  output: output.textContent || ""};
"""
