import init, { AsterSliceWorkspace } from "./consumer/ritk_snap.js";

const canvas = document.querySelector("#workstation");
const notice = document.querySelector("#notice");
const now = () => performance.now() / 1000;
const modifiers = e => Number(e.ctrlKey || e.metaKey) | Number(e.shiftKey) << 1 |
  Number(e.altKey) << 2 | Number(e.metaKey) << 3;

function report(error) {
  notice.textContent = String(error);
  notice.hidden = false;
}

async function presenter(renderer) {
  if (renderer === "webgpu") {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU requires a graphics adapter. Open the viewer without renderer=webgpu to use Canvas2D.");
    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    context.configure({device, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT});
    device.lost.then(info => report(info.message));
    return bytes => device.queue.writeTexture({texture: context.getCurrentTexture()}, bytes,
      {bytesPerRow: canvas.width * 4}, [canvas.width, canvas.height]);
  }
  // The CPU compositor uploads RGBA and readers must be able to retrieve the presented pixels.
  const context = canvas.getContext("2d", {alpha: false, willReadFrequently: true});
  if (!context) throw new Error("Canvas2D could not start.");
  return bytes => context.putImageData(new ImageData(
    new Uint8ClampedArray(bytes.buffer, bytes.byteOffset, bytes.byteLength), canvas.width, canvas.height), 0, 0);
}

async function start() {
  await init();
  const query = new URLSearchParams(location.search);
  const draw = await presenter(query.get("renderer"));
  const viewer = new AsterSliceWorkspace(query.get("layout") === "responsive" ? "responsive" : query.get("projection") ?? "");
  const input = document.querySelector("#file-input");
  const zone = document.querySelector("#drop-zone");
  if (!input || !zone) throw new Error("The study picker could not start.");
  const listeners = new AbortController();
  const listen = (el, name, handler, options = {}) => el.addEventListener(name, handler,
    {...options, signal: listeners.signal});
  listen(input, "change", () => { input.value = ""; });
  let frame;
  let sized = false;
  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h || (sized && canvas.width === w && canvas.height === h)) return;
    sized = true;
    viewer.resize(w, h, now());
    canvas.width = w;
    canvas.height = h;
  };
  const refresh = () => {
    try {
      resize();
      if (viewer.tick(now())) draw(viewer.rgba());
      frame = requestAnimationFrame(refresh);
    } catch (error) { report(error); }
  };
  const route = callback => e => {
    if (!e.isTrusted) return;
    try {
      callback(e);
      if (viewer.take_open_request()) { input.value = ""; input.click(); }
    } catch (error) { report(error); }
  };
  for (const phase of ["pointerdown", "pointerup", "pointermove", "pointercancel"]) {
    listen(canvas, phase, route(e => {
      if (phase === "pointerdown") { canvas.focus(); canvas.setPointerCapture(e.pointerId); }
      viewer.pointer(phase, e.offsetX, e.offsetY, e.button, modifiers(e), now());
    }));
  }
  // A native double-click marker follows the second down, before its release.
  listen(canvas, "mousedown", route(e => {
    if (e.detail === 2) viewer.pointer("dblclick", e.offsetX, e.offsetY, e.button, modifiers(e), now());
  }));
  listen(canvas, "lostpointercapture", route(() => viewer.lifecycle("lostpointercapture", now())));
  listen(canvas, "contextmenu", e => e.preventDefault());
  listen(canvas, "wheel", route(e => {
    e.preventDefault();
    viewer.wheel(e.offsetX, e.offsetY, e.deltaX, -e.deltaY, modifiers(e), now());
  }), {passive: false});
  for (const phase of ["blur", "focus"]) {
    listen(window, phase, route(() => viewer.lifecycle(phase, now())));
  }
  for (const phase of ["keydown", "keyup"]) {
    listen(window, phase, route(e => {
      if (e.isComposing || e.target !== canvas) return;
      // Reserve navigation keys that have a competing browser default. Forward
      // every key to the workstation so F4, Tab and their modifiers stay native.
      if (["Tab", " ", "PageUp", "PageDown", "Home", "End", "ArrowUp", "ArrowDown",
           "ArrowRight", "ArrowLeft", "Backspace", "F4"].includes(e.key) ||
          ((e.ctrlKey || e.metaKey) && ["KeyO", "KeyM", "Digit0"].includes(e.code))) {
        e.preventDefault();
      }
      viewer.key(phase === "keydown", e.keyCode, e.repeat, modifiers(e), now());
      if (phase === "keydown" && [...e.key].length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey)
        viewer.text(e.key, now());
    }));
  }
  listen(window, "dragenter", e => {
    if (e.dataTransfer?.types.includes("Files")) document.body.classList.add("dragging");
  });
  listen(window, "dragleave", e => { if (!e.relatedTarget) document.body.classList.remove("dragging"); });
  listen(window, "drop", () => document.body.classList.remove("dragging"));
  const transfer = new MutationObserver(() => {
    if (zone.dataset.byteState === "failed" || zone.dataset.dropState === "rejected") {
      const status = zone.dataset.byteState === "failed" ? "#drop-byte-status" : "#drop-status";
      report(document.querySelector(status).textContent);
    } else { notice.hidden = true; }
  });
  transfer.observe(zone, {attributes: true});
  listen(window, "pagehide", e => {
    if (e.persisted) return;
    listeners.abort();
    transfer.disconnect();
    cancelAnimationFrame(frame);
    viewer.free();
  });
  canvas.focus();
  refresh();
  window.asterSlice = Object.freeze({state: () => JSON.parse(viewer.state())});
}

start().catch(report);
