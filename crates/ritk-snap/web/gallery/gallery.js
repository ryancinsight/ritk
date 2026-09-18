// RITK owns the viewer loop and consumes Metis's bounded byte handoff.
const status = document.getElementById("gallery-status");
const describeError = (error) => error instanceof Error ? error.message : String(error);
try {
  const { default: init, start_web_orthogonal_canvases,
    start_web_orthogonal_canvases_gpu, stop_web_canvas, web_canvas_listener_count,
    select_web_slice, set_web_cine_rate, set_web_window_preset,
    toggle_web_cine, select_web_tool, web_tool_count, web_tool_name,
    web_window_preset_count, web_window_preset_name } =
    await import("./consumer/ritk_snap.js");
  const runtime = await init();
  const renderer = new URLSearchParams(window.location.search).get("renderer") === "webgpu"
    ? "webgpu" : "raster";
  let mounted = false;
  const windowPreset = document.getElementById("window-preset");
  const windowLevel = document.getElementById("window-level");
  const cineToggle = document.getElementById("cine-toggle");
  const cineRate = document.getElementById("cine-rate");
  const cineRateValue = document.getElementById("cine-rate-value");
  const toolButtons = document.getElementById("tool-buttons");
  const activeTool = document.getElementById("active-tool");
  if (!(windowPreset instanceof HTMLSelectElement) ||
      !(windowLevel instanceof HTMLOutputElement) ||
      !(cineToggle instanceof HTMLButtonElement) ||
      !(cineRate instanceof HTMLInputElement) ||
      !(cineRateValue instanceof HTMLOutputElement) ||
      !(toolButtons instanceof HTMLDivElement) ||
      !(activeTool instanceof HTMLOutputElement)) {
    throw new Error("RITK presentation controls are missing");
  }
  let presetSignature = "";
  const syncPresentation = () => {
    const canvas = document.getElementById("ritk-snap-axial");
    const ready = mounted && canvas instanceof HTMLCanvasElement &&
      canvas.getAttribute("data-ritk-load-state") === "ready" &&
      canvas.getAttribute("data-ritk-frame-state") === "presented";
    if (!ready) {
      windowPreset.disabled = true;
      windowPreset.replaceChildren(new Option("Load a study to choose a preset", ""));
      windowPreset.value = "";
      windowLevel.textContent = "No study";
      presetSignature = "";
      return;
    }
    let count;
    try {
      count = web_window_preset_count();
    } catch (error) {
      windowPreset.disabled = true;
      windowLevel.textContent = `Preset list unavailable: ${describeError(error)}`;
      return;
    }
    const labels = [];
    try {
      for (let index = 0; index < count; index += 1) {
        labels.push(web_window_preset_name(index));
      }
    } catch (error) {
      windowPreset.disabled = true;
      windowLevel.textContent = `Preset list unavailable: ${describeError(error)}`;
      return;
    }
    const signature = labels.join("\u001f");
    if (signature !== presetSignature) {
      windowPreset.replaceChildren(...labels.map((label, index) =>
        new Option(label, String(index))));
      presetSignature = signature;
    }
    windowPreset.disabled = false;
    const selected = canvas.getAttribute("data-ritk-window-preset-index") ?? "";
    windowPreset.value = selected;
    const center = canvas.getAttribute("data-ritk-window-center") ?? "?";
    const width = canvas.getAttribute("data-ritk-window-width") ?? "?";
    windowLevel.textContent = `Center ${center} · Width ${width}`;
  };
  windowPreset.addEventListener("change", () => {
    if (windowPreset.value === "") return;
    try {
      set_web_window_preset(Number(windowPreset.value));
    } catch (error) {
      status.textContent = `Window/level preset could not change: ${describeError(error)}`;
      syncPresentation();
    }
  });
  const syncCine = () => {
    const canvases = ["axial", "coronal", "sagittal"]
      .map((name) => document.getElementById(`ritk-snap-${name}`));
    const canvas = canvases[0];
    const ready = mounted && canvases.every((candidate) =>
      candidate instanceof HTMLCanvasElement &&
      candidate.getAttribute("data-ritk-load-state") === "ready" &&
      candidate.getAttribute("data-ritk-frame-state") === "presented");
    if (!ready) {
      cineToggle.disabled = true;
      cineToggle.textContent = "Play";
      cineToggle.setAttribute("aria-pressed", "false");
      cineRate.disabled = true;
      cineRate.value = "12";
      cineRateValue.textContent = "12 FPS";
      return;
    }
    const enabled = canvas.getAttribute("data-ritk-cine-enabled") === "true";
    const rawRate = Number(canvas.getAttribute("data-ritk-cine-fps"));
    const rate = Number.isInteger(rawRate) && rawRate >= 1 && rawRate <= 60
      ? rawRate : 12;
    cineToggle.disabled = false;
    cineToggle.textContent = enabled ? "Pause" : "Play";
    cineToggle.setAttribute("aria-pressed", String(enabled));
    cineRate.disabled = false;
    cineRate.value = String(rate);
    cineRateValue.textContent = `${rate} FPS`;
  };
  cineToggle.addEventListener("click", () => {
    try {
      toggle_web_cine();
    } catch (error) {
      status.textContent = `Cine playback could not change: ${describeError(error)}`;
      syncCine();
    }
  });
  cineRate.addEventListener("input", () => {
    try {
      set_web_cine_rate(cineRate.valueAsNumber);
    } catch (error) {
      status.textContent = `Cine rate could not change: ${describeError(error)}`;
      syncCine();
    }
  });
  let toolSignature = "";
  const syncTools = () => {
    const canvases = ["axial", "coronal", "sagittal"]
      .map((name) => document.getElementById(`ritk-snap-${name}`));
    const canvas = canvases[0];
    const ready = mounted && canvases.every((candidate) =>
      candidate instanceof HTMLCanvasElement &&
      candidate.getAttribute("data-ritk-load-state") === "ready" &&
      candidate.getAttribute("data-ritk-frame-state") === "presented");
    if (!ready) {
      for (const button of toolButtons.querySelectorAll("button")) button.disabled = true;
      activeTool.textContent = "No study";
      return;
    }
    let count;
    try {
      count = web_tool_count();
    } catch (error) {
      activeTool.textContent = `Tool list unavailable: ${describeError(error)}`;
      for (const button of toolButtons.querySelectorAll("button")) button.disabled = true;
      return;
    }
    if (!Number.isSafeInteger(count) || count < 1 || count > 32) {
      throw new Error(`invalid browser tool count: ${count}`);
    }
    const labels = [];
    for (let index = 0; index < count; index += 1) {
      labels.push(web_tool_name(index));
    }
    const signature = labels.join("\u001f");
    if (signature !== toolSignature) {
      toolButtons.replaceChildren(...labels.map((label, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.dataset.toolIndex = String(index);
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("click", () => {
          try {
            select_web_tool(Number(button.dataset.toolIndex));
            syncTools();
          } catch (error) {
            status.textContent = `Viewer tool could not change: ${describeError(error)}`;
            syncTools();
          }
        });
        return button;
      }));
      toolSignature = signature;
    }
    const selected = Number(canvas.getAttribute("data-ritk-active-tool-index"));
    const selectedName = canvas.getAttribute("data-ritk-active-tool") ?? "?";
    activeTool.textContent = `Active tool: ${selectedName}`;
    for (const button of toolButtons.querySelectorAll("button")) {
      const isSelected = Number(button.dataset.toolIndex) === selected;
      button.disabled = false;
      button.setAttribute("aria-pressed", String(isSelected));
    }
  };
  const controls = ["axial", "coronal", "sagittal"].map((name, axis) => {
    const canvas = document.getElementById(`ritk-snap-${name}`);
    const slider = document.getElementById(`slice-${name}`);
    const position = document.getElementById(`position-${name}`);
    const sync = () => {
      const count = Number(canvas.getAttribute("data-ritk-slice-count"));
      const index = Number(canvas.getAttribute("data-ritk-slice-index"));
      const ready = mounted && count > 0 &&
        canvas.getAttribute("data-ritk-frame-state") === "presented";
      slider.disabled = !ready;
      slider.max = String(ready ? count - 1 : 0);
      slider.value = String(ready ? index : 0);
      position.textContent = ready ? `${index + 1} / ${count}` : "No study";
      slider.setAttribute("aria-valuetext", ready ? `Slice ${index + 1} of ${count}` : "No study");
    };
    slider.addEventListener("input", () => {
      try {
        select_web_slice(axis, slider.valueAsNumber);
      } catch (error) {
        status.textContent = `Slice could not change: ${describeError(error)}`;
        sync();
      }
    });
    const syncAll = () => {
      sync();
      syncPresentation();
      syncCine();
      syncTools();
    };
    const observer = new MutationObserver(syncAll);
    observer.observe(canvas, { attributes: true, attributeFilter: [
      "data-ritk-slice-index", "data-ritk-slice-count", "data-ritk-frame-state",
      "data-ritk-load-state", "data-ritk-window-center", "data-ritk-window-width",
      "data-ritk-window-preset-index", "data-ritk-frame-generation",
      "data-ritk-cine-enabled", "data-ritk-cine-fps",
      "data-ritk-active-tool-index", "data-ritk-active-tool",
    ] });
    return { sync, observer };
  });
  const stop = () => {
    stop_web_canvas();
    mounted = false;
    controls.forEach(({ sync }) => sync());
    syncPresentation();
    syncCine();
    syncTools();
    status.textContent = "Stopped. Viewer resources released.";
  };
  const customizePicker = () => {
    const fileInput = document.getElementById("file-input");
    const fileLabel = document.querySelector('label[for="file-input"]');
    if (!(fileInput instanceof HTMLInputElement) || !(fileLabel instanceof HTMLLabelElement)) {
      throw new Error("Metis file picker controls are missing");
    }
    fileInput.accept = ".dcm,application/dicom";
    fileLabel.textContent = "Choose study files";
  };
  const mount = async () => {
    stop();
    const canvasIds = ["ritk-snap-axial", "ritk-snap-coronal", "ritk-snap-sagittal"];
    if (renderer === "webgpu") {
      await start_web_orthogonal_canvases_gpu(...canvasIds);
    } else {
      start_web_orthogonal_canvases(...canvasIds);
    }
    mounted = true;
    // Host mounting may replace the format-neutral controls on every cycle.
    // Reapply the consumer's DICOM policy after each mount.
    customizePicker();
    syncPresentation();
    syncCine();
    status.textContent = renderer === "webgpu"
      ? "Ready with WebGPU. Drop study files into the area below."
      : "Ready. Drop study files into the area below.";
  };
  await mount();
  window.metisGallery = Object.freeze({
    mount, stop,
    sample: () => ({
      mounted,
      wasm_bytes: runtime.memory.buffer.byteLength,
      host_listeners: Number(document.getElementById("metis-app")
        .getAttribute("data-metis-listener-count")),
      consumer_listeners: web_canvas_listener_count(),
      cine_enabled: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-cine-enabled"),
      cine_fps: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-cine-fps"),
      active_tool_index: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-active-tool-index"),
      active_tool: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-active-tool"),
    }),
  });
  window.addEventListener("pagehide", () => {
    stop();
    controls.forEach(({ observer }) => observer.disconnect());
  }, { once: true });
} catch (error) {
  status.textContent = `Viewer could not start: ${describeError(error)}`;
  status.setAttribute("data-state", "failed");
}
