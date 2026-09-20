// RITK owns the viewer loop and consumes Metis's bounded byte handoff.
const status = document.getElementById("gallery-status");
const describeError = (error) => error instanceof Error ? error.message : String(error);
try {
  const { default: init, start_web_orthogonal_canvases,
    start_web_orthogonal_canvases_gpu,
    start_web_orthogonal_canvases_with_projection,
    start_web_orthogonal_canvases_gpu_with_projection,
    stop_web_canvas, web_canvas_listener_count,
    select_web_slice, set_web_cine_rate, set_web_window_preset,
    toggle_web_cine, toggle_web_crosshair, select_web_tool, web_tool_count, web_tool_name,
    web_window_preset_count, web_window_preset_name } =
    await import("./consumer/ritk_snap.js");
  const runtime = await init();
  const query = new URLSearchParams(window.location.search);
  const renderer = query.get("renderer") === "webgpu"
    ? "webgpu" : "raster";
  const projectionModes = Object.freeze({ mip: 0, minip: 1, average: 2 });
  const projectionMode = query.get("projection");
  if (projectionMode !== null && !Object.hasOwn(projectionModes, projectionMode)) {
    throw new Error(`unsupported projection mode: ${projectionMode}`);
  }
  const projectionIndex = projectionMode === null ? null : projectionModes[projectionMode];
  const projectionFigure = document.getElementById("projection-view");
  const projectionStatistic = document.getElementById("projection-statistic");
  if (!(projectionFigure instanceof HTMLElement) ||
      !(projectionStatistic instanceof HTMLOutputElement)) {
    throw new Error("projection presentation controls are missing");
  }
  projectionFigure.hidden = projectionMode === null;
  projectionStatistic.textContent = projectionMode === null
    ? "No projection"
    : projectionMode === "mip" ? "MIP" : projectionMode === "minip" ? "MinIP" : "Average";
  let mounted = false;
  const windowPreset = document.getElementById("window-preset");
  const windowLevel = document.getElementById("window-level");
  const cineToggle = document.getElementById("cine-toggle");
  const cineRate = document.getElementById("cine-rate");
  const cineRateValue = document.getElementById("cine-rate-value");
  const crosshairToggle = document.getElementById("crosshair-toggle");
  const crosshairState = document.getElementById("crosshair-state");
  const toolButtons = document.getElementById("tool-buttons");
  const activeTool = document.getElementById("active-tool");
  if (!(windowPreset instanceof HTMLSelectElement) ||
      !(windowLevel instanceof HTMLOutputElement) ||
      !(cineToggle instanceof HTMLButtonElement) ||
      !(cineRate instanceof HTMLInputElement) ||
      !(cineRateValue instanceof HTMLOutputElement) ||
      !(crosshairToggle instanceof HTMLButtonElement) ||
      !(crosshairState instanceof HTMLOutputElement) ||
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
  const crosshairCanvases = ["axial", "coronal", "sagittal"]
    .map((name) => document.getElementById(`ritk-snap-${name}`));
  const crosshairCoordinate = (canvas, voxel) => {
    const axis = Number(canvas.getAttribute("data-ritk-axis"));
    const width = Number(canvas.getAttribute("data-ritk-frame-width"));
    const height = Number(canvas.getAttribute("data-ritk-frame-height"));
    const rotation = Number(canvas.getAttribute("data-ritk-view-rotation"));
    const flipH = canvas.getAttribute("data-ritk-view-flip-h") === "true";
    const flipV = canvas.getAttribute("data-ritk-view-flip-v") === "true";
    if (![0, 1, 2].includes(axis) || ![0, 90, 180, 270].includes(rotation) ||
        !Number.isSafeInteger(width) || width < 1 ||
        !Number.isSafeInteger(height) || height < 1) return null;
    const sourceWidth = rotation === 90 || rotation === 270 ? height : width;
    const sourceHeight = rotation === 90 || rotation === 270 ? width : height;
    const row = axis === 0 ? voxel[1] : voxel[0];
    const col = axis === 2 ? voxel[1] : voxel[2];
    if (row < 0 || row >= sourceHeight || col < 0 || col >= sourceWidth) return null;
    let x = col + 0.5;
    let y = row + 0.5;
    if (flipH) x = sourceWidth - x;
    if (flipV) y = sourceHeight - y;
    [x, y] = rotation === 0 ? [x, y]
      : rotation === 90 ? [sourceHeight - y, x]
      : rotation === 180 ? [sourceWidth - x, sourceHeight - y]
      : [y, sourceWidth - x];
    const outputWidth = rotation === 90 || rotation === 270 ? sourceHeight : sourceWidth;
    const outputHeight = rotation === 90 || rotation === 270 ? sourceWidth : sourceHeight;
    return {
      left: Math.max(0, Math.min(100, 100 * x / outputWidth)),
      top: Math.max(0, Math.min(100, 100 * y / outputHeight)),
    };
  };
  const syncCrosshair = () => {
    const ready = mounted && crosshairCanvases.every((canvas) =>
      canvas instanceof HTMLCanvasElement &&
      canvas.getAttribute("data-ritk-load-state") === "ready" &&
      canvas.getAttribute("data-ritk-frame-state") === "presented");
    if (!ready) {
      crosshairToggle.disabled = true;
      crosshairToggle.textContent = "Show crosshair";
      crosshairToggle.setAttribute("aria-pressed", "false");
      crosshairState.textContent = "No study";
      crosshairCanvases.forEach((canvas) => {
        const view = canvas.parentElement;
        view?.querySelectorAll(".crosshair-line").forEach((line) => { line.style.display = "none"; });
      });
      return;
    }
    const canvas = crosshairCanvases[0];
    const visible = canvas.getAttribute("data-ritk-crosshair-visible") === "true";
    const rawCursor = canvas.getAttribute("data-ritk-linked-cursor") ?? "";
    const voxel = rawCursor.split(",").map(Number);
    const validVoxel = voxel.length === 3 && voxel.every((value) => Number.isSafeInteger(value) && value >= 0);
    crosshairToggle.disabled = false;
    crosshairToggle.textContent = visible ? "Hide crosshair" : "Show crosshair";
    crosshairToggle.setAttribute("aria-pressed", String(visible));
    crosshairState.textContent = validVoxel
      ? `${visible ? "Crosshair visible" : "Crosshair hidden"} at ${voxel.join(",")}`
      : "Crosshair unavailable";
    crosshairCanvases.forEach((candidate) => {
      const view = candidate.parentElement;
      const lines = view?.querySelectorAll(".crosshair-line");
      const point = visible && validVoxel ? crosshairCoordinate(candidate, voxel) : null;
      if (!lines || lines.length !== 2 || !point) {
        lines?.forEach((line) => { line.style.display = "none"; });
        return;
      }
      const row = view.querySelector(".crosshair-row");
      const column = view.querySelector(".crosshair-column");
      if (!(row instanceof HTMLElement) || !(column instanceof HTMLElement)) return;
      row.style.top = `${point.top}%`;
      column.style.left = `${point.left}%`;
      row.style.display = "block";
      column.style.display = "block";
    });
  };
  crosshairToggle.addEventListener("click", () => {
    try {
      toggle_web_crosshair();
      syncCrosshair();
    } catch (error) {
      status.textContent = `Crosshair could not change: ${describeError(error)}`;
      syncCrosshair();
    }
  });
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
      syncCrosshair();
      syncTools();
    };
    const observer = new MutationObserver(syncAll);
    observer.observe(canvas, { attributes: true, attributeFilter: [
      "data-ritk-slice-index", "data-ritk-slice-count", "data-ritk-frame-state",
      "data-ritk-load-state", "data-ritk-window-center", "data-ritk-window-width",
      "data-ritk-window-preset-index", "data-ritk-frame-generation",
      "data-ritk-cine-enabled", "data-ritk-cine-fps",
      "data-ritk-active-tool-index", "data-ritk-active-tool",
      "data-ritk-crosshair-visible", "data-ritk-linked-cursor",
      "data-ritk-view-flip-h", "data-ritk-view-flip-v", "data-ritk-view-rotation",
    ] });
    return { sync, observer };
  });
  const stop = () => {
    stop_web_canvas();
    mounted = false;
    controls.forEach(({ sync }) => sync());
    syncPresentation();
    syncCine();
    syncCrosshair();
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
    const orthogonalCanvasIds = ["ritk-snap-axial", "ritk-snap-coronal", "ritk-snap-sagittal"];
    if (projectionIndex === null && renderer === "webgpu") {
      await start_web_orthogonal_canvases_gpu(...orthogonalCanvasIds);
    } else if (projectionIndex === null) {
      start_web_orthogonal_canvases(...orthogonalCanvasIds);
    } else if (renderer === "webgpu") {
      await start_web_orthogonal_canvases_gpu_with_projection(
        orthogonalCanvasIds[0], orthogonalCanvasIds[1], orthogonalCanvasIds[2],
        "ritk-snap-projection", projectionIndex);
    } else {
      start_web_orthogonal_canvases_with_projection(
        orthogonalCanvasIds[0], orthogonalCanvasIds[1], orthogonalCanvasIds[2],
        "ritk-snap-projection", projectionIndex);
    }
    mounted = true;
    // Host mounting may replace the format-neutral controls on every cycle.
    // Reapply the consumer's DICOM policy after each mount.
    customizePicker();
    syncPresentation();
    syncCine();
    syncCrosshair();
    const rendererLabel = renderer === "webgpu" ? " with WebGPU" : "";
    const projectionLabel = projectionMode === null ? "" : ` and ${projectionStatistic.textContent} projection`;
    status.textContent = `Ready${rendererLabel}${projectionLabel}. Drop study files into the area below.`;
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
      crosshair_visible: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-crosshair-visible"),
      linked_cursor: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-linked-cursor"),
      active_tool_index: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-active-tool-index"),
      active_tool: document.getElementById("ritk-snap-axial")
        ?.getAttribute("data-ritk-active-tool"),
      projection_mode: projectionMode,
      projection_statistic: document.getElementById("ritk-snap-projection")
        ?.getAttribute("data-ritk-projection-statistic"),
      projection_frame_state: document.getElementById("ritk-snap-projection")
        ?.getAttribute("data-ritk-frame-state"),
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
