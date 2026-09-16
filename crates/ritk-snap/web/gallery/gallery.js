// RITK owns the viewer loop and consumes Metis's bounded byte handoff.
const status = document.getElementById("gallery-status");
try {
  const { default: init, start_web_orthogonal_canvases, stop_web_canvas,
    web_canvas_listener_count, select_web_slice } =
    await import("./consumer/ritk_snap.js");
  const runtime = await init();
  let mounted = false;
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
        status.textContent = `Slice could not change: ${error.message || error}`;
        sync();
      }
    });
    const observer = new MutationObserver(sync);
    observer.observe(canvas, { attributes: true, attributeFilter: [
      "data-ritk-slice-index", "data-ritk-slice-count", "data-ritk-frame-state",
    ] });
    return { sync, observer };
  });
  const stop = () => {
    stop_web_canvas();
    mounted = false;
    controls.forEach(({ sync }) => sync());
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
  const mount = () => {
    stop();
    start_web_orthogonal_canvases(
      "ritk-snap-axial", "ritk-snap-coronal", "ritk-snap-sagittal",
    );
    mounted = true;
    // Host mounting may replace the format-neutral controls on every cycle.
    // Reapply the consumer's DICOM policy after each mount.
    customizePicker();
    status.textContent = "Ready. Drop study files into the area below.";
  };
  mount();
  window.metisGallery = Object.freeze({
    mount, stop,
    sample: () => ({
      mounted,
      wasm_bytes: runtime.memory.buffer.byteLength,
      host_listeners: Number(document.getElementById("metis-app")
        .getAttribute("data-metis-listener-count")),
      consumer_listeners: web_canvas_listener_count(),
    }),
  });
  window.addEventListener("pagehide", () => {
    stop();
    controls.forEach(({ observer }) => observer.disconnect());
  }, { once: true });
} catch (error) {
  status.textContent = `Viewer could not start: ${error.message}`;
  status.setAttribute("data-state", "failed");
}
