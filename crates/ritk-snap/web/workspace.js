// Browser chrome reads the presented frame. Decoding, tools and image navigation stay in RITK.
export function createWorkspace(chooseTool) {
  const byId = (id) => document.getElementById(id);
  const panes = Array.from(document.querySelectorAll(".gallery-views figure"));
  const hasProjection = new URLSearchParams(location.search).has("projection");
  const preview = byId("series-bar");
  const seriesToggle = byId("series-toggle");
  const compact = matchMedia("(max-width: 639px), (max-height: 479px)");
  const showSeries = (show) => {
    preview.hidden = !show;
    seriesToggle.setAttribute("aria-pressed", String(show));
  };
  showSeries(!compact.matches);
  compact.addEventListener("change", () => showSeries(!compact.matches));
  seriesToggle.onclick = () => showSeries(preview.hidden);
  for (const button of document.querySelectorAll("[data-open-study]")) {
    button.onclick = () => {
      byId("file-menu").hidePopover();
      byId("file-input").click();
    };
  }

  let focus = "all";
  const setFocus = (name) => {
    focus = name;
    for (const pane of panes) {
      const role = pane.dataset.pane ?? "projection";
      pane.hidden = (name !== "all" && role !== name) ||
        (role === "projection" && !hasProjection);
      const button = pane.querySelector("[data-focus]");
      if (button) {
        button.setAttribute("aria-pressed", String(name === role));
        button.setAttribute("aria-label", name === role ? "Restore all panes" : 'Maximize ' + role + ' pane');
      }
    }
    byId("pane-status").textContent = name === "all"
      ? panes.filter((pane) => !pane.hidden).length + " panes" : name[0].toUpperCase() + name.slice(1);
  };
  const responsive = document.body.classList.contains("responsive-layout");
  if (responsive) {
    // The responsive Rust presenter owns track visibility and placement.
    for (const button of document.querySelectorAll('[popovertarget="view-menu"]')) button.hidden = true;
    byId("pane-status").textContent = "Responsive layout";
  } else {
    setFocus("all");
    for (const button of document.querySelectorAll("[data-focus]")) {
      button.onclick = () => setFocus(focus === button.dataset.focus ? "all" : button.dataset.focus);
    }
    for (const button of document.querySelectorAll("[data-view]")) {
      button.onclick = () => { setFocus(button.dataset.view); byId("view-menu").hidePopover(); };
    }
  }

  const quickTools = byId("quick-tools");
  const palette = byId("tool-buttons");
  const quickButtons = ["W/L", "Pan", "Zoom", "Length", "Angle"].map((name) => {
    const button = document.createElement("button");
    button.textContent = name;
    button.dataset.tool = name;
    button.disabled = true;
    button.onclick = () => chooseTool(Number(button.dataset.toolIndex));
    quickTools.append(button);
    return button;
  });
  const syncTools = () => {
    for (const button of quickButtons) {
      const source = palette.querySelector(`[data-tool="${button.dataset.tool}"]`);
      if (source) button.dataset.toolIndex = source.dataset.toolIndex;
      button.disabled = !source || source.disabled;
      button.setAttribute("aria-pressed", source?.getAttribute("aria-pressed") ?? "false");
    }
  };
  palette.addEventListener("click", () => byId("tools").hidePopover());
  const thumbnail = byId("series-thumb");
  const thumbnailContext = thumbnail.getContext("2d");
  const fitFrame = (canvas) => {
    const aspect = canvas.getAttribute("data-ritk-display-aspect");
    if (aspect) canvas.closest(".image-field")?.style.setProperty("--physical-aspect", aspect);
    return aspect;
  };
  const projection = byId("ritk-snap-projection");
  new MutationObserver(() => fitFrame(projection)).observe(projection, {
    attributes: true, attributeFilter: ["data-ritk-display-aspect"],
  });
  const syncFrame = (canvas, ready) => {
    const aspect = fitFrame(canvas);
    if (canvas.id !== "ritk-snap-axial") return;
    byId("series-card").hidden = !ready;
    byId("image-status").textContent = ready
      ? `W ${canvas.dataset.ritkWindowWidth}   C ${canvas.dataset.ritkWindowCenter}` : "";
    if (!ready) return;
    const count = canvas.getAttribute("data-ritk-slice-count");
    byId("series-size").textContent = count + (count === "1" ? " image" : " images");
    // Fit the physical frame ratio into the square thumbnail, including anisotropic images.
    const ratio = Number(aspect ?? 1);
    const w = thumbnail.width * Math.min(1, ratio);
    const h = thumbnail.height * Math.min(1, 1 / ratio);
    thumbnailContext.clearRect(0, 0, thumbnail.width, thumbnail.height);
    thumbnailContext.drawImage(canvas, (thumbnail.width - w) / 2, (thumbnail.height - h) / 2, w, h);
  };
  const mountPicker = () => {
    const fileInput = document.getElementById("file-input");
    const fileLabel = document.querySelector('label[for="file-input"]');
    if (!(fileInput instanceof HTMLInputElement) || !(fileLabel instanceof HTMLLabelElement)) {
      throw new Error("Metis file picker controls are missing");
    }
    fileInput.accept = ".dcm,application/dicom";
    fileLabel.textContent = "Choose study files";

    for (const button of document.querySelectorAll("[data-open-study]")) button.disabled = false;
    byId("drop-zone").querySelector("p").textContent = "Drop DICOM study files here";
    // The native file input stays keyboard accessible through the Open Study button.
    fileInput.tabIndex = -1;
  };
  const stop = () => {
    syncTools();
    for (const button of document.querySelectorAll("[data-open-study]")) button.disabled = true;
  };
  return { syncFrame, syncTools, mountPicker, stop };
}
