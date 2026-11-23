// Publish bounded observations through the standard Safari accessibility tree.
// All workstation input, including file access, comes from the native UI.
import {screenSamples, comparePlane} from "./workstation_pixels.js";

const surface = document.querySelector("#workstation, #ritk-snap-axial");
function report(value) {
  const label = "RITK " + JSON.stringify(value);
  if (surface.getAttribute("aria-label") !== label) surface.setAttribute("aria-label", label);
}

async function workstation() {
  let sampled = false;
  let captureError;
  let selection = [];
  let rejected = false;
  document.addEventListener("change", event => {
    if (event.isTrusted && event.target.id === "file-input") {
      rejected = false;
      selection = [event.target.files.length, Array.from(event.target.files).reduce((n,f)=>n+f.size,0)];
    }
  }, true);
  setInterval(async () => {
    const state = window.asterSlice?.state();
    if (!state) return;
    if (!sampled && state.loaded && state.indices[0] === 47) {
      sampled = true;
      try { sessionStorage.setItem("screenSamples", JSON.stringify(await screenSamples())); }
      catch (error) { captureError = String(error); }
    }
    const zone = document.querySelector("#drop-zone");
    if (zone?.dataset.dropState === "rejected" && zone.dataset.byteState === "failed" && !rejected) {
      const pixels = await screenSamples();
      rejected = JSON.stringify(pixels) === sessionStorage.getItem("screenSamples");
    }
    report({...state, selection, rejected, captured: !!sessionStorage.getItem("screenSamples"),
      overflow: document.documentElement.scrollWidth > innerWidth,
      error: captureError || document.querySelector("#notice")?.textContent || ""});
  }, 175);
}

async function embedded() {
  const picker = document.createElement("button");
  picker.textContent = "Open reference study";
  picker.onclick = () => document.querySelector("#file-input").click();
  document.body.prepend(picker);
  const samples = JSON.parse(sessionStorage.getItem("screenSamples"));
  const oracle = await (await fetch("oracle.json")).json();
  const wasm = await import("./consumer/ritk_snap.js");
  let compared = false;
  const tick = setInterval(async () => {
    if (compared || document.querySelectorAll("canvas[data-ritk-load-state=ready]").length !== 3) return;
    compared = true;
    clearInterval(tick);
    try {
      for (const [index,name] of ["axial","coronal","sagittal"].entries()) {
        const [matches, hash] = await comparePlane(name, samples[index]);
        if (!matches || hash !== oracle[name]) throw Error(name + " reference mismatch: " + hash);
      }
      const before = wasm.web_canvas_listener_count();
      wasm.stop_web_canvas();
      const after = wasm.web_canvas_listener_count();
      report({planes: 3, before, after});
    } catch (error) { report({error:String(error)}); }
  }, 100);
}

if (document.getElementById("workstation")) workstation();
else embedded().catch(error => report({error:String(error)}));
