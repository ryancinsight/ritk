// Publish bounded observations through the standard Safari accessibility tree.
// All workstation input, including file access, comes from the native UI.
import {displayedPlanes, referenceDigest} from "./workstation_pixels.js";

const surface = document.querySelector("#workstation, #ritk-snap-axial");
function report(value) {
  const label = "RITK " + JSON.stringify(value);
  if (surface.getAttribute("aria-label") !== label) surface.setAttribute("aria-label", label);
}

async function workstation() {
  if (document.title !== "AsterSlice") { report({error: "unexpected workstation title"}); return; }
  // Use normal same-tab navigation to retain only this test's reference samples.
  const link = document.createElement("a");
  const gallery = location.pathname.endsWith("gallery.html");
  link.href = gallery ? "embedded/embed.html" : "gallery.html?projection=mip";
  link.textContent = gallery ? "Open independent reference" : "Open projection entry";
  link.className = "qualification-link";
  document.body.append(link);
  let sampled = false;
  let heldCursorSeen = false;
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
      try { sessionStorage.setItem("displayedPlanes", JSON.stringify(await displayedPlanes())); }
      catch (error) {
        const canvas = document.querySelector("#workstation");
        const context = canvas.getContext("2d");
        captureError = {message:String(error), call:String(error.stack).split("\n")[0],
          extent:[canvas.width, canvas.height], lost:context?.isContextLost?.(),
          notice:document.getElementById("notice").textContent};
      }
    }
    if (captureError) { report({error:captureError}); return; }
    const zone = document.querySelector("#drop-zone");
    if (zone?.dataset.dropState === "rejected" && zone.dataset.byteState === "failed" && !rejected) {
      const pixels = await displayedPlanes();
      rejected = JSON.stringify(pixels) === sessionStorage.getItem("displayedPlanes");
    }
    // Safari bounds accessibility labels. Expose only the values XCTest asserts.
    const {width, panes, counts, indices, tool, window: levels, annotations, measurement, zoom,
      orientationIdentity, cine, series, panel, panels, projection} = state;
    let marks;
    if (panels.length === 2 && panels.some(([, count]) => count === 3) &&
        panels.some(([, count]) => count === 5)) {
      const ctx = document.querySelector("#workstation").getContext("2d");
      for (const [i,[x,y,w,h]] of state.images.entries()) {
        if (panels[i][1] !== 3) continue;
        const [r,g,b] = ctx.getImageData(Math.round(x+w*.4),Math.round(y+h*.5),1,1).data;
        if (r===0 && g===229 && b===255) heldCursorSeen = true;
      }
      marks = state.images.map(([x,y,w,h]) => [[.4,.5],[.4,.625],[.75,.4],[.375,.4]].map(([u,v]) => {
        const [r,g,b,a] = ctx.getImageData(Math.round(x + u*w), Math.round(y + v*h), 1, 1).data;
        return r === 255 && g === 255 && b === 0 && a === 255;
      }));
    }
    report({width, panes, counts, indices, tool, window: levels, annotations, measurement, zoom, marks,
      orientationIdentity, cine, series, panel, panels, projection, selection, rejected,
      images: location.search === "?position" ? state.images : undefined, heldCursorSeen,
      captured: !!sessionStorage.getItem("displayedPlanes"), overflow: document.documentElement.scrollWidth > innerWidth});
  }, 175);
}

async function embedded() {
  const picker = document.createElement("button");
  picker.textContent = "Open reference study";
  picker.onclick = () => document.querySelector("#file-input").click();
  document.body.prepend(picker);
  const displayed = JSON.parse(sessionStorage.getItem("displayedPlanes"));
  const oracle = await (await fetch("oracle.json")).json();
  let selected;
  document.addEventListener("change", event => {
    if (event.isTrusted && event.target.id === "file-input") {
      const files = Array.from(event.target.files);
      selected = [files.length, files.reduce((total, file) => total + file.size, 0)];
    }
  }, true);
  let compared = false;
  const tick = setInterval(async () => {
    if (!selected || compared || document.querySelectorAll("canvas[data-ritk-load-state=ready]").length !== 3) return;
    compared = true;
    clearInterval(tick);
    try {
      if (selected[0] !== 94 || selected[1] !== 49807236) {
        throw Error("independent reference selection: " + JSON.stringify(selected));
      }
      for (const name of ["axial","coronal","sagittal"]) {
        const hash = await referenceDigest(name);
        if (hash !== oracle[name]) throw Error(name + " reference mismatch: " + hash);
      }
      const [before, after] = window.stopStudyReference();
      report({planes: 3, before, after, displayed});
    } catch (error) { report({error:String(error)}); }
  }, 100);
}

if (document.getElementById("workstation")) workstation();
else embedded().catch(error => report({error:String(error)}));
