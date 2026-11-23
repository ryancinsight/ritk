<a id="RITK-BROWSER-WEBGPU-001"></a>
## RITK-BROWSER-WEBGPU-001 — Verify browser WebGPU presentation on main [patch] — done
- priority: verification; milestone: M-viewer-compare
- outcome: Render the public MRI-DIR study through the RITK WebGPU canvas path.
- acceptance: The hosted browser obtains an adapter, presents all study canvases, records PNGs and semantic attributes, and tears down within the session budget.
- scope: crates/ritk-snap/src/render/gpu_volume/, browser gallery, and provenance
- [x] [patch] [Chromium WebGPU on merged main `8550491eb`](https://github.com/ryancinsight/ritk/actions/runs/45111738624/job/137486577964) obtained a real adapter, presented the study, matched all three reference planes and completed teardown on hosted macOS.
- basis: 54140426be0173d8aa8f937bc49c6255647231b4
- evidence: [Hosted Chromium WebGPU](https://github.com/ryancinsight/ritk/actions/runs/45079835823/job/137389340243) obtained an adapter on macOS and passed the 94-file study, the three-plane cine-rate trace and session cleanup. The run retains 40 PNG captures.
- needs: none
