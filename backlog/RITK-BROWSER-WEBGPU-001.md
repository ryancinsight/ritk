<a id="RITK-BROWSER-WEBGPU-001"></a>

## RITK-BROWSER-WEBGPU-001 — Demonstrate browser WebGPU presentation — blocked
- outcome: Render the public MRI-DIR study through the RITK WebGPU canvas path.
- acceptance: The hosted browser obtains an adapter, presents all study canvases, records PNGs and semantic attributes, and tears down within the session budget.
- scope: crates/ritk-snap/src/render/gpu_volume/, browser gallery, and provenance
- next: Reopen when the hosted runner supplies a usable WebGPU adapter.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Hosted Chromium run 35759891764 reports no WebGPU adapter; a raster fallback is not proof.
- needs: none
- priority: verification
