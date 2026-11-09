<a id="RITK-BROWSER-TOOL-TRACE-001"></a>
## RITK-BROWSER-TOOL-TRACE-001 — Publish ROI results in the browser journey — todo
- outcome: The scheduled browser study journey publishes every ROI annotation result and the WebGPU gallery reaches Ready.
- acceptance: `metis-browser-dicom.yml` passes for chromium-window, chromium-projection-mip and chromium-webgpu at a default-branch commit. Run 42667640060 failed with `ROI Rect did not publish roi-rect annotation result`, `ROI Ellipse did not publish roi-ellipse annotation result`, and `#gallery-status did not contains 'Ready'`.
- scope: `scripts/browser_gallery_tool_trace.py`, `crates/ritk-snap/src/app/browser_tool.rs`, `crates/ritk-snap/src/browser_trace/`
- next: Reproduce the ROI publication failure against the run's bundle artifact.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-compare
- priority: verification
