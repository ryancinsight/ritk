<a id="RITK-BROWSER-TOOL-TRACE-001"></a>
## RITK-BROWSER-TOOL-TRACE-001 — Publish ROI results in the browser journey — todo
- outcome: The scheduled browser study journey publishes every ROI annotation result and the WebGPU workstation displays the study.
- acceptance: `metis-browser-dicom.yml` passes for chromium-window, chromium-projection-mip and chromium-webgpu at a default-branch commit. Run 42667640060 failed with `ROI Rect did not publish roi-rect annotation result`, `ROI Ellipse did not publish roi-ellipse annotation result`, and `#gallery-status did not contains 'Ready'`. Pan, Zoom, W/L, Crosshair, Label Paint, Label Erase, Closed Polygon, Open Polygon, Arrow, Pencil and Cobb Angle retain input-only browser coverage until each publishes a completed-tool behavior oracle.
- scope: `scripts/browser_workspace.py`, `crates/ritk-snap/src/app/browser_tool.rs`, `crates/ritk-snap/src/browser_trace/`
- next: Reproduce the ROI publication failure against the run's bundle artifact.
- basis: 30b8ba098
- needs: none
- priority: verification; milestone: M-viewer-compare
- [patch] On a local Chromium GPU adapter, the saved MRI passes all 50 slice actions and restores the initial RGBA hashes on all three planes. The upstream rejection-order correction is in [Métis PR 588](https://github.com/ryancinsight/metis/pull/588).
