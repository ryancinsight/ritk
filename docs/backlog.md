# Backlog

Sprint target version: **0.50.46** (next increment)

Triage order: correctness gaps → architecture drift → missing tests → documentation drift → cleanup.

---

## Active / Next Sprint

| ID | Priority | Class | Title | Status |
|----|----------|-------|-------|--------|
| GAP-262-VIZ-02 | High | [minor] | GPU mesh surface: OIT (depth peeling) + SSAO | OPEN |
| GAP-262-IO-02  | Medium | [patch] | C-STORE end-to-end loopback test with real DICOM fixture | OPEN |
| GAP-262-IO-01  | Medium | [minor] | Integrate DIMSE SCU into ritk-snap UI (PACS discover/find/move/store) | OPEN |

---

## Completed

| ID | Sprint | Class | Title |
|----|--------|-------|-------|
| ASYNC-GPU-274  | 274 | [minor] | Async double-buffered GPU readback (MIP + VR) — `Maintain::Wait` removed |
| DIMSE-SCU-273  | 273 | [minor] | DIMSE SCU (C-ECHO/C-FIND/C-STORE/C-MOVE) via dicom-ul |
| GPU-PERF-272   | 272 | [minor] | GPU pipeline perf: in-shader WL+LUT, packed u32 RGBA, GpuFrameCache, zero-copy upload |

---

## Residual Risk

- `ritk-python`, `ritk-core`, `ritk-vtk` have pre-existing diagnostics errors unrelated to GPU/IO work.
- `render_mip` / `render_vr` return `None` on the first call per volume/param change (1-frame latency);
  callers that previously assumed `Some` on every call must handle `None` by falling back to CPU.
- GPU async readback is single-buffered (only one `staging_buf` per pass); a second in-flight frame
  cannot be submitted until the first `map_async` completes. Double staging buffers are a follow-up.
