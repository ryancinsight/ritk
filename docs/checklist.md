# Sprint Checklist — v0.50.45

Sprint: 274 — Async GPU Readback
Status: COMPLETE

---

## Sprint 274 Items

- [x] Audit `mip_pass.rs` / `vr_pass.rs` for blocking `Maintain::Wait` sites
- [x] Extend `GpuFrameCache` with `params_buf` + `lut_buf` (COPY_DST usage)
- [x] Replace `create_buffer_init` per-frame with `queue.write_buffer` into cached buffers
- [x] Split `render_mip_internal` → `submit_mip_async` + `collect_mip_result`
- [x] Split `render_vr_internal` → `submit_vr_async` + `collect_vr_result`
- [x] Add `PendingReadback` struct to `mod.rs`
- [x] Add `mip_pending` / `mip_last` / `vr_pending` / `vr_last` fields
- [x] Rewrite `render_mip` / `render_vr` with non-blocking `poll(Poll)` + async collect
- [x] Viewport resize: clear pending + last on dimension change
- [x] Add `poll_blocking()` `#[cfg(test)]` helper
- [x] Add `render_mip_sync` / `render_vr_sync` test helpers (2-round flush)
- [x] Update all existing GPU tests to use sync helpers
- [x] Add `gpu_mip_async_first_call_none_then_yields_result` contract test
- [x] Add `gpu_vr_async_first_call_none_then_yields_result` contract test
- [x] `cargo check -p ritk-snap` clean (0 errors / 0 warnings)
- [x] `cargo test -p ritk-snap -- render`: 76/76 passed (13/13 GPU, 63 other render)
- [x] Commit `2d087b4` pushed to origin/main
- [x] CHANGELOG.md, backlog.md, checklist.md created/updated

---

## Phase Exit Criteria

- [x] All selected gaps implemented and value-semantically verified
- [x] Diagnostics clean for `ritk-snap`
- [x] Artifacts synchronized

---

## Next Sprint Target: v0.50.46

Candidates (pick one):
1. GAP-262-VIZ-02 — OIT + SSAO for GPU mesh surface pipeline `[minor]`
2. GAP-262-IO-02  — C-STORE end-to-end loopback with real DICOM file `[patch]`
3. GAP-262-IO-01  — DIMSE SCU UI integration in ritk-snap `[minor]`
