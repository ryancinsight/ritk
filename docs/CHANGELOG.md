# Changelog

All notable changes follow [SemVer 2.0.0](https://semver.org/spec/v2.0.0.html).
Change classes: `[patch]` `[minor]` `[major]` `[arch]`.

---

## [Unreleased]

## [0.50.45] — 2026-05-19

### Added
- **Sprint 274 — Async GPU Readback** `[minor]`
  - `GpuFrameCache` extended with `params_buf` (UNIFORM | COPY_DST, 32 B) and
    `lut_buf` (STORAGE | COPY_DST, 4 096 B); both updated via `queue.write_buffer`
    each frame — zero GPU buffer allocations per frame after first call.
  - `mip_pass`: `render_mip_internal` (blocking `Maintain::Wait`) replaced by
    `submit_mip_async` (non-blocking, returns `mpsc::Receiver`) +
    `collect_mip_result` (reads mapped staging buffer).
  - `vr_pass`: identical split — `render_vr_internal` → `submit_vr_async` +
    `collect_vr_result`.
  - `GpuVolumeRenderer`: `PendingReadback` struct added; `mip_pending`,
    `mip_last`, `vr_pending`, `vr_last` fields track in-flight state.
  - `render_mip` / `render_vr` now call `device.poll(Maintain::Poll)` (non-blocking),
    collect completed readbacks, submit new work, and return the last completed
    frame — the render thread is never blocked waiting for the GPU.
  - Viewport resize clears pending readback and cached image (wrong dimensions).
  - `poll_blocking()` helper (`#[cfg(test)]`) added for deterministic test drains.
  - New tests: `gpu_mip_async_first_call_none_then_yields_result`,
    `gpu_vr_async_first_call_none_then_yields_result` verify the non-blocking
    contract (Invariant: first call returns `None`; after `poll_blocking`, second
    call returns `Some`).  Total GPU tests: 13/13.

## [0.50.44] — Sprint 273

### Added
- **Sprint 273 — DIMSE SCU networking module** `[minor]`
  - `crates/ritk-io/src/format/dicom/networking`: C-ECHO, C-FIND, C-STORE, C-MOVE SCU.
  - `dicom-ul = "0.8"` workspace dependency.
  - 24 deterministic DIMSE tests (8 unit + 3 loopback integration).

## [0.50.43] — Sprint 272

### Changed
- **Sprint 272 — GPU pipeline performance** `[minor]`
  - MIP shader: WL normalisation + LUT applied in-shader; output packed `u32` RGBA.
  - VR shader: output packed `u32` RGBA; staging buffer 4× smaller.
  - `RenderParams` extended with `wl_lo`, `wl_range`; `VrParams` added.
  - `GpuFrameCache` introduced for output + staging buffer reuse across frames.
  - Zero-copy single-channel volume upload; Rayon parallel multi-channel extraction.
  - CPU post-processing loop eliminated for both MIP and VR.
