//! GPU-accelerated volume MIP and VR rendering via wgpu compute shaders.
//!
//! # Architecture
//!
//! `GpuVolumeRenderer` is the single public entry point. It owns:
//! - A headless `GpuContext` (wgpu device + queue).
//! - Compiled compute pipelines for Maximum Intensity Projection (`mip.wgsl`)
//!   and Volume Rendering (`vr.wgsl`, front-to-back alpha compositing).
//! - A cached volume buffer: the `Arc<Vec<f32>>` pointer is compared across
//!   calls so the volume is re-uploaded only when it changes.
//! - Per-pass `GpuFrameCache` holding pre-allocated output and staging GPU
//!   buffers, reused across frames whenever output dimensions are stable.
//!
//! # Rendering passes
//!
//! Internal dispatch logic lives in dedicated sub-modules:
//! - `mip_pass` â€” MIP dispatch; WL+LUT applied in-shader; packed u32 RGBA output.
//! - `vr_pass` â€” VR dispatch; packed u32 RGBA output.
//!
//! # Performance
//!
//! - **MIP**: WL normalisation and colormap LUT applied on GPU; zero CPU
//!   post-processing after readback.
//! - **VR**: output packed as u32 RGBA (4 bytes/pixel, vs 16 bytes/pixel
//!   previously); staging buffer 4Ã— smaller; zero CPU conversion.
//! - **Buffer caching**: output and staging buffers are reused across frames
//!   while dimensions are constant (typical medical imaging: fixed viewport).
//! - **Volume upload**: single-channel volumes bypass extraction with a
//!   zero-copy `Arc<Vec<f32>>` slice reference. Multi-channel volumes use
//!   Rayon parallel extraction of the first channel.
//!
//! # Fallback contract
//!
//! `try_create()` returns `None` when no GPU is available. All callers must
//! fall back to the CPU path in that case.
//!
//! # Platform gate
//!
//! This module is compiled only on non-wasm32 targets. The wasm32 target
//! uses the CPU renderer exclusively.
//!
//! # Differential equivalence invariants
//!
//! For all valid inputs and the Grayscale colormap:
//! ```text
//! âˆ€ pixel p: |gpu_mip(p) âˆ’ cpu_mip(p)| â‰¤ 2 (u8 channel value)
//! âˆ€ pixel p: |gpu_vr(p) âˆ’ cpu_vr(p)| â‰¤ 2 (u8 channel value, premultiplied)
//! ```
//! The Â±2 tolerance accounts for f32â†’u8 rounding (LUT truncation vs round)
//! and pack4x8unorm rounding vs CPU truncation.

use crate::render::Colormap;

pub(crate) mod context;
mod frame_cache;
mod mip_pass;
mod params;
mod renderer;
mod vr_pass;

#[cfg(test)]
#[path = "tests_gpu_volume.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_gpu_volume_render.rs"]
mod tests_gpu_volume_render;

use context::GpuContext;
use frame_cache::GpuFrameCache;

/// Build a 256-entry f32 RGBA colormap LUT for GPU upload.
///
/// Entry `i` maps `i / 255.0` through `colormap.map()` and stores the result
/// as `[R, G, B, 1.0]` in f32 âˆˆ [0, 1]. Stored flat as 256 Ã— 4 = 1 024 f32
/// values for direct upload to the `lut` storage buffer.
///
/// # Invariant
///
/// `lut[i * 4 + 0]` = `colormap.map(i as f32 / 255.0)[0] as f32 / 255.0`
/// `lut[i * 4 + 1]` = `colormap.map(i as f32 / 255.0)[1] as f32 / 255.0`
/// `lut[i * 4 + 2]` = `colormap.map(i as f32 / 255.0)[2] as f32 / 255.0`
/// `lut[i * 4 + 3]` = `1.0` (unused alpha slot for alignment)
fn build_colormap_lut(colormap: Colormap) -> Vec<f32> {
    let mut lut = Vec::with_capacity(256 * 4);
    for i in 0u32..256 {
        let norm = i as f32 / super::U8_MAX_F32;
        let [r, g, b] = colormap.map(norm);
        lut.push(r as f32 / super::U8_MAX_F32);
        lut.push(g as f32 / super::U8_MAX_F32);
        lut.push(b as f32 / super::U8_MAX_F32);
        lut.push(1.0f32);
    }
    lut
}

/// In-flight GPU readback awaiting `map_async` completion.
///
/// Produced by `submit_mip_async` / `submit_vr_async`; consumed by the next
/// `render_mip` / `render_vr` call once `device.poll(Poll)` drives the
/// callback to fire.
pub(in crate::render::gpu_volume) struct PendingReadback {
    /// Receiver fired by the `map_async` callback.
    ///
    /// - `Ok(Ok(()))` â€” GPU done; staging buffer is mapped and safe to read.
    /// - `Ok(Err(_))` â€” `map_async` failed (device loss, OOM); retry next cycle.
    /// - `Err(Disconnected)` â€” internal error; treat as `map_async` failure.
    pub(in crate::render::gpu_volume) rx:
        std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    /// Expected output dimensions; validated against the collected image.
    pub(in crate::render::gpu_volume) rows: usize,
    pub(in crate::render::gpu_volume) cols: usize }

/// GPU-accelerated MIP and VR volume renderer with non-blocking async readback.
///
/// Owns a wgpu device, queue, compiled compute pipelines for MIP and VR, and
/// per-pass frame caches that hold pre-allocated GPU output, staging, params,
/// and LUT buffers reused across frames while output dimensions are stable.
///
/// # Async readback model
///
/// Each `render_mip` / `render_vr` call:
/// 1. Calls `device.poll(Poll)` â€” non-blocking; fires any completed callbacks.
/// 2. If a pending readback is complete, collects the result into `*_last`.
/// 3. If no readback is in-flight, submits new GPU work and registers
///    `map_async` (stores `PendingReadback`).
/// 4. Returns the last completed frame â€” `None` only on the first call before
///    any frame has been rendered.
///
/// This decouples GPU execution from CPU readback: the render thread is never
/// blocked waiting for the GPU. One-frame display latency is acceptable for
/// interactive medical visualization.
pub struct GpuVolumeRenderer {
    pub(in crate::render::gpu_volume) ctx: GpuContext,
    pub(in crate::render::gpu_volume) mip_pipeline: wgpu::ComputePipeline,
    pub(in crate::render::gpu_volume) mip_bgl: wgpu::BindGroupLayout,
    pub(in crate::render::gpu_volume) vr_pipeline: wgpu::ComputePipeline,
    pub(in crate::render::gpu_volume) vr_bgl: wgpu::BindGroupLayout,
    /// Cached GPU storage buffer for the scalar volume (first channel only).
    pub(in crate::render::gpu_volume) vol_buffer: Option<wgpu::Buffer>,
    /// Shape `[depth, rows, cols]` of the currently cached volume.
    pub(in crate::render::gpu_volume) vol_size: Option<[usize; 3]>,
    /// `Arc` raw pointer of the volume's `data` field for change detection.
    pub(in crate::render::gpu_volume) vol_data_ptr: Option<usize>,
    /// Cached output, staging, params, and LUT buffers for the MIP pass.
    pub(in crate::render::gpu_volume) mip_cache: Option<GpuFrameCache>,
    /// Cached output, staging, params, and LUT buffers for the VR pass.
    pub(in crate::render::gpu_volume) vr_cache: Option<GpuFrameCache>,
    /// In-flight MIP readback awaiting GPU completion.
    pub(in crate::render::gpu_volume) mip_pending: Option<PendingReadback>,
    /// Last successfully collected MIP frame; `None` before any frame completes.
    pub(in crate::render::gpu_volume) mip_last: Option<egui::ColorImage>,
    /// In-flight VR readback awaiting GPU completion.
    pub(in crate::render::gpu_volume) vr_pending: Option<PendingReadback>,
    /// Last successfully collected VR frame; `None` before any frame completes.
    pub(in crate::render::gpu_volume) vr_last: Option<egui::ColorImage> }
