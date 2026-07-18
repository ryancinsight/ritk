//! Stateless GPU VR rendering pass â€” non-blocking async readback.
//!
//! Mirrors the protocol in [`super::mip_pass`]:
//!
//! 1. [`submit_vr_async`] â€” encodes the VR compute dispatch (front-to-back
//!    alpha compositing), copies output â†’ staging, submits, and registers
//!    a non-blocking `map_async`.
//!
//! 2. [`collect_vr_result`] â€” reads the mapped staging buffer and returns a
//!    [`ColorImage`].  Must only be called after the receiver fires `Ok(Ok(()))`.
//!
//! # Memory efficiency
//!
//! The VR shader emits one packed u32 per pixel (4 bytes) via `pack4x8unorm`,
//! giving 4Ã— smaller staging buffers than the previous 4Ã—f32 layout and
//! eliminating the O(n_pixels) CPU conversion loop.
//!
//! # Per-frame zero-allocation
//!
//! `params_buf` and `lut_buf` are updated via `queue.write_buffer`; only
//! the bind group is allocated per render call.

use std::sync::mpsc;

use egui::ColorImage;

use crate::render::{Colormap, WindowLevel};

use super::build_colormap_lut;
use super::context::GpuContext;
use super::frame_cache::GpuFrameCache;
use super::params::VrParams;

/// Encode and submit the VR compute pass without blocking the calling thread.
///
/// Updates `cache.params_buf` (VrParams) and `cache.lut_buf` via
/// `queue.write_buffer`, encodes the front-to-back compositing dispatch +
/// outputâ†’staging copy, submits, and registers `map_async` on
/// `cache.staging_buf`.
///
/// # Returns
///
/// An `mpsc::Receiver` that fires:
/// - `Ok(Ok(()))` â€” GPU finished; `cache.staging_buf` is mapped and readable
///   via [`collect_vr_result`].
/// - `Ok(Err(_))` â€” `map_async` failed; caller retries on next render cycle.
/// - `Err(Disconnected)` â€” internal error; treat as failure.
///
/// # Non-blocking guarantee
///
/// This function does not call `device.poll`.
pub(super) fn submit_vr_async(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    vol_buf: &wgpu::Buffer,
    cache: &GpuFrameCache,
    vol_shape: [usize; 3],
    wl: WindowLevel,
    colormap: Colormap,
    alpha_scale: f32,
) -> mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let [depth, rows, cols] = vol_shape;
    // Output: 1 packed u32 per pixel = 4 bytes (was 4Ã—f32 = 16 bytes).
    let output_bytes = rows as u64 * cols as u64 * std::mem::size_of::<u32>() as u64;

    let center = wl.center as f32;
    let width = (wl.width as f32).max(1.0);
    let wl_lo = center - 0.5 * width;

    let params = VrParams {
        depth: depth as u32,
        rows: rows as u32,
        cols: cols as u32,
        _pad0: 0,
        wl_lo,
        wl_range: width,
        alpha_scale,
        _pad1: 0.0 };

    // Update cached uniform and LUT buffers without re-allocation.
    ctx.queue
        .write_buffer(&cache.params_buf, 0, bytemuck::bytes_of(&params));
    let lut_data = build_colormap_lut(colormap);
    ctx.queue
        .write_buffer(&cache.lut_buf, 0, bytemuck::cast_slice(&lut_data));

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_vr_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cache.output_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cache.params_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cache.lut_buf.as_entire_binding() },
        ] });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_vr_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_vr_pass"),
            timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg_x = (cols as u32).div_ceil(8);
        let wg_y = (rows as u32).div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&cache.output_buf, 0, &cache.staging_buf, 0, output_bytes);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = cache.staging_buf.slice(..);
    let (tx, rx) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    rx
}

/// Read a completed VR readback from the already-mapped staging buffer.
///
/// # Precondition
///
/// The receiver returned by [`submit_vr_async`] must have fired `Ok(Ok(()))`.
/// Calling before the receiver fires causes `get_mapped_range` to panic.
///
/// # Output
///
/// `pack4x8unorm` writes `[R, G, B, A]` bytes on little-endian; forwarded
/// directly to `from_rgba_unmultiplied` without CPU conversion.
pub(super) fn collect_vr_result(
    staging_buf: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> ColorImage {
    let buffer_slice = staging_buf.slice(..);
    let mapped = buffer_slice.get_mapped_range();
    let rgba: Vec<u8> = mapped.to_vec();
    drop(mapped);
    staging_buf.unmap();
    ColorImage::from_rgba_unmultiplied([cols, rows], &rgba)
}
