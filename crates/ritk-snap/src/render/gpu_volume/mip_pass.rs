//! Stateless GPU MIP rendering pass — non-blocking async readback.
//!
//! # Protocol
//!
//! GPU work is decoupled from CPU readback via two functions:
//!
//! 1. [`submit_mip_async`] — encodes and submits the MIP compute pass, then
//!    registers a non-blocking `map_async` on `cache.staging_buf`.  Returns
//!    an `mpsc::Receiver` that fires when the GPU has finished and the staged
//!    bytes are ready to read.
//!
//! 2. [`collect_mip_result`] — reads the already-mapped `staging_buf` and
//!    returns a [`ColorImage`].  Must only be called after the receiver from
//!    `submit_mip_async` fires `Ok(Ok(()))`.
//!
//! # Non-blocking contract
//!
//! `submit_mip_async` does NOT call `device.poll` or block the calling thread.
//! The caller drives completion by calling `device.poll(wgpu::Maintain::Poll)`
//! and checking the receiver with `try_recv()`.  Blocking is only acceptable
//! in tests via `device.poll(wgpu::Maintain::Wait)`.
//!
//! # Per-frame zero-allocation
//!
//! `params_buf` and `lut_buf` live in [`GpuFrameCache`] and are updated
//! via `queue.write_buffer` each frame — no `create_buffer_init` per call.
//! `output_buf` and `staging_buf` are also cached; the bind group is the
//! only per-frame GPU object created.

use std::sync::mpsc;

use egui::ColorImage;

use crate::render::{Colormap, WindowLevel};

use super::build_colormap_lut;
use super::context::GpuContext;
use super::frame_cache::GpuFrameCache;
use super::params::RenderParams;

/// Encode and submit the MIP compute pass without blocking the calling thread.
///
/// Updates `cache.params_buf` and `cache.lut_buf` via `queue.write_buffer`
/// (zero new GPU buffer allocation), then encodes the compute dispatch +
/// output→staging copy, submits, and registers `map_async` on `cache.staging_buf`.
///
/// # Returns
///
/// An `mpsc::Receiver` that fires:
/// - `Ok(Ok(()))` — GPU finished; `cache.staging_buf` is mapped and readable
///   via [`collect_mip_result`].
/// - `Ok(Err(_))` — `map_async` failed (device loss, OOM); caller should retry
///   on the next render cycle.
/// - `Err(Disconnected)` — internal error; treat as `map_async` failure.
///
/// # Non-blocking guarantee
///
/// This function does not call `device.poll`.  The caller must drive GPU
/// completion externally via `device.poll(wgpu::Maintain::Poll)`.
pub(super) fn submit_mip_async(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    vol_buf: &wgpu::Buffer,
    cache: &GpuFrameCache,
    vol_shape: [usize; 3],
    wl: WindowLevel,
    colormap: Colormap,
) -> mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let [depth, rows, cols] = vol_shape;
    let output_bytes = rows as u64 * cols as u64 * std::mem::size_of::<u32>() as u64;

    let center = wl.center as f32;
    let width  = (wl.width as f32).max(1.0);
    let wl_lo  = center - 0.5 * width;

    let params = RenderParams {
        depth:    depth as u32,
        rows:     rows as u32,
        cols:     cols as u32,
        _pad0:    0,
        wl_lo,
        wl_range: width,
        _pad2:    0.0,
        _pad3:    0.0,
    };

    // Update cached uniform and LUT buffers without re-allocation.
    // write_buffer operations are ordered before the subsequent submit call.
    ctx.queue.write_buffer(&cache.params_buf, 0, bytemuck::bytes_of(&params));
    let lut_data = build_colormap_lut(colormap);
    ctx.queue.write_buffer(&cache.lut_buf, 0, bytemuck::cast_slice(&lut_data));

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("gpu_mip_bg"),
        layout:  bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cache.output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: cache.params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cache.lut_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("gpu_mip_enc") },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("gpu_mip_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg_x = (cols as u32).div_ceil(8);
        let wg_y = (rows as u32).div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&cache.output_buf, 0, &cache.staging_buf, 0, output_bytes);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    // Register the async readback.  The callback fires after device.poll
    // drives the GPU to completion; no blocking here.
    let buffer_slice = cache.staging_buf.slice(..);
    let (tx, rx) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    rx
}

/// Read a completed MIP readback from the already-mapped staging buffer.
///
/// Creates a new `BufferSlice` from `staging_buf` (the buffer holds mapping
/// state, not the slice), reads packed u32 RGBA bytes written by the shader,
/// and returns a [`ColorImage`].
///
/// # Precondition
///
/// The receiver returned by [`submit_mip_async`] must have fired `Ok(Ok(()))`:
/// the GPU has completed and `staging_buf` is mapped.  Calling this before the
/// receiver fires causes `get_mapped_range` to panic.
///
/// # Output
///
/// `pack4x8unorm` writes `[R, G, B, A]` bytes on little-endian; the byte
/// layout is forwarded directly to `from_rgba_unmultiplied` — no CPU
/// conversion required.
pub(super) fn collect_mip_result(
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
