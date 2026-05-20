//! Stateless GPU VR rendering pass and colormap LUT construction.
//!
//! Extracted from `mod.rs` to satisfy the 500-line structural limit.
//! Called exclusively by [`super::GpuVolumeRenderer::render_vr`].
//!
//! # Algorithm
//!
//! 1. Build `VrParams` uniform (shape + WL window + alpha_scale).
//! 2. Upload the 256-entry f32 RGBA colormap LUT to GPU storage.
//! 3. Encode a compute pass: per-pixel front-to-back alpha compositing with
//!    early exit at alpha ≥ 0.99.  The shader packs the composited RGBA into
//!    one `u32` via `pack4x8unorm` — 4 bytes per pixel.
//! 4. Copy output → staging buffer; poll device to completion.
//! 5. Reinterpret staged bytes as `&[u8]` RGBA — no CPU conversion required.
//!
//! # Memory efficiency
//!
//! The previous implementation stored 4 × f32 per pixel (16 bytes), requiring
//! a CPU conversion pass to u8.  The packed u32 layout reduces the staging
//! buffer by 4× and eliminates the O(n_pixels) CPU conversion loop.
//!
//! # Output buffers
//!
//! Both `output_buf` and `staging_buf` are pre-allocated by
//! [`GpuFrameCache::ensure`] (owned by `GpuVolumeRenderer`), sized at exactly
//! `rows * cols * 4` bytes.

use egui::ColorImage;
use wgpu::util::DeviceExt as _;

use crate::render::{Colormap, WindowLevel};

use super::context::GpuContext;
use super::params::VrParams;

/// Encode and submit the GPU VR compute pass, read back packed u32 RGBA, and
/// return a [`ColorImage`].
///
/// # Parameters
///
/// - `output_buf`  — pre-allocated STORAGE | COPY_SRC buffer;
///   size must be `rows * cols * 4` bytes.
/// - `staging_buf` — pre-allocated MAP_READ | COPY_DST buffer; same size.
///
/// Returns `None` on any GPU error (buffer mapping failure, device loss).
pub(super) fn render_vr_internal(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    vol_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    staging_buf: &wgpu::Buffer,
    shape: [usize; 3],
    wl: WindowLevel,
    colormap: Colormap,
    alpha_scale: f32,
) -> Option<ColorImage> {
    let [depth, rows, cols] = shape;
    // Output: 1 packed u32 per pixel = 4 bytes (was 4×f32 = 16 bytes per pixel).
    let output_bytes = rows as u64 * cols as u64 * std::mem::size_of::<u32>() as u64;

    let center = wl.center as f32;
    let width  = (wl.width as f32).max(1.0);
    let wl_lo  = center - 0.5 * width;

    let params = VrParams {
        depth: depth as u32,
        rows:  rows as u32,
        cols:  cols as u32,
        _pad0: 0,
        wl_lo,
        wl_range: width,
        alpha_scale,
        _pad1: 0.0,
    };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("gpu_vr_params"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
    });

    let lut_data = super::build_colormap_lut(colormap);
    let lut_buf  = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("gpu_vr_lut"),
        contents: bytemuck::cast_slice(&lut_data),
        usage:    wgpu::BufferUsages::STORAGE,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("gpu_vr_bg"),
        layout:  bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: lut_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("gpu_vr_enc") },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("gpu_vr_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg_x = (cols as u32 + 7) / 8;
        let wg_y = (rows as u32 + 7) / 8;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(output_buf, 0, staging_buf, 0, output_bytes);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;

    let mapped = buffer_slice.get_mapped_range();
    // The shader packed composited RGBA via pack4x8unorm.  On little-endian the
    // byte layout in memory is [R, G, B, A] per pixel.  Forward directly to
    // from_rgba_unmultiplied — zero CPU conversion work.
    let rgba: Vec<u8> = mapped.to_vec();
    drop(mapped);
    staging_buf.unmap();

    Some(ColorImage::from_rgba_unmultiplied([cols, rows], &rgba))
}
