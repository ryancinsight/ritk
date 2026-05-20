//! Stateless GPU MIP rendering pass.
//!
//! Extracted from `mod.rs` to satisfy the 500-line structural limit.
//! Called exclusively by [`super::GpuVolumeRenderer::render_mip`].
//!
//! # Algorithm
//!
//! 1. Build `RenderParams` uniform buffer (shape + WL).
//! 2. Build the 256-entry f32 RGBA colormap LUT and upload to GPU storage.
//! 3. Encode a compute pass: each thread finds the maximum intensity along
//!    the depth axis for its `(row, col)` pixel, applies the WL + LUT
//!    in-shader, and writes a packed `u32` RGBA pixel to `output_buf`.
//! 4. Copy output → staging buffer; poll device to completion.
//! 5. Reinterpret the staged bytes directly as `&[u8]` RGBA — no CPU
//!    post-processing required.
//!
//! # Output
//!
//! Both `output_buf` and `staging_buf` are pre-allocated by
//! [`GpuFrameCache::ensure`] (owned by `GpuVolumeRenderer`).
//! At 4 bytes/pixel (1 packed u32 RGBA), these buffers are the minimum size
//! required for the given `(rows, cols)` output dimensions.

use egui::ColorImage;
use wgpu::util::DeviceExt as _;

use crate::render::{Colormap, WindowLevel};

use super::build_colormap_lut;
use super::context::GpuContext;
use super::params::RenderParams;

/// Encode and submit the GPU MIP compute pass, read back packed u32 RGBA, and
/// return a [`ColorImage`].
///
/// # Parameters
///
/// - `output_buf`  — pre-allocated STORAGE | COPY_SRC buffer; size must be
///   `rows * cols * 4` bytes.
/// - `staging_buf` — pre-allocated MAP_READ | COPY_DST buffer; same size.
///
/// Returns `None` on any GPU error (buffer mapping failure, device loss).
pub(super) fn render_mip_internal(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    vol_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    staging_buf: &wgpu::Buffer,
    shape: [usize; 3],
    wl: WindowLevel,
    colormap: Colormap,
) -> Option<ColorImage> {
    let [depth, rows, cols] = shape;
    let output_bytes = rows as u64 * cols as u64 * std::mem::size_of::<u32>() as u64;

    let center = wl.center as f32;
    let width  = (wl.width as f32).max(1.0);
    let wl_lo  = center - 0.5 * width;

    let params = RenderParams {
        depth: depth as u32,
        rows:  rows as u32,
        cols:  cols as u32,
        _pad0: 0,
        wl_lo,
        wl_range: width,
        _pad2: 0.0,
        _pad3: 0.0,
    };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("gpu_mip_params"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
    });

    let lut_data = build_colormap_lut(colormap);
    let lut_buf  = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("gpu_mip_lut"),
        contents: bytemuck::cast_slice(&lut_data),
        usage:    wgpu::BufferUsages::STORAGE,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("gpu_mip_bg"),
        layout:  bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: lut_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("gpu_mip_enc") },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:             Some("gpu_mip_pass"),
            timestamp_writes:  None,
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
    // The shader wrote packed u32 RGBA via pack4x8unorm. On little-endian the
    // byte layout is [R, G, B, A] per pixel — exactly what from_rgba_unmultiplied
    // expects. No CPU conversion required.
    let rgba: Vec<u8> = mapped.to_vec();
    drop(mapped);
    staging_buf.unmap();

    Some(ColorImage::from_rgba_unmultiplied([cols, rows], &rgba))
}
