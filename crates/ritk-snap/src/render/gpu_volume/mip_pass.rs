//! Stateless GPU MIP rendering pass.
//!
//! Extracted from `mod.rs` to satisfy the 500-line structural limit.
//! Called exclusively by [`super::GpuVolumeRenderer::render_mip`].
//!
//! # Algorithm
//!
//! 1. Build `RenderParams` uniform buffer (depth, rows, cols).
//! 2. Allocate output storage buffer (`n_pixels × f32`).
//! 3. Allocate staging buffer for CPU readback.
//! 4. Dispatch the MIP compute shader: each thread finds the maximum intensity
//!    along the depth axis for its `(row, col)` output pixel.
//! 5. Copy output → staging and poll device to completion.
//! 6. Apply WL normalisation and colormap on CPU; emit `ColorImage`.

use egui::ColorImage;
use wgpu::util::DeviceExt as _;

use crate::render::{Colormap, WindowLevel};

use super::context::GpuContext;
use super::params::RenderParams;

/// Dispatch the GPU MIP shader, read back results, and apply WL + colormap.
///
/// Returns `None` on any GPU error (buffer mapping failure, device loss).
pub(super) fn render_mip_internal(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    vol_buf: &wgpu::Buffer,
    shape: [usize; 3],
    wl: WindowLevel,
    colormap: Colormap,
) -> Option<ColorImage> {
    let [depth, rows, cols] = shape;
    let n_pixels = rows * cols;
    let output_bytes = (n_pixels * std::mem::size_of::<f32>()) as u64;

    let params = RenderParams {
        depth: depth as u32,
        rows: rows as u32,
        cols: cols as u32,
        _pad: 0,
    };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_mip_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_mip_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_mip_staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_mip_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gpu_mip_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_mip_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg_x = (cols as u32 + 7) / 8;
        let wg_y = (rows as u32 + 7) / 8;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_bytes);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;

    let mapped = buffer_slice.get_mapped_range();
    let raw_floats: &[f32] = bytemuck::cast_slice(&mapped);

    // WL normalisation + colormap on CPU.
    let center = wl.center as f32;
    let width = (wl.width as f32).max(1.0);
    let lo = center - 0.5 * width;
    let mut rgba = Vec::with_capacity(n_pixels * 4);
    for &v in raw_floats {
        let norm = ((v - lo) / width).clamp(0.0, 1.0);
        let [r, g, b] = colormap.map(norm);
        rgba.push(r);
        rgba.push(g);
        rgba.push(b);
        rgba.push(255u8);
    }
    drop(mapped);
    staging_buf.unmap();

    Some(ColorImage::from_rgba_unmultiplied([cols, rows], &rgba))
}
