//! Stateless GPU VR rendering pass and colormap LUT construction.
//!
//! Extracted from `mod.rs` to satisfy the 500-line structural limit.
//! Called exclusively by [`super::GpuVolumeRenderer::render_vr`].
//!
//! # Algorithm
//!
//! 1. Build `VrParams` uniform (shape + WL window + alpha_scale).
//! 2. Build a 256-entry f32 RGBA colormap LUT and upload to GPU storage.
//! 3. Allocate output storage buffer (`n_pixels × 4 × f32`).
//! 4. Allocate staging buffer for CPU readback.
//! 5. Dispatch the VR compute shader: per-pixel front-to-back compositing
//!    with early exit at alpha ≥ 0.99.
//! 6. Copy output → staging and poll device to completion.
//! 7. Convert f32 RGBA per pixel to u8 RGBA for egui.

use egui::ColorImage;
use wgpu::util::DeviceExt as _;

use crate::render::{Colormap, WindowLevel};

use super::context::GpuContext;
use super::params::VrParams;

/// Build a 256-entry f32 RGBA colormap LUT for GPU upload.
///
/// Entry `i` maps `i / 255.0` through `colormap.map()` and stores the result
/// as `[R, G, B, 1.0]` in f32 ∈ [0, 1]. Stored flat as 256 × 4 = 1024
/// f32 values for direct upload to the `lut` storage buffer in `vr.wgsl`.
///
/// # Invariant
///
/// `lut[i * 4 + 0]` = `colormap.map(i as f32 / 255.0)[0] / 255.0`
/// `lut[i * 4 + 1]` = `colormap.map(i as f32 / 255.0)[1] / 255.0`
/// `lut[i * 4 + 2]` = `colormap.map(i as f32 / 255.0)[2] / 255.0`
/// `lut[i * 4 + 3]` = `1.0` (unused alpha slot for alignment)
pub(super) fn build_colormap_lut(colormap: Colormap) -> Vec<f32> {
    let mut lut = Vec::with_capacity(256 * 4);
    for i in 0u32..256 {
        let norm = i as f32 / 255.0;
        let [r, g, b] = colormap.map(norm);
        lut.push(r as f32 / 255.0);
        lut.push(g as f32 / 255.0);
        lut.push(b as f32 / 255.0);
        lut.push(1.0f32);
    }
    lut
}

/// Dispatch the GPU VR shader, read back f32 RGBA per pixel, and convert to u8.
///
/// Returns `None` on any GPU error (buffer mapping failure, device loss).
pub(super) fn render_vr_internal(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    vol_buf: &wgpu::Buffer,
    shape: [usize; 3],
    wl: WindowLevel,
    colormap: Colormap,
    alpha_scale: f32,
) -> Option<ColorImage> {
    let [depth, rows, cols] = shape;
    let n_pixels = rows * cols;
    // Output: 4 f32 per pixel (R, G, B, A).
    let output_bytes = (n_pixels * 4 * std::mem::size_of::<f32>()) as u64;

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
        _pad1: 0.0,
    };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_vr_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let lut_data = build_colormap_lut(colormap);
    let lut_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_vr_lut"),
        contents: bytemuck::cast_slice(&lut_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_vr_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_vr_staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_vr_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: lut_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gpu_vr_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_vr_pass"),
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

    // Convert per-pixel f32 RGBA ∈ [0, 1] to u8 RGBA.
    let mut rgba = Vec::with_capacity(n_pixels * 4);
    for i in 0..n_pixels {
        let base = i * 4;
        rgba.push((raw_floats[base].clamp(0.0, 1.0) * 255.0) as u8);
        rgba.push((raw_floats[base + 1].clamp(0.0, 1.0) * 255.0) as u8);
        rgba.push((raw_floats[base + 2].clamp(0.0, 1.0) * 255.0) as u8);
        rgba.push((raw_floats[base + 3].clamp(0.0, 1.0) * 255.0) as u8);
    }
    drop(mapped);
    staging_buf.unmap();

    Some(ColorImage::from_rgba_unmultiplied([cols, rows], &rgba))
}
