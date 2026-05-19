//! GPU-accelerated volume MIP rendering via wgpu compute shaders.
//!
//! # Architecture
//!
//! `GpuVolumeRenderer` is the single entry point.  It owns:
//! - A headless `GpuContext` (wgpu device + queue).
//! - A compiled compute pipeline for Maximum Intensity Projection (MIP).
//! - A cached volume buffer: the `Arc<Vec<f32>>` pointer is compared across
//!   calls so the volume is re-uploaded only when it changes.
//!
//! # Fallback contract
//!
//! `try_create()` returns `None` when no GPU is available.  All callers must
//! fall back to the CPU path in that case.
//!
//! # Platform gate
//!
//! This entire module is compiled only on non-wasm32 targets.  The wasm32
//! target continues to use the CPU renderer.
//!
//! # Differential equivalence invariant
//!
//! For all valid inputs and the Grayscale colormap:
//! ```text
//! ∀ pixel p: |gpu_mip(p) − cpu_mip(p)| ≤ 2   (u8 channel value)
//! ```
//! The ±2 tolerance accounts for f32→u8 rounding differences between the
//! GPU shader (which outputs raw f32 max values) and the CPU path.

use std::sync::Arc;
use egui::ColorImage;
use wgpu::util::DeviceExt as _;

use crate::LoadedVolume;
use crate::render::{Colormap, WindowLevel};

mod context;
mod params;

#[cfg(test)]
#[path = "tests_gpu_volume.rs"]
mod tests;

use context::GpuContext;
use params::RenderParams;

/// GPU-accelerated MIP/VR volume renderer.
///
/// Owns a wgpu device, queue, and compute pipeline.  The volume buffer is
/// lazily uploaded and cached until the volume pointer changes.
pub struct GpuVolumeRenderer {
    ctx: GpuContext,
    mip_pipeline: wgpu::ComputePipeline,
    mip_bgl: wgpu::BindGroupLayout,
    /// Cached GPU buffer holding the scalar volume data.
    vol_buffer: Option<wgpu::Buffer>,
    /// Shape of the currently cached volume `[depth, rows, cols]`.
    vol_size: Option<[usize; 3]>,
    /// Raw pointer to the `Vec<f32>` payload for change detection.
    vol_data_ptr: Option<usize>,
}

impl GpuVolumeRenderer {
    /// Attempt to create a GPU renderer.
    ///
    /// Initializes a wgpu headless context and compiles the MIP compute shader.
    /// Returns `None` when no suitable GPU is found.
    pub fn try_create() -> Option<Self> {
        let ctx = GpuContext::try_new()?;

        // Compile MIP compute shader.
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_mip_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mip.wgsl").into()),
        });

        // Bind group layout: volume (storage, read), output (storage, rw), params (uniform).
        let mip_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_mip_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout =
            ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_mip_pl"),
                bind_group_layouts: &[&mip_bgl],
                push_constant_ranges: &[],
            });

        let mip_pipeline =
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu_mip_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "mip_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Some(GpuVolumeRenderer {
            ctx,
            mip_pipeline,
            mip_bgl,
            vol_buffer: None,
            vol_size: None,
            vol_data_ptr: None,
        })
    }

    /// Upload `volume` to a GPU storage buffer if the volume has changed since
    /// the last call.
    ///
    /// Change detection uses the `Arc` raw pointer of `volume.data` and the
    /// shape dimensions.  If both are identical to the cached values, the
    /// upload is skipped.
    pub(super) fn ensure_volume_uploaded(&mut self, volume: &LoadedVolume) {
        let ptr = Arc::as_ptr(&volume.data) as usize;
        if self.vol_data_ptr == Some(ptr) && self.vol_size == Some(volume.shape) {
            return;
        }

        let [depth, rows, cols] = volume.shape;
        let ch = volume.channels as usize;
        let n_voxels = depth * rows * cols;

        // Extract first channel in [depth, rows, cols] order.
        let mut data: Vec<f32> = Vec::with_capacity(n_voxels);
        let raw = &*volume.data;
        for d in 0..depth {
            for r in 0..rows {
                for c in 0..cols {
                    let idx = ((d * rows + r) * cols + c) * ch;
                    data.push(raw.get(idx).copied().unwrap_or(0.0));
                }
            }
        }

        let buf = self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_vol_data"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        self.vol_buffer = Some(buf);
        self.vol_size = Some(volume.shape);
        self.vol_data_ptr = Some(ptr);

        tracing::debug!(
            depth,
            rows,
            cols,
            bytes = data.len() * 4,
            "Volume uploaded to GPU"
        );
    }

    /// Render a Maximum Intensity Projection (MIP) of the current volume on
    /// the GPU and return it as an [`egui::ColorImage`].
    ///
    /// # Algorithm
    ///
    /// 1. Upload volume to GPU storage buffer (cached; only re-uploaded on change).
    /// 2. Dispatch compute shader: each thread finds `max_val` along the depth axis.
    /// 3. Read back `Vec<f32>` (one max value per pixel) via a staging buffer.
    /// 4. Apply window/level normalisation and colormap on CPU.
    ///
    /// # Returns
    ///
    /// `None` when no volume is loaded or a GPU error occurs (caller must fall
    /// back to CPU path).
    pub fn render_mip(
        &mut self,
        volume: &LoadedVolume,
        wl: WindowLevel,
        colormap: Colormap,
    ) -> Option<ColorImage> {
        self.ensure_volume_uploaded(volume);
        render_mip_internal(
            &self.ctx,
            &self.mip_pipeline,
            &self.mip_bgl,
            self.vol_buffer.as_ref()?,
            volume.shape,
            wl,
            colormap,
        )
    }
}

/// Inner stateless MIP render function.
///
/// Separated from `GpuVolumeRenderer` methods so that tests can call it
/// without borrowing the renderer mutably twice.
fn render_mip_internal(
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

    // --- Params uniform -------------------------------------------------------
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

    // --- Output storage buffer ------------------------------------------------
    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_mip_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- Staging buffer for readback -----------------------------------------
    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_mip_staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Bind group -----------------------------------------------------------
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_mip_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    });

    // --- Encode and dispatch --------------------------------------------------
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

    // --- Readback (synchronous) -----------------------------------------------
    let buffer_slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;

    let mapped = buffer_slice.get_mapped_range();
    let raw_floats: &[f32] = bytemuck::cast_slice(&mapped);

    // --- WL normalisation + colormap (CPU) ------------------------------------
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
