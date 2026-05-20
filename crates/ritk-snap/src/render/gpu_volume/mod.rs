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
//!
//! # Rendering passes
//!
//! Internal dispatch logic lives in dedicated sub-modules:
//! - [`mip_pass`] — stateless MIP dispatch, readback, WL + colormap.
//! - [`vr_pass`]  — stateless VR dispatch, readback, RGBA conversion.
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
//! ∀ pixel p: |gpu_mip(p) − cpu_mip(p)| ≤ 2   (u8 channel value)
//! ∀ pixel p: |gpu_vr(p)  − cpu_vr(p)|  ≤ 2   (u8 channel value, premultiplied)
//! ```
//! The ±2 tolerance accounts for f32→u8 rounding and shader-side FMA vs
//! sequential CPU arithmetic.

use std::sync::Arc;

use egui::ColorImage;
use wgpu::util::DeviceExt as _;

use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;

mod context;
mod mip_pass;
mod params;
mod vr_pass;

#[cfg(test)]
#[path = "tests_gpu_volume.rs"]
mod tests;

use context::GpuContext;
use mip_pass::render_mip_internal;
use vr_pass::render_vr_internal;

/// GPU-accelerated MIP and VR volume renderer.
///
/// Owns a wgpu device, queue, and compiled compute pipelines for MIP and VR.
/// The volume buffer is lazily uploaded and cached by pointer identity and
/// shape until the volume changes.
pub struct GpuVolumeRenderer {
    ctx: GpuContext,
    mip_pipeline: wgpu::ComputePipeline,
    mip_bgl: wgpu::BindGroupLayout,
    vr_pipeline: wgpu::ComputePipeline,
    vr_bgl: wgpu::BindGroupLayout,
    /// Cached GPU storage buffer for the scalar volume (first channel only).
    vol_buffer: Option<wgpu::Buffer>,
    /// Shape `[depth, rows, cols]` of the currently cached volume.
    vol_size: Option<[usize; 3]>,
    /// `Arc` raw pointer of the volume's `data` field for change detection.
    vol_data_ptr: Option<usize>,
}

impl GpuVolumeRenderer {
    /// Attempt to create a GPU renderer.
    ///
    /// Initializes a headless wgpu context and compiles the MIP and VR compute
    /// shaders. Returns `None` when no suitable GPU is available (headless CI,
    /// VM without compute support, or driver error).
    pub fn try_create() -> Option<Self> {
        let ctx = GpuContext::try_new()?;

        // ── MIP pipeline ──────────────────────────────────────────────────────

        let mip_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_mip_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mip.wgsl").into()),
        });

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

        let mip_pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_mip_pl"),
            bind_group_layouts: &[&mip_bgl],
            push_constant_ranges: &[],
        });

        let mip_pipeline =
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu_mip_pipeline"),
                layout: Some(&mip_pl),
                module: &mip_shader,
                entry_point: "mip_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        // ── VR pipeline ───────────────────────────────────────────────────────

        let vr_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_vr_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vr.wgsl").into()),
        });

        let vr_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_vr_bgl"),
            entries: &[
                // binding 0: volume data (storage, read-only)
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
                // binding 1: output RGBA f32 buffer (storage, read-write)
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
                // binding 2: VrParams uniform
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
                // binding 3: colormap LUT (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let vr_pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_vr_pl"),
            bind_group_layouts: &[&vr_bgl],
            push_constant_ranges: &[],
        });

        let vr_pipeline =
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu_vr_pipeline"),
                layout: Some(&vr_pl),
                module: &vr_shader,
                entry_point: "vr_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Some(GpuVolumeRenderer {
            ctx,
            mip_pipeline,
            mip_bgl,
            vr_pipeline,
            vr_bgl,
            vol_buffer: None,
            vol_size: None,
            vol_data_ptr: None,
        })
    }

    /// Upload `volume` to a GPU storage buffer if the volume has changed since
    /// the last call.
    ///
    /// Change detection uses the `Arc` raw pointer of `volume.data` and the
    /// shape dimensions. If both are identical to the cached values, the
    /// upload is skipped.
    pub(super) fn ensure_volume_uploaded(&mut self, volume: &LoadedVolume) {
        let ptr = Arc::as_ptr(&volume.data) as usize;
        if self.vol_data_ptr == Some(ptr) && self.vol_size == Some(volume.shape) {
            return;
        }

        let [depth, rows, cols] = volume.shape;
        let ch = volume.channels as usize;
        let n_voxels = depth * rows * cols;

        // Extract first channel in [depth, rows, cols] row-major order.
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

    /// Render a Maximum Intensity Projection (MIP) of `volume` on the GPU.
    ///
    /// Returns `None` when no volume is loaded or a GPU error occurs.
    /// Callers must fall back to the CPU path on `None`.
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

    /// Render a Volume Rendering (VR) projection via front-to-back alpha
    /// compositing on the GPU.
    ///
    /// `alpha_scale` controls per-voxel opacity contribution. The canonical
    /// value used by the application is `0.06`.
    ///
    /// Returns `None` on GPU error; callers must fall back to the CPU path.
    pub fn render_vr(
        &mut self,
        volume: &LoadedVolume,
        wl: WindowLevel,
        colormap: Colormap,
        alpha_scale: f32,
    ) -> Option<ColorImage> {
        self.ensure_volume_uploaded(volume);
        render_vr_internal(
            &self.ctx,
            &self.vr_pipeline,
            &self.vr_bgl,
            self.vol_buffer.as_ref()?,
            volume.shape,
            wl,
            colormap,
            alpha_scale,
        )
    }
}
