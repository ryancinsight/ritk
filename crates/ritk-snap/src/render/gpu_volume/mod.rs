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
//! - [`mip_pass`] — MIP dispatch; WL+LUT applied in-shader; packed u32 RGBA output.
//! - [`vr_pass`]  — VR dispatch; packed u32 RGBA output.
//!
//! # Performance
//!
//! - **MIP**: WL normalisation and colormap LUT applied on GPU; zero CPU
//!   post-processing after readback.
//! - **VR**: output packed as u32 RGBA (4 bytes/pixel, vs 16 bytes/pixel
//!   previously); staging buffer 4× smaller; zero CPU conversion.
//! - **Buffer caching**: output and staging buffers are reused across frames
//!   while dimensions are constant (typical medical imaging: fixed viewport).
//! - **Volume upload**: single-channel volumes bypass extraction with a
//!   zero-copy `Arc<Vec<f32>>` slice reference.  Multi-channel volumes use
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
//! ∀ pixel p: |gpu_mip(p) − cpu_mip(p)| ≤ 2   (u8 channel value)
//! ∀ pixel p: |gpu_vr(p)  − cpu_vr(p)|  ≤ 2   (u8 channel value, premultiplied)
//! ```
//! The ±2 tolerance accounts for f32→u8 rounding (LUT truncation vs round)
//! and pack4x8unorm rounding vs CPU truncation.

use std::sync::Arc;

use egui::ColorImage;
use rayon::prelude::*;
use wgpu::util::DeviceExt as _;

use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;

mod context;
mod frame_cache;
mod mip_pass;
mod params;
mod vr_pass;

#[cfg(test)]
#[path = "tests_gpu_volume.rs"]
mod tests;

use context::GpuContext;
use frame_cache::GpuFrameCache;
use mip_pass::render_mip_internal;
use vr_pass::render_vr_internal;

/// Build a 256-entry f32 RGBA colormap LUT for GPU upload.
///
/// Entry `i` maps `i / 255.0` through `colormap.map()` and stores the result
/// as `[R, G, B, 1.0]` in f32 ∈ [0, 1].  Stored flat as 256 × 4 = 1 024 f32
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
        let norm = i as f32 / 255.0;
        let [r, g, b] = colormap.map(norm);
        lut.push(r as f32 / 255.0);
        lut.push(g as f32 / 255.0);
        lut.push(b as f32 / 255.0);
        lut.push(1.0f32);
    }
    lut
}

/// GPU-accelerated MIP and VR volume renderer.
///
/// Owns a wgpu device, queue, compiled compute pipelines for MIP and VR, and
/// per-pass frame caches that hold pre-allocated GPU output and staging buffers
/// reused across frames while output dimensions are stable.
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
    /// Cached output + staging buffers for the MIP pass (reused while cols/rows stable).
    mip_cache: Option<GpuFrameCache>,
    /// Cached output + staging buffers for the VR pass (reused while cols/rows stable).
    vr_cache: Option<GpuFrameCache>,
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
                // binding 1: mip_out (storage, read-write, packed u32 RGBA)
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
                // binding 2: RenderParams uniform (32 bytes: shape + WL)
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
                // binding 3: colormap LUT (storage, read-only, 256 × 4 f32)
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
                // binding 1: output packed u32 RGBA buffer (storage, read-write)
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
            mip_cache: None,
            vr_cache: None,
        })
    }

    /// Upload `volume` to a GPU storage buffer if the volume has changed since
    /// the last call.
    ///
    /// Change detection uses the `Arc` raw pointer of `volume.data` and the
    /// shape dimensions. If both are identical to the cached values, the
    /// upload is skipped.
    ///
    /// # Zero-copy single-channel path
    ///
    /// When `volume.channels == 1` the raw `Arc<Vec<f32>>` slice is cast
    /// directly to bytes without any CPU extraction loop.
    ///
    /// # Multi-channel path
    ///
    /// When `volume.channels > 1` the first channel is extracted in parallel
    /// using Rayon before uploading.
    pub(super) fn ensure_volume_uploaded(&mut self, volume: &LoadedVolume) {
        let ptr = Arc::as_ptr(&volume.data) as usize;
        if self.vol_data_ptr == Some(ptr) && self.vol_size == Some(volume.shape) {
            return;
        }

        let [depth, rows, cols] = volume.shape;
        let ch = volume.channels as usize;
        let n_voxels = depth * rows * cols;
        let raw = &*volume.data;

        // Zero-copy path for single-channel volumes: the data is already in
        // [depth, rows, cols] row-major order — no extraction needed.
        let extracted: Option<Vec<f32>> = if ch == 1 && raw.len() >= n_voxels {
            None
        } else {
            // Multi-channel: extract first channel in parallel with Rayon.
            // Voxel at linear index `lin` has first-channel value at raw[lin * ch].
            Some(
                (0..n_voxels)
                    .into_par_iter()
                    .map(|lin| *raw.get(lin * ch).unwrap_or(&0.0))
                    .collect(),
            )
        };
        let slice: &[f32] = match extracted.as_deref() {
            Some(s) => s,
            None    => &raw[..n_voxels],
        };

        let buf = self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_vol_data"),
            contents: bytemuck::cast_slice(slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        self.vol_buffer    = Some(buf);
        self.vol_size      = Some(volume.shape);
        self.vol_data_ptr  = Some(ptr);

        tracing::debug!(
            depth,
            rows,
            cols,
            ch,
            bytes = n_voxels * 4,
            "Volume uploaded to GPU"
        );
    }

    /// Render a Maximum Intensity Projection (MIP) of `volume` on the GPU.
    ///
    /// WL normalisation and colormap are applied entirely on the GPU.
    /// Output and staging buffers are reused from `mip_cache` when dimensions
    /// are unchanged.
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
        let [_, rows, cols] = volume.shape;

        // Ensure cached frame buffers are allocated for this output size.
        // MIP: 1 packed u32 per pixel = 4 bytes per pixel.
        if self.mip_cache.as_ref().map_or(true, |c| c.rows != rows || c.cols != cols) {
            self.mip_cache = Some(GpuFrameCache::new(&self.ctx.device, rows, cols, 4));
        }
        let cache = self.mip_cache.as_ref().unwrap();

        render_mip_internal(
            &self.ctx,
            &self.mip_pipeline,
            &self.mip_bgl,
            self.vol_buffer.as_ref()?,
            &cache.output_buf,
            &cache.staging_buf,
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
    /// Output and staging buffers are reused from `vr_cache` when dimensions
    /// are unchanged; the staging buffer is 4× smaller than the previous
    /// f32-based layout.
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
        let [_, rows, cols] = volume.shape;

        // Ensure cached frame buffers are allocated for this output size.
        // VR: 1 packed u32 per pixel = 4 bytes per pixel.
        if self.vr_cache.as_ref().map_or(true, |c| c.rows != rows || c.cols != cols) {
            self.vr_cache = Some(GpuFrameCache::new(&self.ctx.device, rows, cols, 4));
        }
        let cache = self.vr_cache.as_ref().unwrap();

        render_vr_internal(
            &self.ctx,
            &self.vr_pipeline,
            &self.vr_bgl,
            self.vol_buffer.as_ref()?,
            &cache.output_buf,
            &cache.staging_buf,
            volume.shape,
            wl,
            colormap,
            alpha_scale,
        )
    }
}
