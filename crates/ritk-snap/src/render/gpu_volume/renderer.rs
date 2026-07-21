//! `GpuVolumeRenderer` implementation — pipeline creation, volume upload, MIP/VR render.
//!
//! This file holds the `impl GpuVolumeRenderer` block. The struct definition,
//! `PendingReadback`, and `build_colormap_lut` live in [`super`] so that the
//! sub-modules (`mip_pass`, `vr_pass`) can reference them via `super::`.

use std::sync::Arc;

use egui::ColorImage;
use wgpu::util::DeviceExt as _;

use crate::render::{NamedColorMap, WindowLevel};
use crate::LoadedVolume;

use super::context::GpuContext;
use super::frame_cache::GpuFrameCache;
use super::mip_pass::{collect_mip_result, submit_mip_async};
use super::vr_pass::{collect_vr_result, submit_vr_async};
use super::GpuVolumeRenderer;
use super::PendingReadback;

impl GpuVolumeRenderer {
    /// Attempt to create a GPU renderer.
    ///
    /// Initializes a headless wgpu context and compiles the MIP and VR compute
    /// shaders. Returns `None` when no suitable GPU is available (headless CI,
    /// VM without compute support, or driver error).
    pub fn try_create() -> Option<Self> {
        let ctx = GpuContext::try_new()?;

        // ── MIP pipeline ──────────────────────────────────────────────────────
        let mip_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_mip_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("mip.wgsl").into()),
            });

        let mip_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let mip_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_mip_pl"),
                bind_group_layouts: &[&mip_bgl],
                push_constant_ranges: &[],
            });

        let mip_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu_mip_pipeline"),
                layout: Some(&mip_pl),
                module: &mip_shader,
                entry_point: "mip_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        // ── VR pipeline ───────────────────────────────────────────────────────
        let vr_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_vr_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("vr.wgsl").into()),
            });

        let vr_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let vr_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_vr_pl"),
                bind_group_layouts: &[&vr_bgl],
                push_constant_ranges: &[],
            });

        let vr_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
            mip_pending: None,
            mip_last: None,
            vr_pending: None,
            vr_last: None,
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
            Some(moirai::map_collect_index_with::<moirai::Adaptive, _, _>(
                n_voxels,
                |lin| *raw.get(lin * ch).unwrap_or(&0.0),
            ))
        };

        let slice: &[f32] = match extracted.as_deref() {
            Some(s) => s,
            None => &raw[..n_voxels],
        };

        let buf = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_vol_data"),
                contents: bytemuck::cast_slice(slice),
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.vol_buffer = Some(buf);
        self.vol_size = Some(volume.shape);
        self.vol_data_ptr = Some(ptr);

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
    /// Buffers are reused from `mip_cache` when dimensions are unchanged.
    ///
    /// # Async readback behaviour
    ///
    /// - Returns `None` on the first call (work submitted, no result yet).
    /// - Subsequent calls return the last completed frame while the GPU works
    ///   on the current one (1-frame display latency).
    /// - The calling thread is never blocked waiting for the GPU.
    ///
    /// Returns `None` when no volume is loaded, the GPU is unavailable, or
    /// no frame has completed yet. Callers fall back to the CPU path on `None`.
    pub fn render_mip(
        &mut self,
        volume: &LoadedVolume,
        wl: WindowLevel,
        colormap: NamedColorMap,
    ) -> Option<ColorImage> {
        self.ensure_volume_uploaded(volume);
        let [_, rows, cols] = volume.shape;

        // Viewport resize: cancel pending readback and reallocate all buffers.
        // mip_last is also cleared because the cached image has wrong dimensions.
        if self
            .mip_cache
            .as_ref()
            .is_none_or(|c| c.rows != rows || c.cols != cols)
        {
            self.mip_pending = None;
            self.mip_last = None;
            self.mip_cache = Some(GpuFrameCache::new(&self.ctx.device, rows, cols, 4));
        }

        // Non-blocking GPU poll: drives map_async callbacks for any completed work.
        self.ctx.device.poll(wgpu::Maintain::Poll);

        // Collect any completed readback.
        if let Some(pending) = self.mip_pending.take() {
            match pending.rx.try_recv() {
                Ok(Ok(())) => {
                    // GPU finished — read the mapped staging buffer.
                    let img = {
                        let cache = self.mip_cache.as_ref().unwrap();
                        collect_mip_result(&cache.staging_buf, pending.rows, pending.cols)
                    };
                    self.mip_last = Some(img);
                    // mip_pending remains None; new work submitted below.
                }
                Ok(Err(e)) => {
                    // map_async failed — log and retry on next cycle.
                    tracing::warn!(?e, "MIP map_async failed; retrying next render cycle");
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    tracing::warn!("MIP readback channel disconnected unexpectedly");
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // GPU still executing — restore pending, return cached frame.
                    self.mip_pending = Some(pending);
                    return self.mip_last.clone();
                }
            }
        }

        // Submit new GPU work (only when no readback is in-flight).
        if let Some(vol_buf) = &self.vol_buffer {
            let rx = {
                let cache = self.mip_cache.as_ref().unwrap();
                submit_mip_async(
                    &self.ctx,
                    &self.mip_pipeline,
                    &self.mip_bgl,
                    vol_buf,
                    cache,
                    volume.shape,
                    wl,
                    colormap,
                )
            };
            self.mip_pending = Some(PendingReadback { rx, rows, cols });
        }

        self.mip_last.clone()
    }

    /// Render a Volume Rendering (VR) projection via front-to-back alpha
    /// compositing on the GPU.
    ///
    /// `alpha_scale` controls per-voxel opacity contribution. The canonical
    /// value used by the application is `0.06`.
    ///
    /// # Async readback behaviour
    ///
    /// Same 1-frame latency model as `render_mip`. Returns `None` on the
    /// first call; subsequent calls return the last completed frame.
    ///
    /// Returns `None` on GPU error or before any frame completes.
    pub fn render_vr(
        &mut self,
        volume: &LoadedVolume,
        wl: WindowLevel,
        colormap: NamedColorMap,
        alpha_scale: f32,
    ) -> Option<ColorImage> {
        self.ensure_volume_uploaded(volume);
        let [_, rows, cols] = volume.shape;

        // Viewport resize: cancel pending readback and reallocate all buffers.
        if self
            .vr_cache
            .as_ref()
            .is_none_or(|c| c.rows != rows || c.cols != cols)
        {
            self.vr_pending = None;
            self.vr_last = None;
            self.vr_cache = Some(GpuFrameCache::new(&self.ctx.device, rows, cols, 4));
        }

        // Non-blocking GPU poll.
        self.ctx.device.poll(wgpu::Maintain::Poll);

        // Collect any completed readback.
        if let Some(pending) = self.vr_pending.take() {
            match pending.rx.try_recv() {
                Ok(Ok(())) => {
                    let img = {
                        let cache = self.vr_cache.as_ref().unwrap();
                        collect_vr_result(&cache.staging_buf, pending.rows, pending.cols)
                    };
                    self.vr_last = Some(img);
                }
                Ok(Err(e)) => {
                    tracing::warn!(?e, "VR map_async failed; retrying next render cycle");
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    tracing::warn!("VR readback channel disconnected unexpectedly");
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.vr_pending = Some(pending);
                    return self.vr_last.clone();
                }
            }
        }

        // Submit new GPU work.
        if let Some(vol_buf) = &self.vol_buffer {
            let rx = {
                let cache = self.vr_cache.as_ref().unwrap();
                submit_vr_async(
                    &self.ctx,
                    &self.vr_pipeline,
                    &self.vr_bgl,
                    vol_buf,
                    cache,
                    volume.shape,
                    wl,
                    colormap,
                    alpha_scale,
                )
            };
            self.vr_pending = Some(PendingReadback { rx, rows, cols });
        }

        self.vr_last.clone()
    }

    /// Block the calling thread until all in-flight GPU work completes.
    ///
    /// Drives `device.poll(Maintain::Wait)` so that any pending `map_async`
    /// callbacks fire before this function returns. After this call,
    /// `mip_pending` and `vr_pending` receivers are guaranteed to have data.
    ///
    /// # Test use only
    ///
    /// Production code must use `device.poll(Maintain::Poll)` via `render_mip`
    /// / `render_vr`. This function exists to give tests deterministic,
    /// blocking access to GPU results without spinning.
    #[cfg(test)]
    pub(super) fn poll_blocking(&mut self) {
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }
}
