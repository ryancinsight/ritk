//! GPU-accelerated mesh surface renderer with OIT depth peeling and SSAO.
//!
//! # Architecture
//!
//! `GpuMeshRenderer` is the public entry point.  It owns:
//! - A headless `GpuMeshContext` (wgpu device + queue + compiled pipelines).
//! - A `GpuMeshFrameCache` holding per-viewport textures and buffers.
//! - A `GpuMeshBufs` holding the current mesh's vertex/index GPU buffers.
//!
//! # Rendering pipeline
//!
//! 1. **Depth peeling** (`N_PEEL_LAYERS` passes): extracts the N closest transparent
//!    surfaces in front-to-back order. Pass 0 also captures the G-buffer
//!    (view-space normal + linear depth) for SSAO.
//! 2. **SSAO** compute pass: hemisphere sampling against the G-buffer produces
//!    a per-pixel ambient-occlusion factor (`ao_buf`).
//! 3. **Composite** compute pass: back-to-front Porter-Duff "over" blend of all
//!    N color layers; SSAO applied to layer 0; result packed as u32 RGBA.
//! 4. **Async readback**: staging buffer `map_async` + `mpsc::Receiver`; no
//!    blocking on the render thread.
//!
//! # Async contract (same as `GpuVolumeRenderer`)
//!
//! - First `render()` call submits GPU work and returns `None`.
//! - Subsequent calls poll for completion; return the last completed frame while
//!   new GPU work is submitted for the current inputs.
//!
//! # Fallback
//!
//! `try_create()` returns `None` when no suitable GPU is available.
//! Callers must fall back to `MeshRenderer` (CPU).
//!
//! # Platform gate
//!
//! Compiled only on non-wasm32 targets (same gate as `gpu_volume`).

mod context;
mod frame_cache;
mod mesh_buf;
mod params;
mod passes;

use egui::ColorImage;
use ritk_io::VtkPolyData;

use crate::render::mesh_render::{DirectionalLight, MeshCamera, PhongMaterial};
use crate::render::mesh_render::{look_at, mat4_mul, normalize, perspective};

use context::GpuMeshContext;
use frame_cache::{GpuMeshFrameCache, N_PEEL_LAYERS};
use mesh_buf::GpuMeshBufs;
use params::{LightBlock, LightUniform, MaterialUniforms, SceneUniforms, SsaoUniforms};
use passes::{collect_mesh_result, submit_mesh_async};

#[cfg(test)]
#[path = "tests_gpu_mesh.rs"]
mod tests;

// ── Configuration ─────────────────────────────────────────────────────────────

/// SSAO-specific render parameters.
#[derive(Debug, Clone)]
pub struct SsaoConfig {
    /// Hemisphere sample radius in view space. Default: 0.5.
    pub radius:   f32,
    /// Depth bias to prevent self-occlusion artefacts. Default: 0.025.
    pub bias:     f32,
    /// AO blend strength [0.0, 1.0]. 0.0 disables SSAO. Default: 0.8.
    pub strength: f32,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self { radius: 0.5, bias: 0.025, strength: 0.8 }
    }
}

/// Per-frame mesh render configuration.
#[derive(Debug, Clone)]
pub struct MeshRenderConfig {
    /// Number of depth-peel layers [1, `N_PEEL_LAYERS`]. Default: `N_PEEL_LAYERS` (4).
    pub peel_layers: usize,
    /// SSAO parameters.
    pub ssao: SsaoConfig,
}

impl Default for MeshRenderConfig {
    fn default() -> Self {
        Self { peel_layers: N_PEEL_LAYERS, ssao: SsaoConfig::default() }
    }
}

// ── Pending readback ──────────────────────────────────────────────────────────

struct PendingMeshReadback {
    rx:   std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    rows: usize,
    cols: usize,
}

// ── Public renderer ───────────────────────────────────────────────────────────

/// GPU mesh renderer: OIT depth peeling + SSAO, non-blocking async readback.
pub struct GpuMeshRenderer {
    ctx:     GpuMeshContext,
    cache:   Option<GpuMeshFrameCache>,
    mesh:    Option<GpuMeshBufs>,
    pending: Option<PendingMeshReadback>,
    last:    Option<ColorImage>,
}

impl GpuMeshRenderer {
    /// Attempt to create a GPU mesh renderer.
    ///
    /// Returns `None` when no GPU is available.
    pub fn try_create() -> Option<Self> {
        let ctx = GpuMeshContext::try_new()?;
        Some(Self { ctx, cache: None, mesh: None, pending: None, last: None })
    }

    /// Render `mesh` with the given camera, material, lights, and config.
    ///
    /// # Async contract
    ///
    /// Returns `None` on the first call (GPU work submitted, no frame ready yet).
    /// Subsequent calls return the last completed frame while new GPU work is
    /// submitted.  Viewport resize clears pending state and restarts.
    pub fn render(
        &mut self,
        mesh:   &VtkPolyData,
        camera: &MeshCamera,
        mat:    &PhongMaterial,
        lights: &[DirectionalLight],
        width:  usize,
        height: usize,
        config: &MeshRenderConfig,
    ) -> Option<ColorImage> {
        // Non-blocking poll: drive any completed map_async callbacks.
        let _ = self.ctx.device.poll(wgpu::Maintain::Poll);

        // Collect completed readback if available.
        if let Some(ref pending) = self.pending {
            if let Ok(Ok(())) = pending.rx.try_recv() {
                let (rows, cols) = (pending.rows, pending.cols);
                // Dimensions must match cache; if resize occurred, discard.
                let collect = self.cache.as_ref()
                    .map(|c| c.rows == rows && c.cols == cols)
                    .unwrap_or(false);
                let image = if collect {
                    Some(collect_mesh_result(self.cache.as_ref().unwrap()))
                } else {
                    None
                };
                self.pending = None;
                if let Some(img) = image {
                    self.last = Some(img);
                }
            }
        }

        // Rebuild frame cache if viewport dimensions changed.
        let needs_cache_rebuild = self.cache
            .as_ref()
            .map(|c| c.rows != height || c.cols != width)
            .unwrap_or(true);

        if needs_cache_rebuild {
            self.cache   = Some(GpuMeshFrameCache::new(&self.ctx.device, height, width));
            self.pending = None;
            self.last    = None;
        }

        // Rebuild mesh buffers if the mesh changed (pointer-based change detection).
        let needs_mesh_rebuild = self.mesh
            .as_ref()
            .map(|b| b.points_ptr != mesh.points.as_ptr() as usize)
            .unwrap_or(true);

        if needs_mesh_rebuild {
            self.mesh = GpuMeshBufs::build(&self.ctx.device, mesh);
        }

        // If no pending readback is in-flight, submit new GPU work.
        if self.pending.is_none() {
            if let (Some(cache), Some(gpu_mesh)) = (self.cache.as_ref(), self.mesh.as_ref()) {
                let (scene, lights_block, material, ssao_u) =
                    build_uniforms(camera, mat, lights, config, height, width);

                let rx = submit_mesh_async(
                    &self.ctx, cache, gpu_mesh,
                    scene, lights_block, material, ssao_u,
                    config.peel_layers,
                );
                self.pending = Some(PendingMeshReadback {
                    rx,
                    rows: height,
                    cols: width,
                });
            }
        }

        self.last.clone()
    }

    /// Block until the GPU finishes the current frame and return the image.
    ///
    /// Only for use in tests. NOT safe to call from the render thread.
    #[cfg(test)]
    pub(super) fn render_sync(
        &mut self,
        mesh:   &VtkPolyData,
        camera: &MeshCamera,
        mat:    &PhongMaterial,
        lights: &[DirectionalLight],
        width:  usize,
        height: usize,
        config: &MeshRenderConfig,
    ) -> Option<ColorImage> {
        // Submit frame.
        let _ = self.render(mesh, camera, mat, lights, width, height, config);
        // Block until GPU completes.
        self.ctx.device.poll(wgpu::Maintain::Wait);
        // Collect result.
        self.render(mesh, camera, mat, lights, width, height, config)
    }

    /// Non-blocking GPU poll helper for tests.
    #[cfg(test)]
    pub(super) fn poll_blocking(&self) {
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }
}

// ── Uniform construction ──────────────────────────────────────────────────────

/// Build all uniform structs from public-API types.
fn build_uniforms(
    camera: &MeshCamera,
    mat:    &PhongMaterial,
    lights: &[DirectionalLight],
    config: &MeshRenderConfig,
    height: usize,
    width:  usize,
) -> (SceneUniforms, LightBlock, MaterialUniforms, SsaoUniforms) {
    let view = look_at(camera.eye, camera.target, camera.up);
    let proj = perspective(camera.fov_y, camera.aspect, camera.near, camera.far);
    let mvp  = mat4_mul(proj, view);

    let scene = SceneUniforms {
        mvp,
        mv:        view,
        peel_pass: 0,
        _pad:      [0; 3],
    };

    // Transform light directions to view space using the upper-left 3×3 of MV.
    let transform_dir = |d: [f32; 3]| -> [f32; 3] {
        let n = normalize(d);
        let x = view[0]*n[0] + view[4]*n[1] + view[8]*n[2];
        let y = view[1]*n[0] + view[5]*n[1] + view[9]*n[2];
        let z = view[2]*n[0] + view[6]*n[1] + view[10]*n[2];
        normalize([x, y, z])
    };

    let mut lu = [LightUniform::zeroed(); 2];
    for (i, l) in lights.iter().take(2).enumerate() {
        lu[i] = LightUniform {
            direction_view: transform_dir(l.direction),
            _pad0: 0.0,
            color:   l.color,
            _pad1:   0.0,
            ambient: mat.ambient,
            _pad2:   0.0,
        };
    }
    // Fill unused light slots with zeroed entries (no contribution).
    let lights_block = LightBlock { lights: lu };

    let material = MaterialUniforms {
        diffuse:        [mat.diffuse[0], mat.diffuse[1], mat.diffuse[2], 1.0],
        specular_shine: [mat.specular[0], mat.specular[1], mat.specular[2], mat.shininess],
        opacity_pad:    [mat.opacity, 0.0, 0.0, 0.0],
    };

    // focal_x = proj[0] = f/aspect, focal_y = proj[5] = f  (column-major).
    let focal_x = proj[0];
    let focal_y = proj[5];
    let ssao_u = SsaoUniforms {
        near:       camera.near,
        far:        camera.far,
        focal_x,
        focal_y,
        radius:     config.ssao.radius,
        bias:       config.ssao.bias,
        n_samples:  16,
        strength:   config.ssao.strength,
        viewport_w: width  as u32,
        viewport_h: height as u32,
        _pad:       [0; 2],
    };

    (scene, lights_block, material, ssao_u)
}

// ── LightUniform::zeroed() helper ─────────────────────────────────────────────

impl LightUniform {
    fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}
