//! Surface mesh overlay operations for [`SnapApp`].
//!
//! Provides three capabilities wired to the 3D-MIP viewport:
//!
//! - [`SnapApp::load_mesh_file`] — dispatch by extension, store in
//!   [`SnapApp::loaded_mesh`].
//! - [`SnapApp::auto_camera_for_poly`] — AABB-fitted pinhole camera.
//! - [`SnapApp::rebuild_mesh_texture`] — Phong-rasterize to an
//!   [`egui::TextureHandle`] composited on the MIP viewport.

use super::state::SnapApp;
use crate::render::mesh_render::{
    normalize, DirectionalLight, MeshCamera, MeshRenderer, PhongMaterial,
};
use ritk_io::VtkPolyData;
use std::f32::consts::PI;
use std::path::Path;
use tracing::{error, info};

impl SnapApp {
    /// Load a surface mesh from `path` and store it in [`Self::loaded_mesh`].
    ///
    /// Dispatch is on the lowercase file extension:
    /// - `.stl` → [`ritk_io::read_stl_mesh`]
    /// - `.obj` → [`ritk_io::read_obj_mesh`]
    /// - `.ply` → [`ritk_io::read_ply_mesh`]
    ///
    /// On success sets `mesh_dirty = true` and `show_mesh_overlay = true`.
    /// On failure the error is logged and written to [`Self::status_message`].
    pub(crate) fn load_mesh_file(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let result = match ext.as_str() {
            "stl" => ritk_io::read_stl_mesh(path),
            "obj" => ritk_io::read_obj_mesh(path),
            "ply" => ritk_io::read_ply_mesh(path),
            other => {
                let msg = format!("Unsupported mesh extension: .{other}");
                error!("{}", msg);
                self.status_message = msg;
                return;
            }
        };

        match result {
            Ok(poly) => {
                let n_poly = poly.polygons.len();
                info!(path = %path.display(), polygons = n_poly, "Mesh loaded");
                self.status_message = format!(
                    "Mesh loaded: {} ({} polygons)",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    n_poly
                );
                self.loaded_mesh = Some(poly);
                self.mesh_dirty = true;
                self.show_mesh_overlay = true;
            }
            Err(err) => {
                let msg = format!("Failed to load mesh {}: {err:#}", path.display());
                error!("{}", msg);
                self.status_message = msg;
            }
        }
    }

    /// Compute an auto-framing [`MeshCamera`] that centers on the AABB of `poly`.
    ///
    /// # Algorithm
    ///
    /// 1. Compute per-axis min/max over all points → AABB.
    /// 2. `center = (min + max) / 2`.
    /// 3. `diag = ||max − min||`, clamped to ≥ 1.0 to avoid degenerate framing.
    /// 4. `eye = [cx, cy, cz + diag · 1.5]` (Z-offset above the model).
    /// 5. `up = [0, 1, 0]`, `fov_y = π/4`.
    /// 6. `near = diag · 0.01`, `far = diag · 10.0`.
    ///
    /// For an empty point set, returns the default camera with the correct
    /// aspect ratio and without panicking.
    pub(crate) fn auto_camera_for_poly(poly: &VtkPolyData, w: usize, h: usize) -> MeshCamera {
        let aspect = w as f32 / h.max(1) as f32;

        if poly.points.is_empty() {
            return MeshCamera {
                aspect,
                ..MeshCamera::default()
            };
        }

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &poly.points {
            for i in 0..3 {
                if p[i] < min[i] {
                    min[i] = p[i];
                }
                if p[i] > max[i] {
                    max[i] = p[i];
                }
            }
        }

        let cx = (min[0] + max[0]) * 0.5;
        let cy = (min[1] + max[1]) * 0.5;
        let cz = (min[2] + max[2]) * 0.5;

        let dx = max[0] - min[0];
        let dy = max[1] - min[1];
        let dz = max[2] - min[2];
        let diag = ((dx * dx + dy * dy + dz * dz).sqrt()).max(1.0);

        MeshCamera {
            eye: [cx, cy, cz + diag * 1.5],
            target: [cx, cy, cz],
            up: [0.0, 1.0, 0.0],
            fov_y: PI / 4.0,
            aspect,
            near: diag * 0.01,
            far: diag * 10.0,
        }
    }

    /// Rebuild the mesh overlay [`egui::TextureHandle`] at pixel dimensions `w × h`.
    ///
    /// Renders `self.loaded_mesh` via the CPU Phong rasterizer with two
    /// directional lights (key + fill) and uploads the result as
    /// `"mesh_overlay_tex"` with linear filtering. Sets `mesh_dirty = false`.
    ///
    /// No-op when `self.loaded_mesh` is `None`.
    pub(crate) fn rebuild_mesh_texture(&mut self, ctx: &egui::Context, w: usize, h: usize) {
        let rgba = {
            let Some(poly) = &self.loaded_mesh else {
                return;
            };
            let camera = Self::auto_camera_for_poly(poly, w, h);
            let key_light = DirectionalLight {
                direction: normalize([1.0, 1.0, 1.0]),
                color: [1.0, 1.0, 1.0],
            };
            let fill_light = DirectionalLight {
                direction: normalize([-0.5, -0.5, -1.0]),
                color: [0.2, 0.2, 0.2],
            };
            let material = PhongMaterial::default();
            let renderer = MeshRenderer::new(w, h);
            renderer.render(poly, &camera, &material, &[key_light, fill_light])
        };
        // Immutable borrow of self.loaded_mesh released; safe to mutate self.

        let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        self.mesh_tex = Some(ctx.load_texture(
            "mesh_overlay_tex",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
        self.mesh_dirty = false;
    }
}

#[cfg(test)]
#[path = "tests/mesh_ops.rs"]
mod tests;
