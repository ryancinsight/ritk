use super::state::SnapApp;

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;

#[cfg(target_arch = "wasm32")]
struct FileDialog;

#[cfg(target_arch = "wasm32")]
impl FileDialog {
    fn new() -> Self {
        Self
    }
    fn set_file_name(self, _: &str) -> Self {
        self
    }
    fn add_filter(self, _: &str, _: &[&str]) -> Self {
        self
    }
    fn save_file(self) -> Option<std::path::PathBuf> {
        None
    }
}

use tracing::{error, info};

fn binary_mask_from_labels(labels: &[u32]) -> Option<Vec<f32>> {
    let mut has_foreground = false;
    let mut binary = Vec::with_capacity(labels.len());
    for &label in labels {
        if label > 0 {
            has_foreground = true;
            binary.push(1.0);
        } else {
            binary.push(0.0);
        }
    }

    if has_foreground {
        Some(binary)
    } else {
        None
    }
}

fn build_label_surface_mesh(
    binary: &[f32],
    shape: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
) -> gaia::IndexedMesh<f64> {
    ritk_filter::surface::MarchingCubesFilter::new()
        .with_isovalue(0.5)
        .with_spacing(spacing)
        .with_origin(origin)
        .extract(binary, shape)
}

impl SnapApp {
    /// Export the active label map as a VTK legacy POLYDATA surface mesh.
    pub(super) fn export_surface_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Export surface: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();
        let Some(binary) = binary_mask_from_labels(map.as_slice()) else {
            self.status_message =
                "Export surface: no foreground voxels - mesh is empty.".to_owned();
            return;
        };

        let Some(path) = FileDialog::new()
            .set_file_name("surface.vtk")
            .add_filter("VTK Polydata", &["vtk"])
            .save_file()
        else {
            return;
        };

        let spacing = [vol.spacing[0], vol.spacing[1], vol.spacing[2]];
        let origin = [vol.origin[0], vol.origin[1], vol.origin[2]];
        let mesh = build_label_surface_mesh(&binary, map.shape.0, spacing, origin);

        match ritk_io::write_mesh_as_vtk(&path, &mesh) {
            Ok(()) => {
                self.status_message = format!(
                    "Exported surface ({} triangles) to {}",
                    mesh.face_count(),
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Surface export failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/surface_export.rs"]
mod tests;
