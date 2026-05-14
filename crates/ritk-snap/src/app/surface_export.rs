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
    ritk_core::filter::surface::MarchingCubesFilter::new()
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
            self.status_message = "Export surface: no foreground voxels - mesh is empty.".to_owned();
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
        let mesh = build_label_surface_mesh(&binary, map.shape, spacing, origin);

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
mod tests {
    use super::*;

    #[test]
    fn binary_mask_reports_empty_for_all_background() {
        let labels = vec![0_u32; 8];
        assert!(binary_mask_from_labels(&labels).is_none());
    }

    #[test]
    fn binary_mask_marks_foreground_labels() {
        let labels = vec![0_u32, 2_u32, 0_u32, 7_u32];
        let binary = binary_mask_from_labels(&labels).expect("foreground expected");
        assert_eq!(binary, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn build_label_surface_mesh_emits_faces_for_center_cube() {
        let nz = 4usize;
        let ny = 4usize;
        let nx = 4usize;
        let mut binary = vec![0.0f32; nz * ny * nx];
        for iz in 1..3 {
            for iy in 1..3 {
                for ix in 1..3 {
                    binary[iz * ny * nx + iy * nx + ix] = 1.0;
                }
            }
        }

        let mesh = build_label_surface_mesh(&binary, [nz, ny, nx], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
        assert_eq!(mesh.face_count(), 44);
    }

    #[test]
    fn marching_cubes_physical_positions_match_spacing() {
        let mut data = vec![0.0f32; 8];
        data[0] = 1.0;
        let mesh = build_label_surface_mesh(&data, [2, 2, 2], [2.0, 3.0, 4.0], [0.0, 0.0, 0.0]);

        assert_eq!(mesh.face_count(), 1);
        let n = mesh.vertex_count();
        let mut xs: Vec<f64> = (0..n)
            .map(|i| mesh.vertices.position(gaia::domain::core::index::VertexId::new(i as u32)).x)
            .collect();
        let mut ys: Vec<f64> = (0..n)
            .map(|i| mesh.vertices.position(gaia::domain::core::index::VertexId::new(i as u32)).y)
            .collect();
        let mut zs: Vec<f64> = (0..n)
            .map(|i| mesh.vertices.position(gaia::domain::core::index::VertexId::new(i as u32)).z)
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).expect("x finite"));
        ys.sort_by(|a, b| a.partial_cmp(b).expect("y finite"));
        zs.sort_by(|a, b| a.partial_cmp(b).expect("z finite"));

        assert!((xs[2] - 1.0_f64).abs() < 1e-4, "edge 0 midpoint x = 1.0, got {}", xs[2]);
        assert!((ys[2] - 1.5_f64).abs() < 1e-4, "edge 3 midpoint y = 1.5, got {}", ys[2]);
        assert!((zs[2] - 2.0_f64).abs() < 1e-4, "edge 8 midpoint z = 2.0, got {}", zs[2]);
    }
}
