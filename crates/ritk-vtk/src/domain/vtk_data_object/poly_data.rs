use std::collections::HashMap;
use super::AttributeArray;

/// VTK polygonal mesh dataset (DATASET POLYDATA).
///
/// Invariant: every index stored in `vertices`, `lines`, `polygons`, and
/// `triangle_strips` is strictly less than `points.len()`.
#[derive(Debug, Clone, Default)]
pub struct VtkPolyData {
    /// Point coordinates [x, y, z] in mesh space.
    pub points: Vec<[f32; 3]>,
    /// VERTICES cells: each inner `Vec` is one vertex cell (list of point indices).
    pub vertices: Vec<Vec<u32>>,
    /// LINES cells: each inner `Vec` is an ordered sequence of point indices forming a polyline.
    pub lines: Vec<Vec<u32>>,
    /// POLYGONS cells: each inner `Vec` is a closed polygon (point indices).
    pub polygons: Vec<Vec<u32>>,
    /// TRIANGLE_STRIPS cells: each inner `Vec` is a triangle strip.
    pub triangle_strips: Vec<Vec<u32>>,
    /// Per-point attribute arrays keyed by name.
    pub point_data: HashMap<String, AttributeArray>,
    /// Per-cell attribute arrays keyed by name.
    pub cell_data: HashMap<String, AttributeArray>,
}

impl VtkPolyData {
    /// Total number of cells across all cell types.
    pub fn num_cells(&self) -> usize {
        self.vertices.len() + self.lines.len() + self.polygons.len() + self.triangle_strips.len()
    }

    /// Validate all index-bound and attribute-length invariants.
    ///
    /// Returns `Ok(())` if all invariants hold, otherwise a `String` describing
    /// the first violated invariant.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.points.len();

        let check_cells = |cells: &[Vec<u32>], kind: &str| -> Result<(), String> {
            for (i, cell) in cells.iter().enumerate() {
                for &idx in cell {
                    if idx as usize >= n {
                        return Err(format!(
                            "{} cell {}: index {} out of range (n_points = {})",
                            kind, i, idx, n
                        ));
                    }
                }
            }
            Ok(())
        };
        check_cells(&self.vertices, "VERTICES")?;
        check_cells(&self.lines, "LINES")?;
        check_cells(&self.polygons, "POLYGONS")?;
        check_cells(&self.triangle_strips, "TRIANGLE_STRIPS")?;

        for (name, attr) in &self.point_data {
            match attr {
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = n * num_components;
                    if values.len() != expected {
                        return Err(format!(
                            "point_data '{}': Scalars length {} != n_points*ncomp = {}",
                            name,
                            values.len(),
                            expected
                        ));
                    }
                }
                AttributeArray::Vectors { values } | AttributeArray::Normals { values } => {
                    if values.len() != n {
                        return Err(format!(
                            "point_data '{}': length {} != n_points = {}",
                            name,
                            values.len(),
                            n
                        ));
                    }
                }
                AttributeArray::TextureCoords { values, dim } => {
                    if values.len() != n * dim {
                        return Err(format!(
                            "point_data '{}': TextureCoords length {} != n_points*dim = {}",
                            name,
                            values.len(),
                            n * dim
                        ));
                    }
                }
            }
        }

        let nc = self.num_cells();
        for (name, attr) in &self.cell_data {
            match attr {
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = nc * num_components;
                    if values.len() != expected {
                        return Err(format!(
                            "cell_data '{}': Scalars length {} != n_cells*ncomp = {}",
                            name,
                            values.len(),
                            expected
                        ));
                    }
                }
                AttributeArray::Vectors { values } | AttributeArray::Normals { values } => {
                    if values.len() != nc {
                        return Err(format!(
                            "cell_data '{}': length {} != n_cells = {}",
                            name,
                            values.len(),
                            nc
                        ));
                    }
                }
                AttributeArray::TextureCoords { values, dim } => {
                    if values.len() != nc * dim {
                        return Err(format!(
                            "cell_data '{}': TextureCoords length {} != n_cells*dim = {}",
                            name,
                            values.len(),
                            nc * dim
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}
