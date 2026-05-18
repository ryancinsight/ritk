use super::AttributeArray;
use std::collections::HashMap;

/// VTK ImageData domain object — a regular Cartesian grid of scalar/vector/tensor fields.
///
/// # Invariants (mathematical)
/// - `whole_extent[2i+1] >= whole_extent[2i]` for i in {0,1,2}.
/// - `n_points() = product over i of (whole_extent[2i+1] - whole_extent[2i] + 1)`.
/// - `n_cells() = product over i of max(1, whole_extent[2i+1] - whole_extent[2i])`.
/// - Each `AttributeArray::Scalars` in `point_data` has `values.len() == n_points() * num_components`.
/// - Each `AttributeArray::Scalars` in `cell_data` has `values.len() == n_cells() * num_components`.
/// - Each `AttributeArray::Vectors` in `point_data` has `values.len() == n_points()`.
#[derive(Debug, Clone, Default)]
pub struct VtkImageData {
    /// `[x0, x1, y0, y1, z0, z1]` — inclusive extent indices (VTK convention).
    pub whole_extent: [i64; 6],
    /// Physical origin of the dataset (coordinate of the first point).
    pub origin: [f64; 3],
    /// Voxel spacing in each axis direction.
    pub spacing: [f64; 3],
    /// Named point-centered attribute arrays, keyed by name.
    pub point_data: HashMap<String, AttributeArray>,
    /// Named cell-centered attribute arrays, keyed by name.
    pub cell_data: HashMap<String, AttributeArray>,
}

impl VtkImageData {
    /// Number of points = product of (extent_max - extent_min + 1) over all 3 axes.
    pub fn n_points(&self) -> usize {
        let e = &self.whole_extent;
        ((e[1] - e[0] + 1) as usize) * ((e[3] - e[2] + 1) as usize) * ((e[5] - e[4] + 1) as usize)
    }

    /// Number of cells = product of max(1, extent_max - extent_min) over all 3 axes.
    pub fn n_cells(&self) -> usize {
        let e = &self.whole_extent;
        (1_usize.max((e[1] - e[0]) as usize))
            * (1_usize.max((e[3] - e[2]) as usize))
            * (1_usize.max((e[5] - e[4]) as usize))
    }

    /// Validate all invariants.
    ///
    /// Returns `Err` with a description of the first violation found.
    pub fn validate(&self) -> Result<(), String> {
        let e = &self.whole_extent;
        for i in 0..3 {
            if e[2 * i + 1] < e[2 * i] {
                return Err(format!(
                    "whole_extent[{}] < whole_extent[{}]: {} < {}",
                    2 * i + 1,
                    2 * i,
                    e[2 * i + 1],
                    e[2 * i]
                ));
            }
        }
        let np = self.n_points();
        #[allow(clippy::collapsible_match)]
        for (name, arr) in &self.point_data {
            match arr {
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = np * (*num_components);
                    if values.len() != expected {
                        return Err(format!(
                            "point_data '{}': expected {} values, got {}",
                            name,
                            expected,
                            values.len()
                        ));
                    }
                }
                AttributeArray::Vectors { values } => {
                    // values: Vec<[f32; 3]> — one 3-vector per point.
                    if values.len() != np {
                        return Err(format!(
                            "point_data vectors '{}': expected {} vectors, got {}",
                            name,
                            np,
                            values.len()
                        ));
                    }
                }
                _ => {}
            }
        }
        let nc = self.n_cells();
        #[allow(clippy::single_match)]
        for (name, arr) in &self.cell_data {
            match arr {
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = nc * (*num_components);
                    if values.len() != expected {
                        return Err(format!(
                            "cell_data '{}': expected {} values, got {}",
                            name,
                            expected,
                            values.len()
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
