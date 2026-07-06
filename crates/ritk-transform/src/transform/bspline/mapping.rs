use super::BSplineTransform;
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};

/// Minimum pivot magnitude for B-spline mapping inverse; guards against singular matrices.
/// Practical threshold above f32 minimum normal (~1.2e-38).
const PIVOT_SINGULARITY_GUARD: f32 = 1e-9;

/// Invert a small square matrix (`D ≤ 4`) using Gaussian elimination with
/// partial pivoting.
///
/// Returns `None` if the matrix is singular (determinant below `1e-9`).
///
/// # Arguments
/// * `data` — Flat row-major slice of `D * D` `f32` values.
///
/// # Panics
/// Debug-panics if `data.len() != D * D`.
pub(crate) fn try_invert_small<const D: usize>(data: &[f32]) -> Option<Vec<f32>> {
    let n = D;
    let n2 = n * n;
    debug_assert_eq!(data.len(), n2);

    // Augmented matrix [A | I] stored row-major: n rows of (n + n) columns.
    let cols = 2 * n;
    let mut aug = vec![0.0f32; n * cols];
    for r in 0..n {
        for c in 0..n {
            aug[r * cols + c] = data[r * n + c];
        }
        aug[r * cols + n + r] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting: find row with largest |value| in this column.
        let mut max_val = aug[col * cols + col].abs();
        let mut pivot_row = col;
        for r in (col + 1)..n {
            let v = aug[r * cols + col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = r;
            }
        }

        if max_val < PIVOT_SINGULARITY_GUARD {
            return None; // singular
        }

        // Swap rows if needed.
        if pivot_row != col {
            for c in 0..cols {
                aug.swap(col * cols + c, pivot_row * cols + c);
            }
        }

        // Scale pivot row to 1.
        let pivot = aug[col * cols + col];
        let inv_pivot = 1.0 / pivot;
        for c in 0..cols {
            aug[col * cols + c] *= inv_pivot;
        }

        // Eliminate this column in all other rows.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r * cols + col];
            if factor.abs() < PIVOT_SINGULARITY_GUARD {
                continue;
            }
            for c in 0..cols {
                aug[r * cols + c] -= factor * aug[col * cols + c];
            }
        }
    }

    // Extract right half (the inverse).
    let mut inv = Vec::with_capacity(n2);
    for r in 0..n {
        for c in 0..n {
            inv.push(aug[r * cols + n + c]);
        }
    }
    Some(inv)
}

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    /// Compute inverse direction matrix as a flat vector of f32 values.
    ///
    /// Returns the flattened inverse direction matrix for index calculation.
    pub(crate) fn compute_inverse_direction(&self, _device: &<B as Backend>::Device) -> Vec<f32> {
        // Get direction matrix data
        let dir_tensor = self.direction.clone();
        let dir_data_result = dir_tensor.to_data();
        let dir_slice = dir_data_result
            .as_slice::<f32>()
            .expect("direction data must be contiguous f32");

        match try_invert_small::<D>(dir_slice) {
            Some(inv) => inv,
            None => {
                // Return identity if singular (fallback)
                let mut identity = vec![0.0f32; D * D];
                for i in 0..D {
                    identity[i * D + i] = 1.0;
                }
                identity
            }
        }
    }

    /// Convert physical points to continuous grid indices.
    ///
    /// Maps from physical space to index space.
    pub(crate) fn world_to_grid_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let [_n_points, _] = points.dims();

        // 1. Prepare Origin Tensor [1, D]
        let origin_tensor = self.origin.clone().reshape([1, D]);

        // 2. Prepare Transform Matrix T = (S^-1 * D^-1)^T
        // We want: Index = (World - Origin) * Direction^-1 * Spacing^-1
        // In matrix form for row vectors: I = (W - O) * (D * S)^-1
        // Note: Direction columns are axes. Spacing scales axes.
        // Let M = Direction * Spacing (diagonal).
        // I = (W - O) * M^-1.

        // Compute inverse direction matrix
        let inv_dir = self.compute_inverse_direction(&device);

        // Compute T = M^-1. Since we multiply on right, T should be (M^-1)^T = (D * S)^-T?
        // Wait, standard affine: p = O + D * S * i
        // p - O = D * S * i
        // i = (D * S)^-1 * (p - O) = S^-1 * D^-1 * (p - O)
        // Since points are row vectors [1, D], we have p = i * (S * D^T) + O ?
        // Usually: p_col = O_col + D_mat * S_diag * i_col
        // p_row = O_row + i_row * S_diag * D_mat^T
        // p_row - O_row = i_row * (S * D^T)
        // i_row = (p_row - O_row) * (S * D^T)^-1 = (p - O) * (D^T)^-1 * S^-1 = (p - O) * (D^-1)^T * S^-1

        // T = (D^-1)^T * S^-1
        // element T[r, c] = (D^-1)^T[r, c] * S^-1[c] = (D^-1)[c, r] / S[c]

        let mut t_data = Vec::with_capacity(D * D);
        let spacing_data: Vec<f32> = self
            .spacing
            .to_data()
            .as_slice::<f32>()
            .expect("spacing data must be contiguous f32")
            .to_vec();

        for r in 0..D {
            for c in 0..D {
                // T[r, c] = inv_dir[c, r] / spacing[c]
                let inv_dir_val = inv_dir[c * D + r];
                let spacing_val = spacing_data[c];
                let val = inv_dir_val / spacing_val;
                t_data.push(val);
            }
        }

        let t_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(t_data, Shape::new([D, D])), &device);

        // Apply transform: (points - origin) @ T
        let diff = points - origin_tensor;
        diff.matmul(t_tensor)
    }
}
#[cfg(test)]
#[path = "tests_mapping.rs"]
mod tests;
