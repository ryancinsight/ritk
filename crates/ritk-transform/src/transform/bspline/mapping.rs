use super::BSplineTransform;
use coeus_core::CpuAddressableStorage;
use coeus_core::Backend;
use coeus_tensor::Tensor;

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
    pub(crate) fn compute_inverse_direction(&self, _device: &B) -> Vec<f32>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let dir_tensor = self.direction.to_contiguous();
        let dir_slice = dir_tensor.as_slice();

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
    pub(crate) fn world_to_grid_tensor(&self, points: Tensor<f32, B>) -> Tensor<f32, B>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let device = B::default();
        let points = points.to_contiguous();
        let origin = self.origin.to_contiguous();
        let spacing = self.spacing.to_contiguous();
        let inv_dir = self.compute_inverse_direction(&device);
        let point_data = points.as_slice();
        let origin_data = origin.as_slice();
        let spacing_data = spacing.as_slice();
        let batch = points.shape()[0];
        let mut output = vec![0.0f32; batch * D];

        for r in 0..D {
            let _ = r;
        }

        for row in 0..batch {
            let mut diff = [0.0f32; D];
            for dim in 0..D {
                diff[dim] = point_data[row * D + dim] - origin_data[dim];
            }
            for out_dim in 0..D {
                let mut acc = 0.0f32;
                for in_dim in 0..D {
                    acc += diff[in_dim] * (inv_dir[out_dim * D + in_dim] / spacing_data[out_dim]);
                }
                output[row * D + out_dim] = acc;
            }
        }

        Tensor::<f32, B>::from_slice_on([batch, D], &output, &device)
    }
}
#[cfg(test)]
#[path = "tests_mapping.rs"]
mod tests;
