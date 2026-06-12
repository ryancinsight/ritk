use super::BSplineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

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

    const EPS: f32 = 1e-9;

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

        if max_val < EPS {
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
            if factor.abs() < EPS {
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
mod tests {
    use super::try_invert_small;

    /// Multiply two flat row-major D×D matrices and return the flat result.
    fn mat_mul<const D: usize>(a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = D;
        let mut out = vec![0.0f32; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[r * n + k] * b[k * n + c];
                }
                out[r * n + c] = sum;
            }
        }
        out
    }

    /// Compute max absolute difference between two flat matrices.
    fn max_abs_diff<const D: usize>(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // ── D = 1 ─────────────────────────────────────────────────────────

    #[test]
    fn invert_1x1_identity() {
        let a = [1.0f32];
        let inv = try_invert_small::<1>(&a).expect("identity should be invertible");
        assert!((inv[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn invert_1x1_non_trivial() {
        let a = [4.0f32];
        let inv = try_invert_small::<1>(&a).expect("4×1 should be invertible");
        assert!((inv[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn invert_1x1_singular() {
        assert!(try_invert_small::<1>(&[0.0f32]).is_none());
        assert!(try_invert_small::<1>(&[1e-12f32]).is_none());
    }

    #[test]
    fn invert_1x1_round_trip() {
        let a = [7.5f32];
        let inv = try_invert_small::<1>(&a).unwrap();
        let prod = mat_mul::<1>(&a, &inv);
        let ident = [1.0f32];
        assert!(max_abs_diff::<1>(&prod, &ident) < 1e-5);
    }

    // ── D = 2 ─────────────────────────────────────────────────────────

    #[test]
    fn invert_2x2_identity() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let inv = try_invert_small::<2>(&a).expect("identity should be invertible");
        assert!(max_abs_diff::<2>(&inv, &[1.0, 0.0, 0.0, 1.0]) < 1e-6);
    }

    #[test]
    fn invert_2x2_known_inverse() {
        // A = [[2, 1], [5, 3]]  →  A⁻¹ = [[3, -1], [-5, 2]]
        let a = [2.0, 1.0, 5.0, 3.0];
        let expected = [3.0, -1.0, -5.0, 2.0];
        let inv = try_invert_small::<2>(&a).expect("should be invertible");
        assert!(max_abs_diff::<2>(&inv, &expected) < 1e-5);
    }

    #[test]
    fn invert_2x2_round_trip() {
        let a = [2.0, 1.0, 5.0, 3.0];
        let inv = try_invert_small::<2>(&a).unwrap();
        let prod = mat_mul::<2>(&a, &inv);
        let ident: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        assert!(max_abs_diff::<2>(&prod, &ident) < 1e-5);
    }

    #[test]
    fn invert_2x2_singular() {
        // Rows are linearly dependent: [1,2] and [2,4]
        let a = [1.0, 2.0, 2.0, 4.0];
        assert!(try_invert_small::<2>(&a).is_none());
    }

    #[test]
    fn invert_2x2_diagonal() {
        let a = [3.0, 0.0, 0.0, 5.0];
        let expected = [1.0 / 3.0, 0.0, 0.0, 1.0 / 5.0];
        let inv = try_invert_small::<2>(&a).expect("diagonal should be invertible");
        assert!(max_abs_diff::<2>(&inv, &expected) < 1e-5);
    }

    // ── D = 3 ─────────────────────────────────────────────────────────

    #[test]
    fn invert_3x3_identity() {
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = try_invert_small::<3>(&a).expect("identity should be invertible");
        assert!(max_abs_diff::<3>(&inv, &a) < 1e-6);
    }

    #[test]
    fn invert_3x3_known_inverse() {
        // A = [[3, 0, 2], [2, 0, -2], [0, 1, 1]]
        // A⁻¹ = [[0.2, 0.2, 0.0], [-0.2, 0.3, 1.0], [0.2, -0.3, 0.0]]
        let a = [3.0, 0.0, 2.0, 2.0, 0.0, -2.0, 0.0, 1.0, 1.0];
        let expected = [0.2, 0.2, 0.0, -0.2, 0.3, 1.0, 0.2, -0.3, 0.0];
        let inv = try_invert_small::<3>(&a).expect("should be invertible");
        assert!(max_abs_diff::<3>(&inv, &expected) < 1e-4);
    }

    #[test]
    fn invert_3x3_round_trip() {
        let a = [3.0, 0.0, 2.0, 2.0, 0.0, -2.0, 0.0, 1.0, 1.0];
        let inv = try_invert_small::<3>(&a).unwrap();
        let prod = mat_mul::<3>(&a, &inv);
        let ident: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(max_abs_diff::<3>(&prod, &ident) < 1e-5);
    }

    #[test]
    fn invert_3x3_singular() {
        // Row 2 = 2 * Row 0
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0];
        assert!(try_invert_small::<3>(&a).is_none());
    }

    #[test]
    fn invert_3x3_diagonal() {
        let a = [2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0];
        let expected = [0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.125];
        let inv = try_invert_small::<3>(&a).expect("diagonal should be invertible");
        assert!(max_abs_diff::<3>(&inv, &expected) < 1e-5);
    }

    #[test]
    fn invert_3x3_rotation_round_trip() {
        // 90° rotation around Z axis
        let a = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let inv = try_invert_small::<3>(&a).unwrap();
        let prod = mat_mul::<3>(&a, &inv);
        let ident: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(max_abs_diff::<3>(&prod, &ident) < 1e-5);
    }

    // ── D = 4 ─────────────────────────────────────────────────────────

    #[test]
    fn invert_4x4_identity() {
        let mut a = vec![0.0f32; 16];
        a[0] = 1.0;
        a[5] = 1.0;
        a[10] = 1.0;
        a[15] = 1.0;
        let inv = try_invert_small::<4>(&a).expect("identity should be invertible");
        assert!(max_abs_diff::<4>(&inv, &a) < 1e-6);
    }

    #[test]
    fn invert_4x4_diagonal() {
        let mut a = vec![0.0f32; 16];
        a[0] = 2.0;
        a[5] = 3.0;
        a[10] = 5.0;
        a[15] = 7.0;
        let inv = try_invert_small::<4>(&a).expect("diagonal should be invertible");
        let mut expected = vec![0.0f32; 16];
        expected[0] = 0.5;
        expected[5] = 1.0 / 3.0;
        expected[10] = 0.2;
        expected[15] = 1.0 / 7.0;
        assert!(max_abs_diff::<4>(&inv, &expected) < 1e-5);
    }

    #[test]
    fn invert_4x4_round_trip() {
        let a = [
            2.0, 1.0, 0.0, 3.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0,
        ];
        let inv = try_invert_small::<4>(&a).unwrap();
        let prod = mat_mul::<4>(&a, &inv);
        let mut ident = vec![0.0f32; 16];
        ident[0] = 1.0;
        ident[5] = 1.0;
        ident[10] = 1.0;
        ident[15] = 1.0;
        assert!(max_abs_diff::<4>(&prod, &ident) < 1e-5);
    }

    #[test]
    fn invert_4x4_singular() {
        // Row 2 is a zero row — clearly rank-deficient.
        let a = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0,
        ];
        assert!(try_invert_small::<4>(&a).is_none());
    }

    // ── Near-singular (pivoting stress) ───────────────────────────────

    #[test]
    fn invert_near_singular_pivots() {
        // Matrix that is invertible but requires pivoting because the
        // (0,0) entry is small relative to (1,0).
        let a = [1e-7, 2.0, 2.0, 3.0];
        let inv = try_invert_small::<2>(&a).expect("should be invertible with pivoting");
        let prod = mat_mul::<2>(&a, &inv);
        let ident = [1.0, 0.0, 0.0, 1.0];
        assert!(max_abs_diff::<2>(&prod, &ident) < 1e-5);
    }
}
