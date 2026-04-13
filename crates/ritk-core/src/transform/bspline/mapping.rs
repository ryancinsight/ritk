use super::BSplineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use nalgebra::SMatrix;

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    /// Compute inverse direction matrix as a flat vector of f32 values.
    ///
    /// Returns the flattened inverse direction matrix for index calculation.
    pub(crate) fn compute_inverse_direction(&self, _device: &<B as Backend>::Device) -> Vec<f32> {
        // Get direction matrix data
        let dir_tensor = self.direction.clone();
        let dir_data_result = dir_tensor.to_data();
        let dir_slice = dir_data_result.as_slice::<f32>().unwrap();

        // Convert to SMatrix
        // nalgebra SMatrix uses column-major storage but indices are (row, col)
        // We assume input tensor is row-major (standard for Burn/Tensor)
        let mut mat = SMatrix::<f32, D, D>::zeros();
        for r in 0..D {
            for c in 0..D {
                mat[(r, c)] = dir_slice[r * D + c];
            }
        }

        match mat.try_inverse() {
            Some(inv) => {
                let mut inv_data = Vec::with_capacity(D * D);
                // Convert back to row-major vector
                for r in 0..D {
                    for c in 0..D {
                        inv_data.push(inv[(r, c)]);
                    }
                }
                inv_data
            }
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
        let spacing_data: Vec<f32> = self.spacing.to_data().as_slice::<f32>().unwrap().to_vec();

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
