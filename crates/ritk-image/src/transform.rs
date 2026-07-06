use crate::types::Image;
use crate::tensor::backend::Backend;
use crate::tensor::{Shape, Tensor, TensorData};
use ritk_spatial::Point;
use ritk_wgpu_compat::apply_row_chunks;

impl<B: Backend, const D: usize> Image<B, D> {
    /// Convert a continuous physical point to a continuous index.
    ///
    /// This transformation maps from physical space to index space using:
    /// `index = (Direction^-1 * (point - origin)) / spacing`
    ///
    /// # Arguments
    /// * `point` - A point in physical space
    ///
    /// # Returns
    /// The corresponding continuous index
    pub fn transform_physical_point_to_continuous_index(&self, point: &Point<D>) -> Point<D> {
        let diff = *point - *self.origin();
        let inv_dir = self
            .direction()
            .try_inverse()
            .expect("Direction matrix must be invertible");
        let rotated = inv_dir * diff;

        let mut index = Point::<D>::origin();
        for i in 0..D {
            index[i] = rotated[i] / self.spacing()[i];
        }
        index
    }

    /// Convert a continuous index to a physical point.
    ///
    /// This transformation maps from index space to physical space using:
    /// `point = origin + Direction * (index * spacing)`
    ///
    /// # Arguments
    /// * `index` - A continuous index
    ///
    /// # Returns
    /// The corresponding physical point
    pub fn transform_continuous_index_to_physical_point(&self, index: &Point<D>) -> Point<D> {
        let mut scaled_index = ritk_spatial::Vector::<D>::zeros();
        for i in 0..D {
            scaled_index[i] = index[i] * self.spacing()[i];
        }

        let rotated = *self.direction() * scaled_index;
        *self.origin() + rotated
    }

    /// Batch transform physical points to continuous indices using tensors.
    ///
    /// Maps from physical space to index space.
    ///
    /// # Arguments
    /// * `points` - A tensor of shape `[Batch, D]` containing physical points
    ///
    /// # Returns
    /// A tensor of shape `[Batch, D]` containing continuous indices
    pub fn world_to_index_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();

        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| self.origin()[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, Shape::new([D])),
            &device,
        )
        .reshape([1, D]);

        // 2. Prepare Transform Matrix T = (S^-1 * D^-1)^T = (D^-1)^T * S^-1
        let inv_dir = self
            .direction()
            .try_inverse()
            .expect("Direction matrix must be invertible");

        // Output index columns are INNERMOST-FIRST (column 0 = x = axis D-1), the
        // inverse of `index_to_world_tensor` and the order the interpolation kernels
        // consume. Output column `c` corresponds to spatial axis `D-1-c`.
        let mut t_data = Vec::with_capacity(D * D);
        for r in 0..D {
            for c in 0..D {
                let axis = D - 1 - c;
                let val = (inv_dir[(axis, r)] / self.spacing()[axis]) as f32;
                t_data.push(val);
            }
        }

        let t_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(t_data, Shape::new([D, D])),
            &device,
        );

        apply_row_chunks(points, ritk_wgpu_compat::WGPU_CHUNK_SIZE, |p| {
            (p - origin_tensor.clone()).matmul(t_tensor.clone())
        })
    }

    /// Batch transform continuous indices to physical points using tensors.
    ///
    /// Maps from index space to physical space.
    ///
    /// # Arguments
    /// * `indices` - A tensor of shape `[Batch, D]` containing continuous indices
    ///
    /// # Returns
    /// A tensor of shape `[Batch, D]` containing physical points
    pub fn index_to_world_tensor(&self, indices: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = indices.device();

        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| self.origin()[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, Shape::new([D])),
            &device,
        )
        .reshape([1, D]);

        // 2. Prepare Transform Matrix M = S * D^T.
        //
        // Index tensors use INNERMOST-FIRST column order (column 0 = x = axis D-1),
        // matching `grid::generate_grid` and the interpolation kernels. Spacing and
        // direction are stored AXIS-major (index 0 = depth/z). So index column `r`
        // corresponds to spatial axis `D-1-r`; pair them accordingly. (Using `r`
        // directly silently scrambled world coordinates for anisotropic spacing or
        // non-identity direction — only identity/isotropic cases were unaffected.)
        let mut m_data = Vec::with_capacity(D * D);
        for r in 0..D {
            let axis = D - 1 - r;
            for c in 0..D {
                let val = (self.spacing()[axis] * self.direction()[(c, axis)]) as f32;
                m_data.push(val);
            }
        }

        let m_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(m_data, Shape::new([D, D])),
            &device,
        );

        apply_row_chunks(indices, ritk_wgpu_compat::WGPU_CHUNK_SIZE, |chunk| {
            chunk.matmul(m_tensor.clone()) + origin_tensor.clone()
        })
    }
}

#[cfg(test)]
#[path = "tests_transform.rs"]
mod tests;
