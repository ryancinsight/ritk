use crate::types::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
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
            TensorData::new(origin_vec, burn::tensor::Shape::new([D])),
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
            TensorData::new(t_data, burn::tensor::Shape::new([D, D])),
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
            TensorData::new(origin_vec, burn::tensor::Shape::new([D])),
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
            TensorData::new(m_data, burn::tensor::Shape::new([D, D])),
            &device,
        );

        apply_row_chunks(indices, ritk_wgpu_compat::WGPU_CHUNK_SIZE, |chunk| {
            chunk.matmul(m_tensor.clone()) + origin_tensor.clone()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_spatial::{Direction, Spacing};

    type Backend = NdArray<f32>;
    type Point3 = Point<3>;
    type Spacing3 = Spacing<3>;
    type Direction3 = Direction<3>;

    #[test]
    fn test_physical_to_index_transform() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();

        let image = Image::new(data, origin, spacing, direction);

        let point = Point3::new([5.0, 5.0, 5.0]);
        let index = image.transform_physical_point_to_continuous_index(&point);

        assert!((index[0] - 5.0).abs() < 1e-6);
        assert!((index[1] - 5.0).abs() < 1e-6);
        assert!((index[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_to_physical_transform() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();

        let image = Image::new(data, origin, spacing, direction);

        let index = Point3::new([5.0, 5.0, 5.0]);
        let point = image.transform_continuous_index_to_physical_point(&index);

        assert!((point[0] - 5.0).abs() < 1e-6);
        assert!((point[1] - 5.0).abs() < 1e-6);
        assert!((point[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_roundtrip() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();

        let image = Image::new(data, origin, spacing, direction);

        let original_point = Point3::new([3.5, 4.5, 5.5]);
        let index = image.transform_physical_point_to_continuous_index(&original_point);
        let transformed_point = image.transform_continuous_index_to_physical_point(&index);

        assert!((original_point[0] - transformed_point[0]).abs() < 1e-6);
        assert!((original_point[1] - transformed_point[1]).abs() < 1e-6);
        assert!((original_point[2] - transformed_point[2]).abs() < 1e-6);
    }

    #[test]
    fn test_non_unit_spacing() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([2.0, 2.0, 2.0]);
        let direction = Direction3::identity();

        let image = Image::new(data, origin, spacing, direction);

        let point = Point3::new([10.0, 10.0, 10.0]);
        let index = image.transform_physical_point_to_continuous_index(&point);

        assert!((index[0] - 5.0).abs() < 1e-6);
        assert!((index[1] - 5.0).abs() < 1e-6);
        assert!((index[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_non_zero_origin() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([10.0, 20.0, 30.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();

        let image = Image::new(data, origin, spacing, direction);

        let point = Point3::new([15.0, 25.0, 35.0]);
        let index = image.transform_physical_point_to_continuous_index(&point);

        assert!((index[0] - 5.0).abs() < 1e-6);
        assert!((index[1] - 5.0).abs() < 1e-6);
        assert!((index[2] - 5.0).abs() < 1e-6);
    }
}
