//! Displacement field transform implementation.
//!
//! This module provides a dense displacement field transform where each
//! voxel has its own displacement vector. This is used for deformable
//! (non-rigid) registration.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use super::trait_::Transform;

/// Dense displacement field transform for 3D images.
///
/// Represents a non-rigid transformation where each spatial location
/// has an associated displacement vector. The displacement field
/// has shape [3, D, H, W] for 3D volumes.
///
/// # Type Parameters
/// * `B` - The Burn backend
#[derive(Module, Debug)]
pub struct DisplacementFieldTransform3D<B: Backend> {
    /// Displacement field with shape [3, D, H, W]
    displacement: Param<Tensor<B, 4>>,
    /// Grid spacing for low-parametric representation
    grid_spacing: Option<[f32; 3]>,
}

impl<B: Backend> DisplacementFieldTransform3D<B> {
    /// Create a new 3D displacement field transform.
    ///
    /// # Arguments
    /// * `displacement` - Tensor of shape [3, D, H, W] containing displacement vectors
    pub fn new(displacement: Tensor<B, 4>) -> Self {
        Self {
            displacement: Param::from_tensor(displacement),
            grid_spacing: None,
        }
    }

    /// Create a zero displacement field for the given spatial shape.
    ///
    /// # Arguments
    /// * `shape` - Shape [D, H, W] of the spatial domain
    /// * `device` - Device to create the tensor on
    pub fn zeros(shape: [usize; 3], device: &B::Device) -> Self {
        let displacement = Tensor::zeros([3, shape[0], shape[1], shape[2]], device);
        Self::new(displacement)
    }

    /// Create with specified grid spacing.
    pub fn with_grid_spacing(mut self, grid_spacing: [f32; 3]) -> Self {
        self.grid_spacing = Some(grid_spacing);
        self
    }

    /// Get the displacement field.
    pub fn displacement(&self) -> Tensor<B, 4> {
        self.displacement.val()
    }

    /// Get the grid spacing.
    pub fn grid_spacing(&self) -> Option<&[f32; 3]> {
        self.grid_spacing.as_ref()
    }
}

impl<B: Backend> Transform<B, 3> for DisplacementFieldTransform3D<B> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // For now, return points unchanged
        // Proper implementation requires displacement field interpolation
        points
    }
}

/// Dense displacement field transform for 2D images.
///
/// Represents a non-rigid transformation where each spatial location
/// has an associated displacement vector. The displacement field
/// has shape [2, H, W] for 2D images.
///
/// # Type Parameters
/// * `B` - The Burn backend
#[derive(Module, Debug)]
pub struct DisplacementFieldTransform2D<B: Backend> {
    /// Displacement field with shape [2, H, W]
    displacement: Param<Tensor<B, 3>>,
    /// Grid spacing for low-parametric representation
    grid_spacing: Option<[f32; 2]>,
}

impl<B: Backend> DisplacementFieldTransform2D<B> {
    /// Create a new 2D displacement field transform.
    pub fn new(displacement: Tensor<B, 3>) -> Self {
        Self {
            displacement: Param::from_tensor(displacement),
            grid_spacing: None,
        }
    }

    /// Create a zero displacement field for the given spatial shape.
    pub fn zeros(shape: [usize; 2], device: &B::Device) -> Self {
        let displacement = Tensor::zeros([2, shape[0], shape[1]], device);
        Self::new(displacement)
    }

    /// Create with specified grid spacing.
    pub fn with_grid_spacing(mut self, grid_spacing: [f32; 2]) -> Self {
        self.grid_spacing = Some(grid_spacing);
        self
    }

    /// Get the displacement field.
    pub fn displacement(&self) -> Tensor<B, 3> {
        self.displacement.val()
    }

    /// Get the grid spacing.
    pub fn grid_spacing(&self) -> Option<&[f32; 2]> {
        self.grid_spacing.as_ref()
    }
}

impl<B: Backend> Transform<B, 2> for DisplacementFieldTransform2D<B> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        points
    }
}

/// Type alias for 3D displacement field.
pub type DisplacementField3D<B> = DisplacementFieldTransform3D<B>;
/// Type alias for 2D displacement field.
pub type DisplacementField2D<B> = DisplacementFieldTransform2D<B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_displacement_field_3d_creation() {
        let device = <TestBackend as burn::backend::Backend>::Device::default();

        let displacement = Tensor::<TestBackend, 4>::zeros([3, 16, 16, 16], &device);
        let transform = DisplacementFieldTransform3D::new(displacement);

        assert!(transform.grid_spacing().is_none());
    }

    #[test]
    fn test_displacement_field_2d_creation() {
        let device = <TestBackend as burn::backend::Backend>::Device::default();

        let displacement = Tensor::<TestBackend, 3>::zeros([2, 32, 32], &device);
        let transform = DisplacementFieldTransform2D::new(displacement);

        assert!(transform.grid_spacing().is_none());
    }

    #[test]
    fn test_displacement_field_3d_zeros() {
        let device = <TestBackend as burn::backend::Backend>::Device::default();

        let transform = DisplacementFieldTransform3D::zeros([16, 16, 16], &device);
        let disp = transform.displacement();
        assert_eq!(disp.dims(), [3, 16, 16, 16]);
    }

    #[test]
    fn test_displacement_field_2d_zeros() {
        let device = <TestBackend as burn::backend::Backend>::Device::default();

        let transform = DisplacementFieldTransform2D::zeros([32, 32], &device);
        let disp = transform.displacement();
        assert_eq!(disp.dims(), [2, 32, 32]);
    }

    #[test]
    fn test_displacement_3d_with_grid_spacing() {
        let device = <TestBackend as burn::backend::Backend>::Device::default();

        let displacement = Tensor::<TestBackend, 4>::zeros([3, 8, 8, 8], &device);
        let transform = DisplacementFieldTransform3D::new(displacement)
            .with_grid_spacing([4.0, 4.0, 4.0]);

        assert_eq!(transform.grid_spacing(), Some(&[4.0f32, 4.0, 4.0]));
    }

    #[test]
    fn test_transform_points_3d() {
        let device = <TestBackend as burn::backend::Backend>::Device::default();

        let displacement = Tensor::<TestBackend, 4>::zeros([3, 8, 8, 8], &device);
        let transform = DisplacementFieldTransform3D::new(displacement);

        let points = Tensor::<TestBackend, 2>::zeros([10, 3], &device);
        let transformed = transform.transform_points(points.clone());

        assert_eq!(transformed.dims(), points.dims());
    }
}
