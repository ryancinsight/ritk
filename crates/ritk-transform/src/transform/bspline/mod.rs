//! B-Spline transform implementation.
//!
//! This module provides a B-Spline free-form deformation transform.

use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// B-Spline Transform (Free-form deformation).
///
/// Uses a grid of control points to define a smooth deformation field.
/// The transform is defined by B-Spline interpolation of control point displacements.
///
/// The transform maps points in physical space to continuous indices in the control grid,
/// interpolates the displacement, and adds it to the original point.
///
/// Points outside the defined grid support (0 to grid_size-1) have zero displacement.
#[derive(Module, Debug)]
pub struct BSplineTransform<B: Backend, const D: usize> {
    /// Control point grid dimensions
    grid_size: [usize; D],
    /// Physical origin of the grid (index 0,0,0) as tensor `[D]`
    origin: Tensor<B, 1>,
    /// Physical spacing between control points as tensor `[D]`
    spacing: Tensor<B, 1>,
    /// Physical orientation of the grid as tensor `[D, D]` (direction matrix)
    direction: Tensor<B, 2>,
    /// Control point displacements `[num_control_points, D]`
    coefficients: Param<Tensor<B, 2>>,
}

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    /// Create a new B-Spline transform.
    ///
    /// # Arguments
    /// * `grid_size` - Number of control points along each dimension
    /// * `origin` - Physical origin of the grid (position of first control point) as tensor `\[D\]`
    /// * `spacing` - Physical spacing between control points as tensor `\[D\]`
    /// * `direction` - Physical direction matrix of the grid as tensor `\[D, D\]`
    /// * `coefficients` - Initial control point displacements `\[num_control_points, D\]`
    pub fn new(
        grid_size: [usize; D],
        origin: Tensor<B, 1>,
        spacing: Tensor<B, 1>,
        direction: Tensor<B, 2>,
        coefficients: Tensor<B, 2>,
    ) -> Self {
        assert!(
            grid_size.iter().all(|&x| x >= 4),
            "BSpline grid size must be at least 4 in all dimensions to support cubic B-splines"
        );

        Self {
            grid_size,
            origin,
            spacing,
            direction,
            coefficients: Param::from_tensor(coefficients),
        }
    }

    /// Create a B-Spline transform from spatial types.
    ///
    /// # Arguments
    /// * `grid_size` - Number of control points along each dimension
    /// * `origin` - Physical origin as `Point<D>`
    /// * `spacing` - Physical spacing as `Spacing<D>` (`Vector<D>`)
    /// * `direction` - Physical direction as `Direction<D>`
    /// * `coefficients` - Initial control point displacements `\[num_control_points, D\]`
    /// * `device` - Device to create tensors on
    pub fn from_spatial(
        grid_size: [usize; D],
        origin: &ritk_core::spatial::Point<D>,
        spacing: &ritk_core::spatial::Vector<D>,
        direction: &ritk_core::spatial::Direction<D>,
        coefficients: Tensor<B, 2>,
        device: &B::Device,
    ) -> Self {
        // Convert origin to tensor
        let origin_vec: Vec<f32> = (0..D).map(|i| origin[i] as f32).collect();
        let origin_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(origin_vec, Shape::new([D])), device);

        // Convert spacing to tensor
        let spacing_vec: Vec<f32> = (0..D).map(|i| spacing[i] as f32).collect();
        let spacing_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(spacing_vec, Shape::new([D])), device);

        // Convert direction to tensor
        let mut dir_data = Vec::with_capacity(D * D);
        for c in 0..D {
            for r in 0..D {
                dir_data.push(direction[(r, c)] as f32);
            }
        }
        let direction_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(dir_data, Shape::new([D, D])), device);

        Self::new(
            grid_size,
            origin_tensor,
            spacing_tensor,
            direction_tensor,
            coefficients,
        )
    }

    /// Get the grid size.
    pub fn grid_size(&self) -> [usize; D] {
        self.grid_size
    }

    /// Get the origin tensor `\[D\]`.
    pub fn origin(&self) -> Tensor<B, 1> {
        self.origin.clone()
    }

    /// Get the spacing tensor `\[D\]`.
    pub fn spacing(&self) -> Tensor<B, 1> {
        self.spacing.clone()
    }

    /// Get the direction tensor `\[D, D\]`.
    pub fn direction(&self) -> Tensor<B, 2> {
        self.direction.clone()
    }

    /// Get the coefficients.
    pub fn coefficients(&self) -> Tensor<B, 2> {
        self.coefficients.val().clone()
    }
}
pub(crate) mod interpolation;
pub(crate) mod mapping;
#[cfg(test)]
#[path = "tests_bspline.rs"]
mod tests;
