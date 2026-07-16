//! B-Spline transform implementation.
//!
//! This module provides a B-Spline free-form deformation transform.

use coeus_core::Backend;
use coeus_tensor::Tensor;

/// B-Spline Transform (Free-form deformation).
///
/// Uses a grid of control points to define a smooth deformation field.
/// The transform is defined by B-Spline interpolation of control point displacements.
///
/// The transform maps points in physical space to continuous indices in the control grid,
/// interpolates the displacement, and adds it to the original point.
///
/// Points outside the defined grid support (0 to grid_size-1) have zero displacement.
#[derive(Clone)]
pub struct BSplineTransform<B: Backend, const D: usize> {
    /// Control point grid dimensions
    grid_size: [usize; D],
    /// Physical origin of the grid (index 0,0,0) as tensor `[D]`
    origin: Tensor<f32, B>,
    /// Physical spacing between control points as tensor `[D]`
    spacing: Tensor<f32, B>,
    /// Physical orientation of the grid as tensor `[D, D]` (direction matrix)
    direction: Tensor<f32, B>,
    /// Control point displacements `[num_control_points, D]`
    coefficients: Tensor<f32, B>,
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
        origin: Tensor<f32, B>,
        spacing: Tensor<f32, B>,
        direction: Tensor<f32, B>,
        coefficients: Tensor<f32, B>,
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
            coefficients,
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
        coefficients: Tensor<f32, B>,
        device: &B,
    ) -> Self {
        // Convert origin to tensor
        let origin_vec: Vec<f32> = (0..D).map(|i| origin[i] as f32).collect();
        let origin_tensor = Tensor::<f32, B>::from_slice_on([D], &origin_vec, device);

        // Convert spacing to tensor
        let spacing_vec: Vec<f32> = (0..D).map(|i| spacing[i] as f32).collect();
        let spacing_tensor = Tensor::<f32, B>::from_slice_on([D], &spacing_vec, device);

        // Convert direction to tensor
        let mut dir_data = Vec::with_capacity(D * D);
        for c in 0..D {
            for r in 0..D {
                dir_data.push(direction[(r, c)] as f32);
            }
        }
        let direction_tensor = Tensor::<f32, B>::from_slice_on([D, D], &dir_data, device);

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
    pub fn origin(&self) -> Tensor<f32, B> {
        self.origin.clone()
    }

    /// Get the spacing tensor `\[D\]`.
    pub fn spacing(&self) -> Tensor<f32, B> {
        self.spacing.clone()
    }

    /// Get the direction tensor `\[D, D\]`.
    pub fn direction(&self) -> Tensor<f32, B> {
        self.direction.clone()
    }

    /// Get the coefficients.
    pub fn coefficients(&self) -> Tensor<f32, B> {
        self.coefficients.clone()
    }
}
pub(crate) mod interpolation;
pub(crate) mod mapping;
#[cfg(test)]
#[path = "tests_bspline.rs"]
mod tests;
