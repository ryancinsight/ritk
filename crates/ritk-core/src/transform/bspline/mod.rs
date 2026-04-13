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
    /// Physical origin of the grid (index 0,0,0) as tensor [D]
    origin: Tensor<B, 1>,
    /// Physical spacing between control points as tensor [D]
    spacing: Tensor<B, 1>,
    /// Physical orientation of the grid as tensor [D, D] (direction matrix)
    direction: Tensor<B, 2>,
    /// Control point displacements [num_control_points, D]
    coefficients: Param<Tensor<B, 2>>,
}

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    /// Create a new B-Spline transform.
    ///
    /// # Arguments
    /// * `grid_size` - Number of control points along each dimension
    /// * `origin` - Physical origin of the grid (position of first control point) as tensor [D]
    /// * `spacing` - Physical spacing between control points as tensor [D]
    /// * `direction` - Physical direction matrix of the grid as tensor [D, D]
    /// * `coefficients` - Initial control point displacements [num_control_points, D]
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
    /// * `origin` - Physical origin as Point<D>
    /// * `spacing` - Physical spacing as Spacing<D> (Vector<D>)
    /// * `direction` - Physical direction as Direction<D>
    /// * `coefficients` - Initial control point displacements [num_control_points, D]
    /// * `device` - Device to create tensors on
    pub fn from_spatial(
        grid_size: [usize; D],
        origin: &crate::spatial::Point<D>,
        spacing: &crate::spatial::Vector<D>,
        direction: &crate::spatial::Direction<D>,
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

    /// Get the origin tensor [D].
    pub fn origin(&self) -> Tensor<B, 1> {
        self.origin.clone()
    }

    /// Get the spacing tensor [D].
    pub fn spacing(&self) -> Tensor<B, 1> {
        self.spacing.clone()
    }

    /// Get the direction tensor [D, D].
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
mod tests {
    use super::*;
    use crate::transform::trait_::Transform;
    use burn_ndarray::NdArray;
    type TestBackend = NdArray<f32>;

    #[test]
    fn test_bspline_transform_creation() {
        let device = Default::default();
        let grid_size = [4, 4, 4];
        let origin = Tensor::<TestBackend, 1>::zeros([3], &device);
        let spacing = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0, 1.0], &device);
        let direction_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let direction = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(direction_data, Shape::new([3, 3])),
            &device,
        );

        let num_control_points = grid_size.iter().product();
        let coefficients = Tensor::<TestBackend, 2>::zeros([num_control_points, 3], &device);

        let transform = BSplineTransform::<TestBackend, 3>::new(
            grid_size,
            origin,
            spacing,
            direction,
            coefficients,
        );

        assert_eq!(transform.grid_size(), grid_size);
    }

    #[test]
    fn test_bspline_transform_from_spatial() {
        let device = Default::default();
        let grid_size = [4, 4, 4];
        let origin = crate::spatial::Point3::origin();
        let spacing = crate::spatial::Spacing3::uniform(10.0);
        let direction = crate::spatial::Direction3::identity();

        let num_control_points = grid_size.iter().product();
        let coefficients = Tensor::<TestBackend, 2>::zeros([num_control_points, 3], &device);

        let transform = BSplineTransform::<TestBackend, 3>::from_spatial(
            grid_size,
            &origin,
            &spacing,
            &direction,
            coefficients,
            &device,
        );

        assert_eq!(transform.grid_size(), grid_size);
    }

    #[test]
    fn test_bspline_transform_2d() {
        let device = Default::default();
        // 4x4 grid
        let grid_size = [4, 4];
        let origin = Tensor::<TestBackend, 1>::zeros([2], &device);
        let spacing = Tensor::<TestBackend, 1>::from_floats([10.0, 10.0], &device);
        let direction_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let direction = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(direction_data, Shape::new([2, 2])),
            &device,
        );

        let num_control_points = 16; // 4*4

        // Displace the control point at (1, 1) by (1.0, 1.0)
        // Index 5
        let mut coeffs_data = vec![0.0; num_control_points * 2];
        coeffs_data[5 * 2] = 1.0;
        coeffs_data[5 * 2 + 1] = 1.0;

        let coefficients = Tensor::from_floats(
            burn::tensor::TensorData::new(
                coeffs_data,
                burn::tensor::Shape::new([num_control_points, 2]),
            ),
            &device,
        );

        let transform = BSplineTransform::<TestBackend, 2>::new(
            grid_size,
            origin,
            spacing,
            direction,
            coefficients,
        );

        // Point at (10.0, 10.0) corresponds to index (1.0, 1.0)
        let points = Tensor::<TestBackend, 2>::from_floats([[10.0, 10.0]], &device);
        let transformed = transform.transform_points(points);
        let result = transformed.into_data().as_slice::<f32>().unwrap().to_vec();

        // Expected: 10.0 + 4/9
        let expected_disp = 4.0 / 9.0;
        assert!((result[0] - (10.0 + expected_disp)).abs() < 1e-5);
        assert!((result[1] - (10.0 + expected_disp)).abs() < 1e-5);
    }

    #[test]
    fn test_bspline_transform_out_of_bounds() {
        let device = Default::default();
        let grid_size = [4, 4];
        let origin = Tensor::<TestBackend, 1>::zeros([2], &device);
        let spacing = Tensor::<TestBackend, 1>::from_floats([10.0, 10.0], &device);
        let direction_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let direction = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(direction_data, Shape::new([2, 2])),
            &device,
        );

        let num_control_points = 16;
        let coefficients = Tensor::<TestBackend, 2>::zeros([num_control_points, 2], &device);

        let transform = BSplineTransform::<TestBackend, 2>::new(
            grid_size,
            origin,
            spacing,
            direction,
            coefficients,
        );

        // Point outside grid should remain unchanged (zero displacement)
        let points = Tensor::<TestBackend, 2>::from_floats([[100.0, 100.0]], &device);
        let transformed = transform.transform_points(points);
        let result = transformed.into_data().as_slice::<f32>().unwrap().to_vec();

        assert!((result[0] - 100.0).abs() < 1e-5);
        assert!((result[1] - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_bspline_transform_1d() {
        let device = Default::default();
        let grid_size = [4];
        let origin = Tensor::<TestBackend, 1>::zeros([1], &device);
        let spacing = Tensor::<TestBackend, 1>::from_floats([10.0], &device);
        let direction = Tensor::<TestBackend, 2>::eye(1, &device);

        let num_control_points = 4;
        let mut coeffs_data = vec![0.0; num_control_points * 1];
        // Index 1 (x=1) -> 2.0.
        coeffs_data[1] = 2.0;

        let coefficients = Tensor::from_floats(
            burn::tensor::TensorData::new(
                coeffs_data,
                burn::tensor::Shape::new([num_control_points, 1]),
            ),
            &device,
        );

        let transform = BSplineTransform::<TestBackend, 1>::new(
            grid_size,
            origin,
            spacing,
            direction,
            coefficients,
        );

        // Point at 10.0 corresponds to index 1.0.
        // B-spline basis for index 1.0: B0(1)=1/6(0)=0, B1(1)=...
        // Wait, index u=1.0. u is local.
        // grid coord = 1.0. floor=1. u=0.0.
        // Basis at u=0.0: B0=1/6, B1=4/6, B2=1/6, B3=0.
        // Indices involved: floor-1, floor, floor+1, floor+2 => 0, 1, 2, 3.
        // Coefficients: c0=0, c1=2.0, c2=0, c3=0.
        // Displacement = c0*B0 + c1*B1 + c2*B2 + c3*B3
        // = 0*1/6 + 2.0*4/6 + 0*1/6 + 0
        // = 8/6 = 4/3 = 1.3333...

        let points = Tensor::<TestBackend, 2>::from_floats([[10.0]], &device);
        let transformed = transform.transform_points(points);
        let result = transformed.into_data().as_slice::<f32>().unwrap()[0];

        assert!((result - (10.0 + 4.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_bspline_transform_4d_creation() {
        let device = Default::default();
        let grid_size = [4, 4, 4, 4];
        let origin = Tensor::<TestBackend, 1>::zeros([4], &device);
        let spacing = Tensor::<TestBackend, 1>::ones([4], &device);
        let direction = Tensor::<TestBackend, 2>::eye(4, &device);

        let num_control_points = 256;
        let coefficients = Tensor::<TestBackend, 2>::zeros([num_control_points, 4], &device);

        let transform = BSplineTransform::<TestBackend, 4>::new(
            grid_size,
            origin,
            spacing,
            direction,
            coefficients,
        );

        assert_eq!(transform.grid_size(), grid_size);

        // Basic transform test (identity)
        let points = Tensor::<TestBackend, 2>::from_floats([[1.5, 1.5, 1.5, 1.5]], &device);
        let transformed = transform.transform_points(points.clone());
        let result = transformed.sub(points).abs().sum();
        let diff = result.into_data().as_slice::<f32>().unwrap()[0];
        assert!(diff < 1e-6);
    }
}
