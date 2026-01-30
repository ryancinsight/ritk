//! B-Spline transform implementation.
//!
//! This module provides a B-Spline free-form deformation transform.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use super::trait_::Transform;

/// B-Spline Transform (Free-form deformation).
///
/// Uses a grid of control points to define a smooth deformation field.
/// The transform is defined by B-Spline interpolation of control point displacements.
#[derive(Module, Debug)]
pub struct BSplineTransform<B: Backend, const D: usize> {
    /// Control point grid dimensions
    grid_size: [usize; D],
    /// Physical size of the transform domain
    physical_size: [f64; D],
    /// Control point displacements [num_control_points, D]
    coefficients: Param<Tensor<B, 2>>,
    /// Spacing between control points
    control_point_spacing: [f64; D],
}

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    /// Create a new B-Spline transform.
    ///
    /// # Arguments
    /// * `grid_size` - Number of control points along each dimension
    /// * `physical_size` - Physical size of the transform domain along each dimension
    /// * `coefficients` - Initial control point displacements [num_control_points, D]
    pub fn new(
        grid_size: [usize; D],
        physical_size: [f64; D],
        coefficients: Tensor<B, 2>,
    ) -> Self {
        let control_point_spacing: [f64; D] = std::array::from_fn(|i| {
            physical_size[i] / (grid_size[i] - 1) as f64
        });

        Self {
            grid_size,
            physical_size,
            coefficients: Param::from_tensor(coefficients),
            control_point_spacing,
        }
    }

    /// Get the grid size.
    pub fn grid_size(&self) -> [usize; D] {
        self.grid_size
    }

    /// Get the physical size.
    pub fn physical_size(&self) -> [f64; D] {
        self.physical_size
    }

    /// Get the control point spacing.
    pub fn control_point_spacing(&self) -> [f64; D] {
        self.control_point_spacing
    }

    /// Get the coefficients.
    pub fn coefficients(&self) -> Tensor<B, 2> {
        self.coefficients.val().clone()
    }
    
    /// Compute Cubic B-Spline basis functions.
    fn bspline_basis(u: Tensor<B, 1>) -> [Tensor<B, 1>; 4] {
        // u is in [0, 1)
        let one = 1.0;
        let two = 2.0;
        let three = 3.0;
        let four = 4.0;
        let six = 6.0;
        
        // B0 = (1-u)^3 / 6
        let one_minus_u = u.clone().neg().add_scalar(one);
        let b0 = one_minus_u.powf_scalar(three) / six;
        
        // B1 = (3u^3 - 6u^2 + 4) / 6
        let u2 = u.clone().powf_scalar(two);
        let u3 = u.clone().powf_scalar(three);
        let b1 = (u3.clone().mul_scalar(three) - u2.clone().mul_scalar(six)).add_scalar(four) / six;
        
        // B2 = (-3u^3 + 3u^2 + 3u + 1) / 6
        let b2 = (u3.clone().mul_scalar(-three) + u2.clone().mul_scalar(three) + u.clone().mul_scalar(three)).add_scalar(one) / six;
        
        // B3 = u^3 / 6
        let b3 = u3 / six;
        
        [b0, b1, b2, b3]
    }
    
    /// Compute Cubic B-Spline basis functions and stack them into a tensor [Batch, 4].
    fn compute_basis_tensor(u: Tensor<B, 1>) -> Tensor<B, 2> {
        let [b0, b1, b2, b3] = Self::bspline_basis(u);
        // Stack along dim 1: [Batch, 1] -> [Batch, 4]
        Tensor::cat(vec![
            b0.unsqueeze_dim::<2>(1),
            b1.unsqueeze_dim::<2>(1),
            b2.unsqueeze_dim::<2>(1),
            b3.unsqueeze_dim::<2>(1)
        ], 1)
    }
    
    fn transform_2d(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let batch_size = points.shape().dims[0];
        
        // Grid spacing [2]
        let spacing_tensor = Tensor::<B, 1>::from_floats([self.control_point_spacing[0] as f32, self.control_point_spacing[1] as f32], &device).reshape([1, 2]);
        let grid_coords = points.clone() / spacing_tensor; // [Batch, 2]
        
        let grid_indices_float = grid_coords.clone().floor(); // [Batch, 2]
        let u_vec = grid_coords - grid_indices_float.clone(); // [Batch, 2]
        
        let base_index = grid_indices_float.int() - 1; // [Batch, 2]
        
        let ux = u_vec.clone().slice([0..batch_size, 0..1]).squeeze(1); // [Batch]
        let uy = u_vec.clone().slice([0..batch_size, 1..2]).squeeze(1);
        
        let bx = Self::compute_basis_tensor(ux); // [Batch, 4]
        let by = Self::compute_basis_tensor(uy); // [Batch, 4]
        
        // Weights: [Batch, 4, 4] -> Flatten to [Batch, 16]
        // W[b, i, j] = Bx[b, i] * By[b, j]
        // [Batch, 4, 1] * [Batch, 1, 4] -> [Batch, 4, 4]
        let weights = bx.unsqueeze_dim::<3>(2) * by.unsqueeze_dim::<3>(1);
        let weights = weights.reshape([batch_size, 16, 1]); // [Batch, 16, 1] for broadcasting
        
        let nx = self.grid_size[0] as i32;
        let ny = self.grid_size[1] as i32;
        
        // Create range [0, 1, 2, 3]
        let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);
        
        // Indices construction
        // i_idx: [1, 4, 1] (broadcast over j)
        let i_idx = range.clone().reshape([1, 4, 1]);
        // j_idx: [1, 1, 4] (broadcast over i)
        let j_idx = range.clone().reshape([1, 1, 4]);
        
        // base_x: [Batch, 1] -> [Batch, 1, 1]
        let base_x = base_index.clone().slice([0..batch_size, 0..1]).unsqueeze_dim::<3>(2);
        let base_y = base_index.clone().slice([0..batch_size, 1..2]).unsqueeze_dim::<3>(2); // Note: unsqueeze dim 2 to match [Batch, 1, 1]
        
        // idx_x: [Batch, 4, 1] (constant along j)
        let idx_x = base_x + i_idx;
        // idx_y: [Batch, 1, 4] (constant along i)
        let idx_y = base_y + j_idx;
        
        // Broadcast to [Batch, 4, 4] and flatten
        // We use addition with zeros to broadcast, as explicit broadcast_to/expand might vary by version
        let zeros = Tensor::<B, 3, burn::tensor::Int>::zeros([1, 4, 4], &device);
        
        let idx_x_flat = (idx_x + zeros.clone()).reshape([batch_size, 16]);
        let idx_y_flat = (idx_y + zeros.clone()).reshape([batch_size, 16]);
        
        let idx_x_clamped = idx_x_flat.clamp(0, nx - 1);
        let idx_y_clamped = idx_y_flat.clamp(0, ny - 1);
        
        // Flat index = y * Nx + x
        let flat_indices = idx_y_clamped * nx + idx_x_clamped; // [Batch, 16]
        
        // Gather coefficients
        // Flatten batch dim: [Batch * 16]
        let gather_indices = flat_indices.reshape([batch_size * 16]);
        
        let coeffs = self.coefficients.val().clone().select(0, gather_indices); // [Batch*16, 2]
        
        let coeffs = coeffs.reshape([batch_size, 16, 2]);
        
        // Weighted sum: sum(coeffs * weights, dim=1)
        // [Batch, 16, 2] * [Batch, 16, 1] -> [Batch, 16, 2] -> sum -> [Batch, 1, 2] -> squeeze -> [Batch, 2]
        let displacement = (coeffs * weights).sum_dim(1).squeeze(1);
        
        points + displacement
    }

    fn transform_3d(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let batch_size = points.shape().dims[0];
        
        // Grid spacing [3]
        let spacing_tensor = Tensor::<B, 1>::from_floats([
            self.control_point_spacing[0] as f32, 
            self.control_point_spacing[1] as f32,
            self.control_point_spacing[2] as f32
        ], &device).reshape([1, 3]);
        
        let grid_coords = points.clone() / spacing_tensor; // [Batch, 3]
        
        let grid_indices_float = grid_coords.clone().floor(); // [Batch, 3]
        let u_vec = grid_coords - grid_indices_float.clone(); // [Batch, 3]
        
        let base_index = grid_indices_float.int() - 1; // [Batch, 3]
        
        let ux = u_vec.clone().slice([0..batch_size, 0..1]).squeeze(1); // [Batch]
        let uy = u_vec.clone().slice([0..batch_size, 1..2]).squeeze(1);
        let uz = u_vec.clone().slice([0..batch_size, 2..3]).squeeze(1);
        
        let bx = Self::compute_basis_tensor(ux); // [Batch, 4]
        let by = Self::compute_basis_tensor(uy); // [Batch, 4]
        let bz = Self::compute_basis_tensor(uz); // [Batch, 4]
        
        // Weights: [Batch, 4, 4, 4] -> Flatten to [Batch, 64]
        // W[b, i, j, k] = Bx[b, i] * By[b, j] * Bz[b, k]
        // [Batch, 4, 1, 1] * [Batch, 1, 4, 1] * [Batch, 1, 1, 4]
        let weights = bx.unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3) * 
                      by.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(3) * 
                      bz.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
                      
        let weights = weights.reshape([batch_size, 64, 1]); // [Batch, 64, 1]
        
        let nx = self.grid_size[0] as i32;
        let ny = self.grid_size[1] as i32;
        let nz = self.grid_size[2] as i32;
        
        // Range [0, 1, 2, 3]
        let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);
        
        let i_idx = range.clone().reshape([1, 4, 1, 1]);
        let j_idx = range.clone().reshape([1, 1, 4, 1]);
        let k_idx = range.clone().reshape([1, 1, 1, 4]);
        
        let base_x = base_index.clone().slice([0..batch_size, 0..1]).unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3); // [Batch, 1, 1, 1]
        let base_y = base_index.clone().slice([0..batch_size, 1..2]).unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);
        let base_z = base_index.clone().slice([0..batch_size, 2..3]).unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);
        
        let idx_x = base_x + i_idx; // [Batch, 4, 1, 1]
        let idx_y = base_y + j_idx; // [Batch, 1, 4, 1]
        let idx_z = base_z + k_idx; // [Batch, 1, 1, 4]
        
        // Broadcast via addition with zeros
        let zeros = Tensor::<B, 4, burn::tensor::Int>::zeros([1, 4, 4, 4], &device);
        
        let idx_x_flat = (idx_x + zeros.clone()).reshape([batch_size, 64]);
        let idx_y_flat = (idx_y + zeros.clone()).reshape([batch_size, 64]);
        let idx_z_flat = (idx_z + zeros.clone()).reshape([batch_size, 64]);
        
        let idx_x_clamped = idx_x_flat.clamp(0, nx - 1);
        let idx_y_clamped = idx_y_flat.clamp(0, ny - 1);
        let idx_z_clamped = idx_z_flat.clamp(0, nz - 1);
        
        // Flat index = z * (Nx * Ny) + y * Nx + x
        let stride_z = nx * ny;
        let stride_y = nx;
        
        let flat_indices = idx_z_clamped * stride_z + idx_y_clamped * stride_y + idx_x_clamped; // [Batch, 64]
        
        let gather_indices = flat_indices.reshape([batch_size * 64]);
        
        let coeffs = self.coefficients.val().clone().select(0, gather_indices); // [Batch*64, 3]
        
        let coeffs = coeffs.reshape([batch_size, 64, 3]);
        
        let displacement = (coeffs * weights).sum_dim(1).squeeze(1);
        
        points + displacement
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for BSplineTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        if D == 2 {
            self.transform_2d(points)
        } else if D == 3 {
            self.transform_3d(points)
        } else {
            panic!("BSplineTransform only supports 2D and 3D");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_bspline_transform_creation() {
        let device = Default::default();
        let grid_size = [4, 4, 4];
        let physical_size = [100.0, 100.0, 100.0];
        let num_control_points = grid_size.iter().product();
        let coefficients = Tensor::<TestBackend, 2>::zeros([num_control_points, 3], &device);

        let transform = BSplineTransform::<TestBackend, 3>::new(
            grid_size,
            physical_size,
            coefficients,
        );

        assert_eq!(transform.grid_size(), grid_size);
        assert_eq!(transform.physical_size(), physical_size);
    }

    #[test]
    fn test_bspline_transform_2d() {
        let device = Default::default();
        // 4x4 grid
        let grid_size = [4, 4];
        // 30x30 physical size -> spacing = 10.0
        let physical_size = [30.0, 30.0];
        let num_control_points = 16; // 4*4
        
        // Displace the control point at (1, 1) by (1.0, 1.0)
        // Index of (1, 1) in 4x4 grid is 1*4 + 1 = 5
        // We need to create a new tensor with this value
        let mut coeffs_data = vec![0.0; num_control_points * 2];
        coeffs_data[5 * 2] = 1.0;     // x displacement
        coeffs_data[5 * 2 + 1] = 1.0; // y displacement
        
        let coefficients = Tensor::from_floats(
            burn::tensor::TensorData::new(coeffs_data, burn::tensor::Shape::new([num_control_points, 2])), 
            &device
        );

        let transform = BSplineTransform::<TestBackend, 2>::new(
            grid_size,
            physical_size,
            coefficients,
        );
        
        // Point at (10.0, 10.0) corresponds to index (1.0, 1.0) exactly.
        // At exact control point location, the displacement should be dominated by that control point.
        // B-spline basis at u=0 is B0(0)=1/6, B1(0)=4/6, B2(0)=1/6, B3(0)=0.
        // Wait, the basis functions are for the interval [i, i+1).
        // If we are exactly at a control point index, say 1.0.
        // The floor is 1. u = 0.
        // The 4x4 neighborhood starts at 1-1 = 0.
        // So indices are 0, 1, 2, 3.
        // u=0. Basis values: B0=1/6, B1=4/6, B2=1/6, B3=0.
        // Weights:
        // i=0 (idx=0): B0(0) = 1/6
        // i=1 (idx=1): B1(0) = 4/6
        // i=2 (idx=2): B2(0) = 1/6
        // i=3 (idx=3): B3(0) = 0
        
        // For 2D, weight is Bx * By.
        // At (1,1), contribution from (1,1) (which is i=1, j=1 in the window starting at (0,0))
        // Weight = B1(0) * B1(0) = (2/3) * (2/3) = 4/9 ≈ 0.444
        
        // So displacement should be approx 0.444 * (1.0, 1.0).
        
        let points = Tensor::<TestBackend, 2>::from_floats([[10.0, 10.0]], &device);
        let transformed = transform.transform_points(points);
        let transformed_data = transformed.into_data();
        let result = transformed_data.as_slice::<f32>().unwrap();
        
        // Expected: 10.0 + 0.444..., 10.0 + 0.444...
        let expected_disp = 4.0 / 9.0;
        
        println!("Result: {:?}", result);
        
        assert!((result[0] - (10.0 + expected_disp)).abs() < 1e-5);
        assert!((result[1] - (10.0 + expected_disp)).abs() < 1e-5);
    }

    #[test]
    fn test_bspline_transform_3d() {
        let device = Default::default();
        // 4x4x4 grid
        let grid_size = [4, 4, 4];
        // 30x30x30 physical size -> spacing = 10.0
        let physical_size = [30.0, 30.0, 30.0];
        let num_control_points = 4 * 4 * 4; // 64
        
        // Displace the control point at (1, 1, 1) by (1.0, 1.0, 1.0)
        // Index of (1, 1, 1) in 4x4x4 grid
        // flat_idx = z * (Nx * Ny) + y * Nx + x
        // flat_idx = 1 * (4 * 4) + 1 * 4 + 1 = 16 + 4 + 1 = 21
        
        let mut coeffs_data = vec![0.0; num_control_points * 3];
        coeffs_data[21 * 3] = 1.0;     // x displacement
        coeffs_data[21 * 3 + 1] = 1.0; // y displacement
        coeffs_data[21 * 3 + 2] = 1.0; // z displacement
        
        let coefficients = Tensor::from_floats(
            burn::tensor::TensorData::new(coeffs_data, burn::tensor::Shape::new([num_control_points, 3])), 
            &device
        );

        let transform = BSplineTransform::<TestBackend, 3>::new(
            grid_size,
            physical_size,
            coefficients,
        );
        
        // Point at (10.0, 10.0, 10.0) corresponds to index (1.0, 1.0, 1.0)
        let points = Tensor::<TestBackend, 2>::from_floats([[10.0, 10.0, 10.0]], &device);
        let transformed = transform.transform_points(points);
        let transformed_data = transformed.into_data();
        let result = transformed_data.as_slice::<f32>().unwrap();
        
        // Expected displacement weight: (2/3)^3 = 8/27
        let weight = 8.0 / 27.0;
        
        println!("Result: {:?}", result);
        
        assert!((result[0] - (10.0 + weight)).abs() < 1e-5);
        assert!((result[1] - (10.0 + weight)).abs() < 1e-5);
        assert!((result[2] - (10.0 + weight)).abs() < 1e-5);
    }
}
