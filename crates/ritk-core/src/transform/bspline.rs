//! B-Spline transform implementation.
//!
//! This module provides a B-Spline free-form deformation transform.

use burn::tensor::{Tensor, TensorData, Shape};
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use super::trait_::Transform;

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
        assert!(grid_size.iter().all(|&x| x >= 4), "BSpline grid size must be at least 4 in all dimensions to support cubic B-splines");

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
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, Shape::new([D])),
            device,
        );

        // Convert spacing to tensor
        let spacing_vec: Vec<f32> = (0..D).map(|i| spacing[i] as f32).collect();
        let spacing_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(spacing_vec, Shape::new([D])),
            device,
        );

        // Convert direction to tensor
        let mut dir_data = Vec::with_capacity(D * D);
        for c in 0..D {
            for r in 0..D {
                dir_data.push(direction[(r, c)] as f32);
            }
        }
        let direction_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(dir_data, Shape::new([D, D])),
            device,
        );

        Self::new(grid_size, origin_tensor, spacing_tensor, direction_tensor, coefficients)
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

    /// Compute inverse direction matrix as a flat vector of f32 values.
    ///
    /// Returns the flattened inverse direction matrix for index calculation.
    fn compute_inverse_direction(&self, _device: &<B as Backend>::Device) -> Vec<f32> {
        // Get direction matrix data
        let dir_tensor = self.direction.clone();
        let dir_data_result = dir_tensor.to_data();
        let dir_slice = dir_data_result.as_slice::<f32>().unwrap();
        
        // For a D x D matrix, compute inverse
        // For small matrices (2x2, 3x3), we can compute the inverse analytically
        match D {
            2 => {
                let a = dir_slice[0];
                let b = dir_slice[1];
                let c = dir_slice[2];
                let d = dir_slice[3];
                
                let det = a * d - b * c;
                if det.abs() < 1e-10 {
                    // Singular matrix - return identity
                    return vec![1.0, 0.0, 0.0, 1.0];
                }
                
                let inv_det = 1.0 / det;
                vec![
                    d * inv_det, -b * inv_det,
                    -c * inv_det, a * inv_det,
                ]
            }
            3 => {
                let a = dir_slice[0];
                let b = dir_slice[1];
                let c = dir_slice[2];
                let d = dir_slice[3];
                let e = dir_slice[4];
                let f = dir_slice[5];
                let g = dir_slice[6];
                let h = dir_slice[7];
                let i = dir_slice[8];
                
                // Compute minors and cofactors
                let m11 = e * i - f * h;
                let m12 = -(d * i - f * g);
                let m13 = d * h - e * g;
                let m21 = -(b * i - c * h);
                let m22 = a * i - c * g;
                let m23 = -(a * h - b * g);
                let m31 = b * f - c * e;
                let m32 = -(a * f - c * d);
                let m33 = a * e - b * d;
                
                let det = a * m11 + b * m12 + c * m13;
                if det.abs() < 1e-10 {
                    // Singular matrix - return identity
                    return vec![
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0,
                    ];
                }
                
                let inv_det = 1.0 / det;
                vec![
                    m11 * inv_det, m12 * inv_det, m13 * inv_det,
                    m21 * inv_det, m22 * inv_det, m23 * inv_det,
                    m31 * inv_det, m32 * inv_det, m33 * inv_det,
                ]
            }
            _ => {
                // For other sizes, return pseudo-identity
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
    fn world_to_grid_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
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
        
        let t_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(t_data, Shape::new([D, D])),
            &device,
        );
        
        // Apply transform: (points - origin) @ T
        let diff = points - origin_tensor;
        diff.matmul(t_tensor)
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
        let batch_size = points.shape().dims[0];
        const CHUNK_SIZE: usize = 32768;

        if batch_size <= CHUNK_SIZE {
            self.transform_2d_chunk(points)
        } else {
            let num_chunks = (batch_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut chunks = Vec::with_capacity(num_chunks);
            
            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, batch_size);
                
                let chunk_points = points.clone().slice([start..end]);
                let chunk_result = self.transform_2d_chunk(chunk_points);
                chunks.push(chunk_result);
            }
            Tensor::cat(chunks, 0)
        }
    }

    fn transform_2d_chunk(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let batch_size = points.shape().dims[0];
        
        // Convert physical points to grid indices
        let grid_coords = self.world_to_grid_tensor(points.clone()); // [Batch, 2]
        
        // Compute Mask for Out-of-Bounds
        // Valid if 0 <= coord <= grid_size - 1
        let zero_tensor = Tensor::<B, 1>::zeros([2], &device).reshape([1, 2]);
        let size_tensor = Tensor::<B, 1>::from_floats(
            [self.grid_size[0] as f32 - 1.0, self.grid_size[1] as f32 - 1.0], 
            &device
        ).reshape([1, 2]);
        
        let mask_ge_zero = grid_coords.clone().greater_equal(zero_tensor);
        let mask_le_size = grid_coords.clone().lower_equal(size_tensor);
        let mask = mask_ge_zero.equal(mask_le_size);
        
        // Reduce mask along dim 1 (both x and y must be valid)
        let mask_float = mask.float();
        let valid_mask = mask_float.sum_dim(1).equal_elem(2.0).float(); // [Batch, 1] (1.0 if valid, 0.0 else)
        
        // Interpolation Logic
        let grid_indices_float = grid_coords.clone().floor(); // [Batch, 2]
        let u_vec = grid_coords - grid_indices_float.clone(); // [Batch, 2]
        
        let base_index = grid_indices_float.int() - 1; // [Batch, 2]
        
        let ux = u_vec.clone().slice([0..batch_size, 0..1]).squeeze(1); // [Batch]
        let uy = u_vec.clone().slice([0..batch_size, 1..2]).squeeze(1); // [Batch]
        
        let bx = Self::compute_basis_tensor(ux); // [Batch, 4]
        let by = Self::compute_basis_tensor(uy); // [Batch, 4]
        
        // Weights: [Batch, 4, 4] -> Flatten to [Batch, 16]
        let weights = bx.unsqueeze_dim::<3>(2) * by.unsqueeze_dim::<3>(1);
        let weights = weights.reshape([batch_size, 16, 1]); // [Batch, 16, 1]
        
        let nx = self.grid_size[0] as i32;
        let ny = self.grid_size[1] as i32;
        
        // Create range [0, 1, 2, 3]
        let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);
        
        // Indices construction
        let i_idx = range.clone().reshape([1, 4, 1]);
        let j_idx = range.clone().reshape([1, 1, 4]);
        
        let base_x = base_index.clone().slice([0..batch_size, 0..1]).unsqueeze_dim::<3>(2);
        let base_y = base_index.clone().slice([0..batch_size, 1..2]).unsqueeze_dim::<3>(2);
        
        let idx_x = base_x + i_idx;
        let idx_y = base_y + j_idx;
        
        // Broadcast
        let zeros = Tensor::<B, 3, burn::tensor::Int>::zeros([1, 4, 4], &device);
        
        let idx_x_flat = (idx_x + zeros.clone()).reshape([batch_size, 16]);
        let idx_y_flat = (idx_y + zeros.clone()).reshape([batch_size, 16]);
        
        // Clamp indices to valid grid range for lookups
        let idx_x_clamped = idx_x_flat.clamp(0, nx - 1);
        let idx_y_clamped = idx_y_flat.clamp(0, ny - 1);
        
        // Flat index = y * Nx + x
        let flat_indices = idx_y_clamped * nx + idx_x_clamped; // [Batch, 16]
        
        let gather_indices = flat_indices.reshape([batch_size * 16]);
        let coeffs = self.coefficients.val().clone().select(0, gather_indices); // [Batch*16, 2]
        let coeffs = coeffs.reshape([batch_size, 16, 2]);
        
        let displacement = (coeffs * weights).sum_dim(1).squeeze(1); // [Batch, 2]
        
        // Apply Mask (Zero displacement if out of bounds)
        let masked_displacement = displacement * valid_mask;
        
        points + masked_displacement
    }

    fn transform_3d(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch_size = points.shape().dims[0];
        const CHUNK_SIZE: usize = 32768;

        if batch_size <= CHUNK_SIZE {
            self.transform_3d_chunk(points)
        } else {
            let num_chunks = (batch_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut chunks = Vec::with_capacity(num_chunks);
            
            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, batch_size);
                
                let chunk_points = points.clone().slice([start..end]);
                let chunk_result = self.transform_3d_chunk(chunk_points);
                chunks.push(chunk_result);
            }
            Tensor::cat(chunks, 0)
        }
    }

    fn transform_3d_chunk(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let batch_size = points.shape().dims[0];
        
        // Convert physical points to grid indices
        let grid_coords = self.world_to_grid_tensor(points.clone()); // [Batch, 3]
        
        // Compute Mask for Out-of-Bounds
        let zero_tensor = Tensor::<B, 1>::zeros([3], &device).reshape([1, 3]);
        let size_tensor = Tensor::<B, 1>::from_floats(
            [self.grid_size[0] as f32 - 1.0, self.grid_size[1] as f32 - 1.0, self.grid_size[2] as f32 - 1.0], 
            &device
        ).reshape([1, 3]);
        
        let mask_ge_zero = grid_coords.clone().greater_equal(zero_tensor);
        let mask_le_size = grid_coords.clone().lower_equal(size_tensor);
        let mask = mask_ge_zero.equal(mask_le_size);
        
        let mask_float = mask.float();
        let valid_mask = mask_float.sum_dim(1).equal_elem(3.0).float(); // [Batch, 1]
        
        // Interpolation Logic
        let grid_indices_float = grid_coords.clone().floor(); // [Batch, 3]
        let u_vec = grid_coords - grid_indices_float.clone(); // [Batch, 3]
        
        let base_index = grid_indices_float.int() - 1; // [Batch, 3]
        
        let ux = u_vec.clone().slice([0..batch_size, 0..1]).squeeze(1);
        let uy = u_vec.clone().slice([0..batch_size, 1..2]).squeeze(1);
        let uz = u_vec.clone().slice([0..batch_size, 2..3]).squeeze(1);
        
        let bx = Self::compute_basis_tensor(ux);
        let by = Self::compute_basis_tensor(uy);
        let bz = Self::compute_basis_tensor(uz);
        
        // Weights: [Batch, 4, 4, 4] -> Flatten to [Batch, 64]
        let weights = bx.unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3) * 
                      by.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(3) * 
                      bz.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
                      
        let weights = weights.reshape([batch_size, 64, 1]);
        
        let nx = self.grid_size[0] as i32;
        let ny = self.grid_size[1] as i32;
        let nz = self.grid_size[2] as i32;
        
        let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);
        
        let i_idx = range.clone().reshape([1, 4, 1, 1]);
        let j_idx = range.clone().reshape([1, 1, 4, 1]);
        let k_idx = range.clone().reshape([1, 1, 1, 4]);
        
        let base_x = base_index.clone().slice([0..batch_size, 0..1]).unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);
        let base_y = base_index.clone().slice([0..batch_size, 1..2]).unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);
        let base_z = base_index.clone().slice([0..batch_size, 2..3]).unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);
        
        let idx_x = base_x + i_idx;
        let idx_y = base_y + j_idx;
        let idx_z = base_z + k_idx;
        
        let zeros = Tensor::<B, 4, burn::tensor::Int>::zeros([1, 4, 4, 4], &device);
        
        let idx_x_flat = (idx_x + zeros.clone()).reshape([batch_size, 64]);
        let idx_y_flat = (idx_y + zeros.clone()).reshape([batch_size, 64]);
        let idx_z_flat = (idx_z + zeros.clone()).reshape([batch_size, 64]);
        
        let idx_x_clamped = idx_x_flat.clamp(0, nx - 1);
        let idx_y_clamped = idx_y_flat.clamp(0, ny - 1);
        let idx_z_clamped = idx_z_flat.clamp(0, nz - 1);
        
        let stride_z = nx * ny;
        let stride_y = nx;
        
        let flat_indices = idx_z_clamped * stride_z + idx_y_clamped * stride_y + idx_x_clamped;
        
        let gather_indices = flat_indices.reshape([batch_size * 64]);
        let coeffs = self.coefficients.val().clone().select(0, gather_indices); // [Batch*64, 3]
        let coeffs = coeffs.reshape([batch_size, 64, 3]);
        
        let displacement = (coeffs * weights).sum_dim(1).squeeze(1);
        
        // Apply Mask
        let masked_displacement = displacement * valid_mask;
        
        points + masked_displacement
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
        let origin = Tensor::<TestBackend, 1>::zeros([3], &device);
        let spacing = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0, 1.0], &device);
        let direction_data: Vec<f32> = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
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
        let direction_data: Vec<f32> = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
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
            burn::tensor::TensorData::new(coeffs_data, burn::tensor::Shape::new([num_control_points, 2])), 
            &device
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
        let direction_data: Vec<f32> = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
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
}
