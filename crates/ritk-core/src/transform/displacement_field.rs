//! Displacement field transform implementation.
//!
//! This module implements a dense displacement field transform, where a vector
//! is stored at each discrete location in a grid. The transform maps points
//! by adding the interpolated displacement vector to the input point.
//!
//! $T(x) = x + D(x)$

use burn::tensor::{Tensor, TensorData, Shape};
use burn::tensor::backend::Backend;
use burn::module::Module;
use crate::spatial::{Point, Spacing, Direction};
use crate::transform::{Transform, Resampleable};
use crate::interpolation::{Interpolator, LinearInterpolator};

/// Displacement field data representing a vector field on a regular grid.
///
/// This struct holds the displacement vectors and their physical space metadata.
/// It is essentially a vector-valued image.
#[derive(Module, Debug)]
pub struct DisplacementField<B: Backend, const D: usize> {
    /// Vector components stored as separate scalar tensors.
    /// Length must be equal to D.
    /// Each tensor has shape corresponding to the spatial dimensions (e.g., [Z, Y, X] for 3D).
    components: Vec<Tensor<B, D>>,
    /// Physical coordinate of the first pixel (index 0,0,0).
    origin: Point<D>,
    /// Physical distance between pixels along each axis.
    spacing: Spacing<D>,
    /// Orientation of the image axes.
    direction: Direction<D>,
    
    // Precomputed matrices for coordinate transformation
    world_to_index_matrix: Tensor<B, 2>,
    origin_tensor: Tensor<B, 2>,
}

impl<B: Backend, const D: usize> DisplacementField<B, D> {
    /// Create a new displacement field.
    ///
    /// # Arguments
    /// * `components` - Vector components. Must have length D.
    /// * `origin` - Physical origin.
    /// * `spacing` - Physical spacing.
    /// * `direction` - Physical direction matrix.
    pub fn new(
        components: Vec<Tensor<B, D>>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        assert_eq!(components.len(), D, "Number of components must match dimensionality");
        // Verify all components have same shape
        if !components.is_empty() {
            let shape = components[0].shape();
            for c in &components[1..] {
                assert_eq!(c.shape(), shape, "All components must have the same shape");
            }
        }
        
        let device = components[0].device();

        // Components are stored as tensors directly.
        // This allows creating DisplacementField from computed tensors (non-leaf).

        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| origin[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, Shape::new([D])),
            &device,
        ).reshape([1, D]);
            
        // 2. Prepare Transform Matrix T = (S^-1 * D^-1)^T
        // Invert direction matrix
        let inv_dir_vec = match D {
            2 => {
                let m = nalgebra::Matrix2::new(
                    direction[(0, 0)], direction[(0, 1)],
                    direction[(1, 0)], direction[(1, 1)]
                );
                let inv = m.try_inverse().expect("Direction matrix must be invertible");
                vec![inv[(0,0)], inv[(0,1)], inv[(1,0)], inv[(1,1)]]
            },
            3 => {
                let m = nalgebra::Matrix3::new(
                    direction[(0, 0)], direction[(0, 1)], direction[(0, 2)],
                    direction[(1, 0)], direction[(1, 1)], direction[(1, 2)],
                    direction[(2, 0)], direction[(2, 1)], direction[(2, 2)]
                );
                let inv = m.try_inverse().expect("Direction matrix must be invertible");
                vec![
                    inv[(0,0)], inv[(0,1)], inv[(0,2)],
                    inv[(1,0)], inv[(1,1)], inv[(1,2)],
                    inv[(2,0)], inv[(2,1)], inv[(2,2)]
                ]
            },
            _ => panic!("DisplacementField only supports 2D and 3D"),
        };

        let mut t_data = Vec::with_capacity(D * D);
        for r in 0..D {
            for c in 0..D {
                // T[r, c] uses inv_dir[c, r] and spacing[c]
                let inv_dir_val = inv_dir_vec[c * D + r];
                let spacing_val = spacing[c];
                let val = (inv_dir_val / spacing_val) as f32;
                t_data.push(val);
            }
        }
        
        let world_to_index_matrix = Tensor::<B, 2>::from_data(
            TensorData::new(t_data, Shape::new([D, D])),
            &device,
        );

        Self {
            components,
            origin,
            spacing,
            direction,
            world_to_index_matrix,
            origin_tensor,
        }
    }

    /// Get the components.
    pub fn components(&self) -> Vec<Tensor<B, D>> {
        self.components.clone()
    }

    /// Get the origin.
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Get the spacing.
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Get the direction.
    pub fn direction(&self) -> Direction<D> {
        self.direction
    }

    /// Resample the displacement field to a new grid.
    ///
    /// # Arguments
    /// * `new_shape` - The new grid shape.
    /// * `new_origin` - The new physical origin.
    /// * `new_spacing` - The new physical spacing.
    /// * `new_direction` - The new physical direction.
    ///
    /// # Returns
    /// A new DisplacementField with resampled components.
    pub fn resample(
        &self,
        new_shape: [usize; D],
        new_origin: Point<D>,
        new_spacing: Spacing<D>,
        new_direction: Direction<D>,
    ) -> Self {
        let device = self.components[0].device();
        let interpolator = LinearInterpolator::new();

        // 1. Generate new grid indices [N, D]
        let new_indices = crate::image::grid::generate_grid(new_shape, &device);
        
        // 2. Prepare transform matrix for new grid index -> world
        // Matrix M[k, c] = Direction[c, k] * Spacing[k]
        let mut m_data = Vec::with_capacity(D * D);
        for k in 0..D { // Input dimension (index)
            for c in 0..D { // Output dimension (physical)
                let val = (new_direction[(c, k)] * new_spacing[k]) as f32;
                m_data.push(val);
            }
        }
        let m_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(m_data, Shape::new([D, D])),
            &device
        );
        
        let origin_vec: Vec<f32> = (0..D).map(|i| new_origin[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, Shape::new([D])),
            &device
        ).reshape([1, D]);
        
        // 3. Process in chunks
        let [n_points, _] = new_indices.dims();
        const CHUNK_SIZE: usize = 32768;
        
        let mut component_chunks: Vec<Vec<Tensor<B, 1>>> = vec![Vec::new(); D];
        
        let num_chunks = (n_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        for i in 0..num_chunks {
            let start = i * CHUNK_SIZE;
            let end = std::cmp::min(start + CHUNK_SIZE, n_points);
            let chunk_indices = new_indices.clone().slice([start..end]);
            
            // World = Origin + Indices @ M
            let offset = chunk_indices.matmul(m_tensor.clone());
            let world = offset + origin_tensor.clone();
            
            // Convert to old index space
            let old_indices = self.world_to_index_tensor(world);
            
            // Interpolate each component
            for d in 0..D {
                let comp = &self.components[d];
                let val = interpolator.interpolate(comp, old_indices.clone());
                component_chunks[d].push(val);
            }
        }
        
        // 4. Concat and reshape
        let mut final_components = Vec::with_capacity(D);
        for d in 0..D {
            let flat = Tensor::cat(component_chunks[d].clone(), 0);
            let reshaped = flat.reshape(Shape::new(new_shape));
            final_components.push(reshaped);
        }
        
        Self::new(final_components, new_origin, new_spacing, new_direction)
    }

    /// Compose this displacement field with another one.
    /// Result = self o other.
    /// T(x) = self(other(x)) = other(x) + self(other(x))
    /// The resulting field is defined on the grid of `other`.
    pub fn compose(&self, other: &Self) -> Self {
        let device = other.components[0].device();
        // Burn 0.16 Shape.dims is Vec<usize>, convert to array [usize; D]
        let shape_vec = other.components[0].shape().dims;
        let shape_dims: [usize; D] = shape_vec.try_into().expect("Dimension mismatch");
        
        // 1. Generate grid indices [N, D]
        let indices = crate::image::grid::generate_grid(shape_dims, &device);
        let [n_points, _] = indices.dims();

        // 2. Compute Physical Points P = Origin + Indices @ (D * S)^T
        // We can compute M = (D * S)^T on CPU and upload
        let m_data = match D {
            2 => {
                // D[c, r] * S[r]
                // But we want (D*S)^T.
                // (D*S)[r, c] = D[r, c] * S[c]? No.
                // P = O + D * S * I.
                // D is matrix, S is diagonal. I is vector.
                // P = O + D * (S * I).
                // With row vectors: P = O + (I * S) * D^T.
                // Let's verify scaling.
                // I * S (elementwise) scales each coord.
                // Then matmul D^T.
                // Or just construct M = diag(S) * D^T.
                // M[c, r] = S[c] * D[r, c].
                let sx = other.spacing[0]; // x
                let sy = other.spacing[1]; // y
                let d00 = other.direction[(0,0)];
                let d01 = other.direction[(0,1)];
                let d10 = other.direction[(1,0)];
                let d11 = other.direction[(1,1)];
                
                // M = [sx*d00, sx*d10]
                //     [sy*d01, sy*d11]
                // Row 0: sx*d00, sx*d10 (x contrib to x, y)
                // Row 1: sy*d01, sy*d11
                // Wait.
                // I = [x, y].
                // P = [x', y'].
                // P = x * col0(M) + y * col1(M).
                // P = x * (S[0]*col0(D)) + y * (S[1]*col1(D)).
                // So Row 0 of M should be S[0]*col0(D)^T = S[0]*row0(D^T).
                // M[0, 0] = sx * d00
                // M[0, 1] = sx * d10
                // M[1, 0] = sy * d01
                // M[1, 1] = sy * d11
                vec![sx*d00, sx*d10, sy*d01, sy*d11]
            },
            3 => {
                let sx = other.spacing[0];
                let sy = other.spacing[1];
                let sz = other.spacing[2];
                let d = &other.direction;
                // M row 0: sx * D col 0
                // M row 1: sy * D col 1
                // M row 2: sz * D col 2
                vec![
                    sx*d[(0,0)], sx*d[(1,0)], sx*d[(2,0)],
                    sy*d[(0,1)], sy*d[(1,1)], sy*d[(2,1)],
                    sz*d[(0,2)], sz*d[(1,2)], sz*d[(2,2)]
                ]
            },
            _ => panic!("Only 2D/3D supported"),
        };

        let m_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(m_data, Shape::new([D, D])),
            &device
        );

        let origin_vec: Vec<f32> = (0..D).map(|i| other.origin[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, Shape::new([D])),
            &device
        ).reshape([1, D]);

        // 3. Process in chunks
        const CHUNK_SIZE: usize = 32768;
        let num_chunks = (n_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut final_components = vec![Vec::new(); D];
        
        // We also need other.components flattened to add to physical points
        let other_flat: Vec<Tensor<B, 1>> = other.components.iter()
            .map(|c| c.clone().reshape([n_points]))
            .collect();
        
        let interpolator = LinearInterpolator::new();

        for i in 0..num_chunks {
            let start = i * CHUNK_SIZE;
            let end = std::cmp::min(start + CHUNK_SIZE, n_points);
            let chunk_indices = indices.clone().slice([start..end]);
            
            // Physical points P
            let p = chunk_indices.matmul(m_tensor.clone()) + origin_tensor.clone();
            
            // Get u2 chunk
            let mut u2_chunks = Vec::with_capacity(D);
            for d in 0..D {
                u2_chunks.push(other_flat[d].clone().slice([start..end]));
            }
            let u2 = Tensor::stack(u2_chunks.clone(), 1); // [Batch, D]
            
            // P' = P + u2
            let p_prime = p + u2;
            
            // Interpolate self at P'
            // Convert to indices in self
            let self_indices = self.world_to_index_tensor(p_prime);
            
            // Interpolate each component of self
            for d in 0..D {
                let u1_val = interpolator.interpolate(&self.components[d], self_indices.clone());
                // u1_val is [Batch]
                
                // Total = u2 + u1
                // We have u2_chunks[d] which is [Batch]
                let total_val = u2_chunks[d].clone() + u1_val;
                final_components[d].push(total_val);
            }
        }
        
        // Concat and reshape
        let mut new_components = Vec::with_capacity(D);
        for d in 0..D {
            let flat = Tensor::cat(final_components[d].clone(), 0);
            let reshaped = flat.reshape(Shape::new(shape_dims));
            new_components.push(reshaped);
        }
        
        Self::new(new_components, other.origin.clone(), other.spacing.clone(), other.direction.clone())
    }

    /// Convert physical points to continuous indices in the field's grid.
    ///
    /// Maps from physical space to index space.
    ///
    /// # Arguments
    /// * `points` - A tensor of shape `[Batch, D]` containing physical points
    ///
    /// # Returns
    /// A tensor of shape `[Batch, D]` containing continuous indices
    pub fn world_to_index_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_points, _] = points.dims();
        
        // WGPU dispatch limit workaround (copied from Image implementation for consistency)
        const CHUNK_SIZE: usize = 32768;

        if n_points <= CHUNK_SIZE {
            let diff = points - self.origin_tensor.clone();
            diff.matmul(self.world_to_index_matrix.clone())
        } else {
            let mut chunks = Vec::new();
            let num_chunks = (n_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n_points);
                let chunk_points = points.clone().slice([start..end]);
                
                let diff = chunk_points - self.origin_tensor.clone();
                let result = diff.matmul(self.world_to_index_matrix.clone());
                chunks.push(result);
            }
            
            Tensor::cat(chunks, 0)
        }
    }
}

/// Displacement field transform.
///
/// Transforms points by adding a displacement vector interpolated from a field.
///
/// # Type Parameters
/// * `B` - The Burn backend
/// * `D` - The spatial dimensionality
/// * `I` - The interpolator type (defaults to LinearInterpolator)
#[derive(Module, Debug)]
pub struct DisplacementFieldTransform<B: Backend, const D: usize> {
    field: DisplacementField<B, D>,
    interpolator: LinearInterpolator,
}

impl<B: Backend, const D: usize> DisplacementFieldTransform<B, D> {
    /// Create a new displacement field transform.
    pub fn new(field: DisplacementField<B, D>, interpolator: LinearInterpolator) -> Self {
        Self {
            field,
            interpolator,
        }
    }

    /// Get the underlying displacement field.
    pub fn field(&self) -> &DisplacementField<B, D> {
        &self.field
    }
    
    /// Get the interpolator.
    pub fn interpolator(&self) -> &LinearInterpolator {
        &self.interpolator
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for DisplacementFieldTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // 1. Convert physical points to field indices
        let indices = self.field.world_to_index_tensor(points.clone());
        
        // 2. Interpolate each component
        let mut displacement_components = Vec::with_capacity(D);
        for i in 0..D {
            let component = &self.field.components[i];
            // interpolate returns [Batch]
            let val = self.interpolator.interpolate(component, indices.clone());
            displacement_components.push(val);
        }
        
        // 3. Stack components to get displacement vectors [Batch, D]
        let displacement = Tensor::stack(displacement_components, 1);
        
        // 4. Add displacement to original points
        points + displacement
    }
}

impl<B: Backend, const D: usize> Resampleable<B, D> for DisplacementFieldTransform<B, D> {
    fn resample(
        &self,
        shape: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        let new_field = self.field.resample(shape, origin, spacing, direction);
        Self::new(new_field, self.interpolator.clone())
    }
}

// Type aliases
pub type DisplacementField2D<B> = DisplacementField<B, 2>;
pub type DisplacementField3D<B> = DisplacementField<B, 3>;
pub type DisplacementFieldTransform2D<B> = DisplacementFieldTransform<B, 2>;
pub type DisplacementFieldTransform3D<B> = DisplacementFieldTransform<B, 3>;
