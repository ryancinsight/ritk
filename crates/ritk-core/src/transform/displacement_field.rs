//! Displacement field transform implementation.
//!
//! This module implements a dense displacement field transform, where a vector
//! is stored at each discrete location in a grid. The transform maps points
//! by adding the interpolated displacement vector to the input point.
//!
//! $T(x) = x + D(x)$

use burn::tensor::{Tensor, TensorData, Shape};
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
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
    components: Vec<Param<Tensor<B, D>>>,
    /// Physical coordinate of the first pixel (index 0,0,0).
    origin: Point<D>,
    /// Physical distance between pixels along each axis.
    spacing: Spacing<D>,
    /// Orientation of the image axes.
    direction: Direction<D>,
    
    // Precomputed matrices for coordinate transformation
    world_to_index_matrix: Param<Tensor<B, 2>>,
    origin_tensor: Param<Tensor<B, 2>>,
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

        // Wrap components in Param
        let components = components.into_iter()
            .map(|c| Param::from_tensor(c.require_grad()))
            .collect();

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
            world_to_index_matrix: Param::from_tensor(world_to_index_matrix),
            origin_tensor: Param::from_tensor(origin_tensor),
        }
    }

    /// Get the components.
    pub fn components(&self) -> Vec<Tensor<B, D>> {
        self.components.iter().map(|p| p.val()).collect()
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
                let comp = &self.components[d].val();
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
            let diff = points - self.origin_tensor.val();
            diff.matmul(self.world_to_index_matrix.val())
        } else {
            let mut chunks = Vec::new();
            let num_chunks = (n_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n_points);
                let chunk_points = points.clone().slice([start..end]);
                
                let diff = chunk_points - self.origin_tensor.val();
                let result = diff.matmul(self.world_to_index_matrix.val());
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
