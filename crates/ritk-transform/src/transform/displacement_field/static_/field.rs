//! Displacement field transform implementation.
//!
//! This module implements a dense displacement field transform, where a vector
//! is stored at each discrete location in a grid. The transform maps points
//! by adding the interpolated displacement vector to the input point.
//!
//! $T(x) = x + D(x)$

use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_core::spatial::{Direction, Point, Spacing};
use crate::transform::{Resampleable, Transform};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Displacement field data representing a vector field on a regular grid.
///
/// This struct holds the displacement vectors and their physical space metadata.
/// It is essentially a vector-valued image.
#[derive(Clone, Debug)]
pub struct StaticDisplacementField<B: Backend, const D: usize> {
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
    origin_tensor: Tensor<B, 1>,
}

impl<B: Backend, const D: usize> StaticDisplacementField<B, D> {
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
        assert_eq!(
            components.len(),
            D,
            "Number of components must match dimensionality"
        );
        // Verify all components have same shape
        if !components.is_empty() {
            let shape = components[0].shape();
            for c in &components[1..] {
                assert_eq!(c.shape(), shape, "All components must have the same shape");
            }
        }

        let device = components[0].device();

        // Gather components directly without `Param` wrapping. We do NOT call `require_grad()` here,
        // as the components may be generated intermediate tensors from an upstream network (e.g., SSMMorph).
        // Leaf modules must configure their tracking requirements prior to initialization.
        let components = components;

        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| origin[i] as f32).collect();
        let origin_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(origin_vec, Shape::new([D])), &device);

        // 2. Prepare Transform Matrix T = (S^-1 * D^-1)^T
        let inv_dir = direction
            .try_inverse()
            .expect("Direction matrix must be invertible");

        let mut t_data = Vec::with_capacity(D * D);
        for r in 0..D {
            for c in 0..D {
                let val = (inv_dir[(c, r)] / spacing[c]) as f32;
                t_data.push(val);
            }
        }

        let world_to_index_matrix =
            Tensor::<B, 2>::from_data(TensorData::new(t_data, Shape::new([D, D])), &device);

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
        self.components.to_vec()
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
        let new_indices = ritk_core::image::grid::generate_grid(new_shape, &device);

        // 2. Prepare transform matrix for new grid index -> world
        // Matrix M[k, c] = Direction[c, k] * Spacing[k]
        let mut m_data = Vec::with_capacity(D * D);
        for k in 0..D {
            // Input dimension (index)
            for c in 0..D {
                // Output dimension (physical)
                let val = (new_direction[(c, k)] * new_spacing[k]) as f32;
                m_data.push(val);
            }
        }
        let m_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(m_data, Shape::new([D, D])), &device);

        let origin_vec: Vec<f32> = (0..D).map(|i| new_origin[i] as f32).collect();
        let origin_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(origin_vec, Shape::new([D])), &device)
                .reshape([1, D]);

        // 3. Process in chunks
        let [n_points, _] = new_indices.dims();

        let mut component_chunks: Vec<Vec<Tensor<B, 1>>> =
            vec![Vec::with_capacity(n_points.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE)); D];

        let num_chunks = n_points.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);

        for i in 0..num_chunks {
            let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
            let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n_points);
            let chunk_range = start..end;
            let chunk_indices = new_indices.clone().slice([chunk_range]);

            // World = Origin + Indices @ M
            let offset = chunk_indices.matmul(m_tensor.clone());
            let world = offset + origin_tensor.clone();

            // Convert to old index space
            let old_indices = self.world_to_index_tensor(world);

            // Interpolate each component
            for (d, cc) in component_chunks.iter_mut().enumerate() {
                let comp = &self.components[d];
                let val = interpolator.interpolate(comp, old_indices.clone());
                cc.push(val);
            }
        }

        // 4. Concat and reshape
        let mut final_components = Vec::with_capacity(D);
        for cc in component_chunks.into_iter() {
            let flat = Tensor::cat(cc, 0);
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
        if n_points <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            let diff = points - self.origin_tensor.clone().reshape([1, D]);
            diff.matmul(self.world_to_index_matrix.clone())
        } else {
            let num_chunks = n_points.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n_points);
                let chunk_range = start..end;
                let chunk_points = points.clone().slice([chunk_range]);

                let diff = chunk_points - self.origin_tensor.clone().reshape([1, D]);
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
#[derive(Clone, Debug)]
pub struct StaticDisplacementFieldTransform<B: Backend, const D: usize> {
    field: StaticDisplacementField<B, D>,
    interpolator: LinearInterpolator,
}

impl<B: Backend, const D: usize> StaticDisplacementFieldTransform<B, D> {
    /// Create a new displacement field transform.
    pub fn new(field: StaticDisplacementField<B, D>, interpolator: LinearInterpolator) -> Self {
        Self {
            field,
            interpolator,
        }
    }

    /// Get the underlying displacement field.
    pub fn field(&self) -> &StaticDisplacementField<B, D> {
        &self.field
    }

    /// Get the interpolator.
    pub fn interpolator(&self) -> &LinearInterpolator {
        &self.interpolator
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for StaticDisplacementFieldTransform<B, D> {
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

impl<B: Backend, const D: usize> Resampleable<B, D> for StaticDisplacementFieldTransform<B, D> {
    fn resample(
        &self,
        shape: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        let new_field = self.field.resample(shape, origin, spacing, direction);
        Self::new(new_field, self.interpolator)
    }
}

// Dimension-specific convenience aliases removed.
// Use `StaticDisplacementField<B, 3>` / `StaticDisplacementFieldTransform<B, D>` directly.
