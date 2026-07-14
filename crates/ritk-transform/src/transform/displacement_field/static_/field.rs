//! Non-trainable Coeus displacement field and transform.

use super::super::core::DisplacementFieldError;
use super::super::geometry::{geometry_tensors, physical_grid, validate_components};
use super::super::resample::ResampleError;
use super::super::transform::DisplacementTransformError;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::{BackendOps, Dimension, Replicate, SupportedDimension};
use coeus_tensor::{Tensor, Transpose};
use ritk_core::spatial::{Direction, Point, Spacing};

/// Dense non-trainable displacement vectors on a regular physical grid.
#[derive(Clone)]
pub struct StaticDisplacementField<B: Backend, const D: usize>
where
    B: BackendOps<f32>,
{
    components: Vec<Tensor<f32, B>>,
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
    world_to_index_matrix: Tensor<f32, B>,
    origin_tensor: Tensor<f32, B>,
}

impl<B: Backend + BackendOps<f32>, const D: usize> StaticDisplacementField<B, D>
where
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Validate components and construct a static field.
    pub fn new(
        components: Vec<Tensor<f32, B>>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Result<Self, DisplacementFieldError> {
        validate_components::<B, D>(&components)?;
        let geometry = geometry_tensors::<B, D>(origin, spacing, direction)?;
        Ok(Self {
            components,
            origin,
            spacing,
            direction,
            world_to_index_matrix: geometry.world_to_index,
            origin_tensor: geometry.origin,
        })
    }

    /// Borrow the non-trainable component tensors.
    #[must_use]
    pub fn components(&self) -> &[Tensor<f32, B>] {
        &self.components
    }

    /// Physical origin.
    #[must_use]
    pub const fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Physical voxel spacing.
    #[must_use]
    pub const fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Physical direction cosine matrix.
    #[must_use]
    pub const fn direction(&self) -> Direction<D> {
        self.direction
    }

    /// Map physical `[N, D]` points to continuous field indices.
    #[must_use]
    pub fn world_to_index_tensor(&self, points: &Tensor<f32, B>) -> Tensor<f32, B> {
        let backend = B::default();
        coeus_ops::matmul(
            &coeus_ops::sub(points, &self.origin_tensor, &backend),
            &self.world_to_index_matrix,
            &backend,
        )
    }

    pub(crate) fn sample_components(
        &self,
        points: &Tensor<f32, B>,
        boundary: Replicate,
    ) -> Result<Vec<Tensor<f32, B>>, coeus_ops::InterpolationError> {
        let indices = self.world_to_index_tensor(points);
        let point_count = points.shape()[0];
        let mut grid_shape = vec![1, D, point_count];
        grid_shape.resize(D + 2, 1);
        let grid = indices.transpose().to_contiguous().reshape(grid_shape);
        self.components
            .iter()
            .map(|component| {
                let image_shape = [1, 1]
                    .into_iter()
                    .chain(component.shape().iter().copied())
                    .collect::<Vec<_>>();
                let image = component.reshape(image_shape);
                coeus_ops::linear_interpolation::<D, _, _>(&image, &grid, boundary)
                    .map(|sampled| sampled.reshape([point_count]))
            })
            .collect()
    }

    /// Resample the field onto a new regular physical grid.
    pub fn resample(
        &self,
        new_shape: [usize; D],
        new_origin: Point<D>,
        new_spacing: Spacing<D>,
        new_direction: Direction<D>,
    ) -> Result<Self, ResampleError> {
        let values = physical_grid(new_shape, new_origin, new_spacing, new_direction)?;
        let points = Tensor::from_slice_on([values.len() / D, D], &values, &B::default());
        let components = self
            .sample_components(&points, Replicate)?
            .iter()
            .map(|sampled| sampled.reshape(new_shape))
            .collect();
        Ok(Self::new(
            components,
            new_origin,
            new_spacing,
            new_direction,
        )?)
    }
}

/// Non-trainable displacement transform with replicated-border interpolation.
#[derive(Clone)]
pub struct StaticDisplacementFieldTransform<B: Backend, const D: usize>
where
    B: BackendOps<f32>,
{
    field: StaticDisplacementField<B, D>,
    boundary: Replicate,
}

impl<B: Backend + BackendOps<f32>, const D: usize> StaticDisplacementFieldTransform<B, D>
where
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct a static displacement transform.
    #[must_use]
    pub const fn new(field: StaticDisplacementField<B, D>) -> Self {
        Self {
            field,
            boundary: Replicate,
        }
    }

    /// Borrow the underlying field.
    #[must_use]
    pub const fn field(&self) -> &StaticDisplacementField<B, D> {
        &self.field
    }

    /// Transform physical `[N, D]` points.
    pub fn transform_points(
        &self,
        points: &Tensor<f32, B>,
    ) -> Result<Tensor<f32, B>, DisplacementTransformError> {
        let shape = points.shape();
        if shape.len() != 2 || shape[1] != D {
            return Err(DisplacementTransformError::PointShape {
                dimension: D,
                actual: shape.to_vec(),
            });
        }
        let components = self.field.sample_components(points, self.boundary)?;
        let references = components.iter().collect::<Vec<_>>();
        Ok(coeus_ops::add(
            points,
            &coeus_ops::stack(&references, 1),
            &B::default(),
        ))
    }

    /// Resample the transform onto a new grid.
    pub fn resample(
        &self,
        shape: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Result<Self, ResampleError> {
        Ok(Self::new(
            self.field.resample(shape, origin, spacing, direction)?,
        ))
    }
}
