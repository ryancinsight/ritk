use super::core::{DisplacementField, DisplacementFieldError};
use coeus_autograd::{reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::{BackendOps, Dimension, InterpolationError, SupportedDimension};
use coeus_tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};

use super::geometry::physical_grid;

/// Resampling failure for a trainable displacement field.
#[derive(Debug, thiserror::Error)]
pub enum ResampleError {
    /// Interpolation rejected the field or coordinate grid.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
    /// The reconstructed field violated a construction invariant.
    #[error(transparent)]
    Field(#[from] DisplacementFieldError),
}

impl<B: Backend + BackendOps<f32>, const D: usize> DisplacementField<B, D>
where
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Resample the field onto a new regular physical grid.
    pub fn resample(
        &self,
        new_shape: [usize; D],
        new_origin: Point<D>,
        new_spacing: Spacing<D>,
        new_direction: Direction<D>,
    ) -> Result<Self, ResampleError> {
        let points = physical_grid(new_shape, new_origin, new_spacing, new_direction)?;
        let points = Var::new(
            Tensor::from_slice_on([points.len() / D, D], &points, &B::default()),
            false,
        );
        let components = self
            .sample_components(&points, coeus_ops::Replicate)?
            .iter()
            .map(|sampled| reshape(sampled, new_shape).tensor)
            .collect::<Vec<_>>();
        Ok(Self::new(
            components,
            new_origin,
            new_spacing,
            new_direction,
        )?)
    }
}
