use super::core::{DisplacementField, DisplacementFieldError};
use coeus_autograd::{reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::{BackendOps, Dimension, InterpolationError, SupportedDimension};
use coeus_tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};

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
        let points = physical_grid(new_shape, new_origin, new_spacing, new_direction);
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

fn physical_grid<const D: usize>(
    shape: [usize; D],
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
) -> Vec<f32> {
    let points = shape.iter().product::<usize>();
    let mut coordinates = Vec::with_capacity(points * D);
    for linear in 0..points {
        let mut remainder = linear;
        let mut index = [0usize; D];
        for axis in (0..D).rev() {
            index[axis] = remainder % shape[axis];
            remainder /= shape[axis];
        }
        for row in 0..D {
            let offset = (0..D)
                .map(|column| direction[(row, column)] * spacing[column] * index[column] as f64)
                .sum::<f64>();
            coordinates.push((origin[row] + offset) as f32);
        }
    }
    coordinates
}
