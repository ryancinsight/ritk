//! Shared physical-grid geometry for trainable and static displacement fields.

use super::core::DisplacementFieldError;
use coeus_core::Backend;
use coeus_tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};

pub(crate) struct GeometryTensors<B: Backend, const D: usize> {
    pub(crate) world_to_index: Tensor<f32, B>,
    pub(crate) origin: Tensor<f32, B>,
}

pub(crate) fn validate_components<B: Backend, const D: usize>(
    components: &[Tensor<f32, B>],
) -> Result<Vec<usize>, DisplacementFieldError> {
    if components.len() != D {
        return Err(DisplacementFieldError::ComponentCount {
            expected: D,
            actual: components.len(),
        });
    }
    let shape = components
        .first()
        .expect("invariant: supported dimensions are nonzero and component count equals D")
        .shape()
        .to_vec();
    if shape.len() != D {
        return Err(DisplacementFieldError::ComponentRank {
            expected: D,
            actual: shape.len(),
        });
    }
    for (component, tensor) in components.iter().enumerate().skip(1) {
        if tensor.shape() != shape {
            return Err(DisplacementFieldError::ComponentShape {
                component,
                expected: shape.clone(),
                actual: tensor.shape().to_vec(),
            });
        }
    }
    Ok(shape)
}

pub(crate) fn geometry_tensors<B: Backend, const D: usize>(
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
) -> Result<GeometryTensors<B, D>, DisplacementFieldError> {
    let inverse = direction
        .try_inverse()
        .ok_or(DisplacementFieldError::SingularDirection)?;
    let origin_values = (0..D).map(|axis| origin[axis] as f32).collect::<Vec<_>>();
    let matrix_values = (0..D)
        .flat_map(|row| (0..D).map(move |column| (inverse[(column, row)] / spacing[column]) as f32))
        .collect::<Vec<_>>();
    let backend = B::default();
    Ok(GeometryTensors {
        world_to_index: Tensor::from_slice_on([D, D], &matrix_values, &backend),
        origin: Tensor::from_slice_on([1, D], &origin_values, &backend),
    })
}

pub(crate) fn physical_grid<const D: usize>(
    shape: [usize; D],
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
) -> Result<Vec<f32>, DisplacementFieldError> {
    if let Some(axis) = shape.iter().position(|extent| *extent == 0) {
        return Err(DisplacementFieldError::EmptyShapeAxis { axis });
    }
    let points = shape
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(DisplacementFieldError::SizeOverflow)?;
    let coordinate_count = points
        .checked_mul(D)
        .ok_or(DisplacementFieldError::SizeOverflow)?;
    let mut coordinates = Vec::new();
    coordinates
        .try_reserve_exact(coordinate_count)
        .map_err(|_| DisplacementFieldError::Allocation {
            elements: coordinate_count,
        })?;
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
    Ok(coordinates)
}
