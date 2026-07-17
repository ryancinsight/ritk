//! Coordinate-frame conversion for classical affine registration.

use coeus_core::ComputeBackend;
use eunomia::CastFrom;
use leto::FixedMatrix;
use ritk_image::native::Image;
use ritk_transform::transform::affine::AtlasAffineTransform;

use crate::types::AffineTransform;

use super::NativeConversionError;

type Matrix3 = FixedMatrix<f64, 3, 3>;

/// Convert a classical index-space affine into a physical-space native affine.
///
/// [`AffineTransform`] maps a fixed output index `[x, y, z]` to a moving
/// source index in the same innermost-first coordinate order. Native
/// resampling, in contrast, applies an affine to axis-major physical points.
/// This conversion therefore constructs
/// `moving_index_to_physical * index_affine * fixed_physical_to_index` and
/// preserves both images' distinct origins, spacings, and directions.
///
/// # Errors
///
/// Returns an error when the fixed physical frame is singular or when the
/// resulting physical transform cannot be represented by the native `f32`
/// transform contract.
pub fn index_affine_to_physical<B>(
    index_affine: &AffineTransform,
    fixed: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
) -> std::result::Result<AtlasAffineTransform<B, 3>, NativeConversionError>
where
    B: ComputeBackend,
{
    let fixed_index_to_physical = index_to_physical_matrix(fixed);
    let fixed_physical_to_index = fixed_index_to_physical
        .try_inverse()
        .ok_or(NativeConversionError::SingularFixedPhysicalFrame)?;
    let moving_index_to_physical = index_to_physical_matrix(moving);
    let index_linear = Matrix3::from_rows([
        [index_affine.0[0], index_affine.0[1], index_affine.0[2]],
        [index_affine.0[4], index_affine.0[5], index_affine.0[6]],
        [index_affine.0[8], index_affine.0[9], index_affine.0[10]],
    ]);
    let index_translation = [index_affine.0[3], index_affine.0[7], index_affine.0[11]];
    let physical_linear = moving_index_to_physical * index_linear * fixed_physical_to_index;
    let physical_translation = std::array::from_fn(|axis| {
        moving.origin()[axis]
            + (0..3)
                .map(|column| moving_index_to_physical[(axis, column)] * index_translation[column])
                .sum::<f64>()
            - (0..3)
                .map(|column| physical_linear[(axis, column)] * fixed.origin()[column])
                .sum::<f64>()
    });
    let matrix = narrow_matrix(physical_linear)?;
    let translation = narrow_vector(physical_translation, "physical affine translation")?;

    AtlasAffineTransform::try_new(&matrix, &translation, &[0.0; 3])
        .map_err(NativeConversionError::PhysicalAffineConstruction)
}

fn index_to_physical_matrix<B>(image: &Image<f32, B, 3>) -> Matrix3
where
    B: ComputeBackend,
{
    Matrix3::from_rows(std::array::from_fn(|world_axis| {
        std::array::from_fn(|index_column| {
            let metadata_axis = 2 - index_column;
            image.direction()[(world_axis, metadata_axis)] * image.spacing()[metadata_axis]
        })
    }))
}

fn narrow_matrix(matrix: Matrix3) -> std::result::Result<[f32; 9], NativeConversionError> {
    let values = matrix.into_row_major();
    let mut narrowed = [0.0; 9];
    for (index, value) in values.into_iter().enumerate() {
        narrowed[index] = narrow(value, "physical affine matrix")?;
    }
    Ok(narrowed)
}

fn narrow_vector(
    values: [f64; 3],
    role: &'static str,
) -> std::result::Result<[f32; 3], NativeConversionError> {
    let mut narrowed = [0.0; 3];
    for (index, value) in values.into_iter().enumerate() {
        narrowed[index] = narrow(value, role)?;
    }
    Ok(narrowed)
}

fn narrow(value: f64, role: &'static str) -> std::result::Result<f32, NativeConversionError> {
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(NativeConversionError::NonRepresentablePhysicalAffine { role, value });
    }
    Ok(f32::cast_from(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use coeus_core::SequentialBackend;
    use ritk_spatial::{Direction, Point, Spacing};

    #[test]
    fn index_affine_uses_distinct_fixed_and_moving_physical_frames() -> Result<()> {
        let backend = SequentialBackend;
        let fixed = Image::from_flat_on(
            vec![0.0],
            [1, 1, 1],
            Point::new([10.0, 20.0, 30.0]),
            Spacing::new([2.0, 3.0, 5.0]),
            Direction::identity(),
            &backend,
        )?;
        let moving = Image::from_flat_on(
            vec![0.0],
            [1, 1, 1],
            Point::new([-4.0, 8.0, 12.0]),
            Spacing::new([7.0, 11.0, 13.0]),
            Direction::identity(),
            &backend,
        )?;
        let index_affine = AffineTransform::new([
            1.0, 0.0, 0.0, 1.5, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 1.0, 3.25, 0.0, 0.0, 0.0, 1.0,
        ]);
        let physical_affine = index_affine_to_physical(&index_affine, &fixed, &moving)?;
        let fixed_world_point = Image::from_flat_on(
            vec![22.0, 35.0, 50.0],
            [1, 3],
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
            &backend,
        )?;

        let transformed = physical_affine.transform_points(&fixed_world_point)?;
        let expected = [60.75_f32, 41.0, 83.5];
        for (axis, (actual, expected)) in
            transformed.data_vec().into_iter().zip(expected).enumerate()
        {
            // The native affine carrier stores matrix and translation values as
            // f32. Eight unit roundoffs cover the three-term dot product plus
            // its two additions and the initial physical-value narrowing.
            let rounding_bound = 8.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= rounding_bound,
                "physical axis {axis} differs: actual={actual}, expected={expected}, bound={rounding_bound}"
            );
        }
        Ok(())
    }
}
