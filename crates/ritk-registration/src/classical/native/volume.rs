//! Coeus image and Leto volume conversion for classical registration.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use eunomia::CastFrom;
use leto::Array3;
use ritk_image::native::Image;

use super::NativeConversionError;

/// Convert a contiguous native image into the Leto volume consumed by the
/// classical registration engine.
///
/// The row-major RITK `[Z, Y, X]` layout is preserved. `f32` values widen to
/// `f64` exactly, so the conversion cannot change voxel values.
///
/// # Errors
///
/// Returns an error when the image storage is not contiguous or Leto rejects
/// the checked image shape.
pub fn image_to_leto_volume<B>(
    image: &Image<f32, B, 3>,
) -> std::result::Result<Array3<f64>, NativeConversionError>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let shape = image.shape();
    let values = image
        .data_slice()
        .map_err(|error| NativeConversionError::ImageData(error.into_boxed_dyn_error()))?
        .iter()
        .copied()
        .map(f64::from)
        .collect();
    Array3::from_shape_vec(shape, values).map_err(NativeConversionError::LetoVolume)
}

/// Construct a native image from a classical Leto volume in `reference`'s
/// physical frame.
///
/// The classical engine computes in `f64`; RITK native image storage is `f32`.
/// Values are narrowed through Eunomia's explicit scalar conversion contract.
/// The output keeps the reference's `[Z, Y, X]` shape, origin, spacing, and
/// direction.
///
/// # Errors
///
/// Returns an error when the Leto volume shape cannot be represented by the
/// native image constructor.
pub fn leto_volume_to_image<B>(
    volume: &Array3<f64>,
    reference: &Image<f32, B, 3>,
    backend: &B,
) -> std::result::Result<Image<f32, B, 3>, NativeConversionError>
where
    B: ComputeBackend,
{
    let shape = volume.shape();
    let values = volume.iter().copied().map(f32::cast_from).collect();
    Image::from_flat_on(
        values,
        shape,
        *reference.origin(),
        *reference.spacing(),
        *reference.direction(),
        backend,
    )
    .map_err(|error| NativeConversionError::ImageConstruction(error.into_boxed_dyn_error()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use coeus_core::SequentialBackend;
    use ritk_spatial::{Direction, Point, Spacing};

    #[test]
    fn volume_round_trip_preserves_voxels_and_physical_frame() -> Result<()> {
        let backend = SequentialBackend;
        let shape = [2, 2, 2];
        let values = vec![-3.5, -1.0, 0.0, 0.125, 1.0, 3.25, 12.5, f32::MAX];
        let image = Image::from_flat_on(
            values.clone(),
            shape,
            Point::new([10.0, -2.5, 7.25]),
            Spacing::new([2.0, 1.5, 0.75]),
            Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            &backend,
        )?;

        let volume = image_to_leto_volume(&image)?;
        assert_eq!(volume.shape(), shape);
        assert_eq!(
            volume.iter().copied().collect::<Vec<_>>(),
            values.iter().copied().map(f64::from).collect::<Vec<_>>(),
        );

        let restored = leto_volume_to_image(&volume, &image, &backend)?;
        assert_eq!(restored.shape(), shape);
        assert_eq!(restored.origin(), image.origin());
        assert_eq!(restored.spacing(), image.spacing());
        assert_eq!(restored.direction(), image.direction());
        assert_eq!(restored.data_slice()?, values.as_slice());
        Ok(())
    }
}
