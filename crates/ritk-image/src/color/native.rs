//! Coeus-backed multi-component volumes with three-dimensional metadata.

use std::fmt;

use anyhow::{bail, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

/// Native 3-D image volume with `C` interleaved components per voxel.
///
/// Tensor layout is `[depth, rows, columns, component]`. The component axis is
/// not spatial and therefore does not participate in origin, spacing, or
/// direction metadata.
#[derive(Clone)]
pub struct ColorVolume<T, B, const C: usize>
where
    T: Scalar,
    B: ComputeBackend,
{
    data: Tensor<T, B>,
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

/// Native RGB volume with interleaved channels.
pub type RgbVolume<T, B> = ColorVolume<T, B, 3>;

impl<T, B, const C: usize> fmt::Debug for ColorVolume<T, B, C>
where
    T: Scalar,
    B: ComputeBackend,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ColorVolume")
            .field("shape", &self.data.shape())
            .field("origin", &self.origin)
            .field("spacing", &self.spacing)
            .field("direction", &self.direction)
            .finish()
    }
}

impl<T, B, const C: usize> ColorVolume<T, B, C>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Constructs a color volume from interleaved flat component storage.
    ///
    /// # Errors
    ///
    /// Returns an error for zero channels, shape-product overflow, or a data
    /// length that differs from `depth * rows * columns * C`.
    pub fn from_flat_on(
        data: Vec<T>,
        spatial_shape: [usize; 3],
        origin: Point<3>,
        spacing: Spacing<3>,
        direction: Direction<3>,
        backend: &B,
    ) -> Result<Self> {
        if C == 0 {
            bail!("color volume channel count must be positive");
        }
        let expected = spatial_shape
            .into_iter()
            .chain([C])
            .try_fold(1usize, |product, dimension| product.checked_mul(dimension))
            .context("color volume shape product overflows usize")?;
        if data.len() != expected {
            bail!(
                "color volume data length {} does not match spatial shape {spatial_shape:?} with {C} channels ({expected})",
                data.len()
            );
        }
        let [depth, rows, columns] = spatial_shape;
        Ok(Self {
            data: Tensor::from_slice_on([depth, rows, columns, C], &data, backend),
            origin,
            spacing,
            direction,
        })
    }

    /// Returns `[depth, rows, columns, channels]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 4] {
        self.data
            .shape()
            .try_into()
            .expect("invariant: construction creates a rank-4 tensor")
    }

    /// Returns the three spatial dimensions.
    #[must_use]
    pub fn spatial_shape(&self) -> [usize; 3] {
        let [depth, rows, columns, _] = self.shape();
        [depth, rows, columns]
    }

    /// Returns the compile-time component count.
    #[must_use]
    pub const fn channels(&self) -> usize {
        C
    }

    /// Returns the physical origin.
    #[must_use]
    pub const fn origin(&self) -> &Point<3> {
        &self.origin
    }

    /// Returns the physical voxel spacing.
    #[must_use]
    pub const fn spacing(&self) -> &Spacing<3> {
        &self.spacing
    }

    /// Returns the direction cosine matrix.
    #[must_use]
    pub const fn direction(&self) -> &Direction<3> {
        &self.direction
    }

    /// Returns the underlying Coeus tensor.
    #[must_use]
    pub const fn data(&self) -> &Tensor<T, B> {
        &self.data
    }
}

impl<T, B, const C: usize> ColorVolume<T, B, C>
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Returns logical row-major host data, borrowing contiguous storage.
    #[must_use]
    pub fn data_cow_on(&self, backend: &B) -> std::borrow::Cow<'_, [T]> {
        if self.data.is_contiguous() {
            std::borrow::Cow::Borrowed(self.data.as_slice())
        } else {
            std::borrow::Cow::Owned(self.data.to_contiguous_on(backend).as_slice().to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn rgb_volume_keeps_component_axis_out_of_spatial_metadata() -> Result<()> {
        let backend = SequentialBackend;
        let volume = RgbVolume::from_flat_on(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 2],
            Point::new([7.0, 8.0, 9.0]),
            Spacing::new([1.0, 2.0, 3.0]),
            Direction::identity(),
            &backend,
        )?;
        assert_eq!(volume.shape(), [1, 1, 2, 3]);
        assert_eq!(volume.spatial_shape(), [1, 1, 2]);
        assert_eq!(volume.channels(), 3);
        assert_eq!(volume.origin().to_array(), [7.0, 8.0, 9.0]);
        assert_eq!(
            volume.data_cow_on(&backend).as_ref(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn rgb_volume_rejects_mismatched_component_storage() {
        let error = RgbVolume::<f32, SequentialBackend>::from_flat_on(
            vec![0.0; 5],
            [1, 1, 2],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not match"));
    }
}
