//! Coeus-backed image contract.
//!
//! This module is the Atlas tensor migration target for image metadata.  The
//! legacy crate root still exposes the Burn-backed [`crate::Image`] while
//! downstream callers migrate to `ritk_image::coeus::Image`.

use std::fmt;

use anyhow::{anyhow, bail};
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

/// Medical image backed by a Coeus tensor.
///
/// The `D` const generic is the image dimensionality. Construction validates
/// that the tensor rank matches `D`, so index-space metadata cannot be paired
/// with a tensor of a different rank.
#[derive(Clone)]
pub struct Image<T, B, const D: usize>
where
    T: Scalar,
    B: ComputeBackend,
{
    data: Tensor<T, B>,
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
}

impl<T, B, const D: usize> fmt::Debug for Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("shape", &self.data.shape())
            .field("origin", &self.origin)
            .field("spacing", &self.spacing)
            .field("direction", &self.direction)
            .finish()
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Create an image from Coeus tensor data and physical metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when `data.ndim() != D`.
    pub fn new(
        data: Tensor<T, B>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> anyhow::Result<Self> {
        let rank = data.ndim();
        if rank != D {
            bail!("image tensor rank mismatch: expected {D}, got {rank}");
        }

        Ok(Self {
            data,
            origin,
            spacing,
            direction,
        })
    }

    /// Get the image data tensor.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &Tensor<T, B> {
        &self.data
    }

    /// Get the physical coordinate of the first pixel.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Get the physical distance between neighboring pixels.
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Get the direction cosine matrix for the image axes.
    #[inline]
    #[must_use]
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }

    /// Get the image shape as a fixed-rank array.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        self.data
            .shape()
            .try_into()
            .expect("invariant: Image::new validates tensor rank equals D")
    }

    /// Consume the image and return the underlying Coeus tensor.
    #[inline]
    #[must_use]
    pub fn into_tensor(self) -> Tensor<T, B> {
        self.data
    }

    /// Consume the image and return all components.
    ///
    /// Returns `(tensor, origin, spacing, direction)`.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (Tensor<T, B>, Point<D>, Spacing<D>, Direction<D>) {
        (self.data, self.origin, self.spacing, self.direction)
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Borrow contiguous host-addressable image data.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not row-major contiguous. Backends
    /// without CPU-addressable storage do not satisfy this method's trait
    /// bounds and must transfer through an explicit backend operation first.
    #[inline]
    pub fn data_slice(&self) -> anyhow::Result<&[T]> {
        if !self.data.is_contiguous() {
            return Err(anyhow!(
                "image data is not contiguous: shape={:?}, strides={:?}",
                self.data.shape(),
                self.data.strides()
            ));
        }

        Ok(self.data.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use coeus_core::SequentialBackend;

    use super::*;

    type TensorImage<const D: usize> = Image<f32, SequentialBackend, D>;

    fn metadata_2d() -> (Point<2>, Spacing<2>, Direction<2>) {
        (
            Point::new([10.0, 20.0]),
            Spacing::new([0.5, 1.5]),
            Direction::identity(),
        )
    }

    #[test]
    fn construction_preserves_shape_and_metadata() {
        let data =
            Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (origin, spacing, direction) = metadata_2d();

        let image = TensorImage::<2>::new(data, origin, spacing, direction).unwrap();

        assert_eq!(image.shape(), [2, 3]);
        assert_eq!(image.origin(), &origin);
        assert_eq!(image.spacing(), &spacing);
        assert_eq!(image.direction(), &direction);
        assert_eq!(image.data_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn construction_rejects_rank_mismatch() {
        let data = Tensor::<f32, SequentialBackend>::from_slice([2, 3, 1], &[0.0; 6]);
        let (origin, spacing, direction) = metadata_2d();

        let err = TensorImage::<2>::new(data, origin, spacing, direction).unwrap_err();

        assert_eq!(
            err.to_string(),
            "image tensor rank mismatch: expected 2, got 3"
        );
    }

    #[test]
    fn data_slice_rejects_non_contiguous_layout() {
        let data =
            Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let column_view = Tensor::from_raw_parts(
            data.storage().clone(),
            data.layout().slice(&[(0, 2), (1, 2)]),
        );
        let (origin, spacing, direction) = metadata_2d();
        let image = TensorImage::<2>::new(column_view, origin, spacing, direction).unwrap();

        let err = image.data_slice().unwrap_err();

        assert_eq!(
            err.to_string(),
            "image data is not contiguous: shape=[2, 1], strides=[3, 1]"
        );
    }

    #[test]
    fn into_parts_returns_exact_tensor_and_metadata() {
        let data = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let (origin, spacing, direction) = metadata_2d();
        let image = TensorImage::<2>::new(data, origin, spacing, direction).unwrap();

        let (data, returned_origin, returned_spacing, returned_direction) = image.into_parts();

        assert_eq!(data.shape(), &[2, 2]);
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(returned_origin, origin);
        assert_eq!(returned_spacing, spacing);
        assert_eq!(returned_direction, direction);
    }
}
