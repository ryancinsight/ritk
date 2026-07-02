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
    /// Create an image from flat voxel data, shape, and physical metadata.
    ///
    /// This constructor validates the shape product before constructing the
    /// tensor, so malformed external buffers fail at the image boundary instead
    /// of relying on a downstream tensor panic.
    ///
    /// # Errors
    ///
    /// Returns an error when the checked product of `dims` overflows, or when
    /// `data.len()` does not equal that product.
    pub fn from_flat_on(
        data: Vec<T>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
        backend: &B,
    ) -> anyhow::Result<Self> {
        let expected = checked_numel(&dims)?;
        if data.len() != expected {
            bail!(
                "image flat data length {} does not match shape {:?} product {}",
                data.len(),
                dims,
                expected
            );
        }

        Self::new(
            Tensor::from_slice_on(dims, &data, backend),
            origin,
            spacing,
            direction,
        )
    }

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
    B: ComputeBackend + Default,
{
    /// Create an image from flat voxel data on `B::default()`.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Image::from_flat_on`].
    #[inline]
    pub fn from_flat(
        data: Vec<T>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> anyhow::Result<Self> {
        Self::from_flat_on(data, dims, origin, spacing, direction, &B::default())
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

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Host image data in logical row-major order, borrowing when the tensor
    /// is already contiguous and materializing a compact copy otherwise.
    ///
    /// The layout-independent host-extraction surface format writers and
    /// boundary code need (ADR 0002 cutover prerequisite): unlike
    /// [`Self::data_slice`], it never fails on a strided view — it pays the
    /// copy exactly when the layout requires one (`Cow::Owned`), and is
    /// zero-copy otherwise (`Cow::Borrowed`). Mirrors the Burn `Image`'s
    /// `data_slice() -> Cow` contract. (`B: Default` follows from
    /// `Tensor::to_contiguous_on`'s own bound.)
    #[must_use]
    pub fn data_cow_on(&self, backend: &B) -> std::borrow::Cow<'_, [T]> {
        if self.data.is_contiguous() {
            std::borrow::Cow::Borrowed(self.data.as_slice())
        } else {
            std::borrow::Cow::Owned(self.data.to_contiguous_on(backend).as_slice().to_vec())
        }
    }

    /// Owned host image data in logical row-major order (layout-independent).
    ///
    /// Thin wrapper over [`Self::data_cow_on`] for callers that need a `Vec`
    /// (the Coeus counterpart of the Burn `Image`'s `try_data_vec`).
    #[must_use]
    pub fn data_vec_on(&self, backend: &B) -> Vec<T> {
        self.data_cow_on(backend).into_owned()
    }

    /// [`Self::data_cow_on`] on `B::default()` (mirrors [`Self::from_flat`]).
    #[must_use]
    pub fn data_cow(&self) -> std::borrow::Cow<'_, [T]> {
        self.data_cow_on(&B::default())
    }

    /// [`Self::data_vec_on`] on `B::default()`.
    #[must_use]
    pub fn data_vec(&self) -> Vec<T> {
        self.data_vec_on(&B::default())
    }
}

fn checked_numel(dims: &[usize]) -> anyhow::Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("image shape {:?} product overflows usize", dims))
    })
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
    fn from_flat_preserves_shape_values_and_metadata() {
        let (origin, spacing, direction) = metadata_2d();

        let image = TensorImage::<2>::from_flat(
            vec![1.0, 2.0, 3.0, 4.0],
            [2, 2],
            origin,
            spacing,
            direction,
        )
        .unwrap();

        assert_eq!(image.shape(), [2, 2]);
        assert_eq!(image.origin(), &origin);
        assert_eq!(image.spacing(), &spacing);
        assert_eq!(image.direction(), &direction);
        assert_eq!(image.data_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn data_cow_borrows_when_contiguous() {
        let (origin, spacing, direction) = metadata_2d();
        let image = TensorImage::<2>::from_flat(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            origin,
            spacing,
            direction,
        )
        .unwrap();

        let cow = image.data_cow();
        assert!(
            matches!(cow, std::borrow::Cow::Borrowed(_)),
            "contiguous image must borrow (zero-copy)"
        );
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(image.data_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn data_cow_materializes_logical_order_for_permuted_view() {
        // Build a [2, 3] image, permute the tensor to [3, 2] (non-contiguous
        // strided view), and re-wrap. Logical row-major order of the permuted
        // view is the host transpose — the oracle the extraction must match.
        let (origin, spacing, direction) = metadata_2d();
        let base =
            Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let permuted = base.permute(&[1, 0]); // shape [3, 2], strides non-contiguous
        let image = TensorImage::<2>::new(permuted, origin, spacing, direction).unwrap();

        // The strict borrow API must refuse the strided view (existing contract).
        assert!(image.data_slice().is_err(), "data_slice must reject non-contiguous");

        // Host transpose oracle: [[1,4],[2,5],[3,6]] row-major.
        let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let cow = image.data_cow();
        assert!(
            matches!(cow, std::borrow::Cow::Owned(_)),
            "non-contiguous image must materialize (owned)"
        );
        assert_eq!(cow.as_ref(), &expected);
        assert_eq!(image.data_vec(), expected.to_vec());
    }

    #[test]
    fn from_flat_rejects_shape_product_mismatch() {
        let (origin, spacing, direction) = metadata_2d();

        let err =
            TensorImage::<2>::from_flat(vec![1.0, 2.0, 3.0], [2, 2], origin, spacing, direction)
                .unwrap_err();

        assert_eq!(
            err.to_string(),
            "image flat data length 3 does not match shape [2, 2] product 4"
        );
    }

    #[test]
    fn from_flat_rejects_shape_product_overflow() {
        let (origin, spacing, direction) = metadata_2d();

        let err =
            TensorImage::<2>::from_flat(Vec::new(), [usize::MAX, 2], origin, spacing, direction)
                .unwrap_err();

        assert_eq!(
            err.to_string(),
            "image shape [18446744073709551615, 2] product overflows usize"
        );
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
