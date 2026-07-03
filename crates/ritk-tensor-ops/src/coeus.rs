//! Coeus tensor host-buffer helpers.
//!
//! These functions are the Coeus counterpart to the legacy Burn-backed
//! `Image<B, D>` helpers in the crate root. They keep the migration boundary
//! explicit: callers that already own a Coeus tensor can borrow contiguous host
//! data without allocating, and only request an owned buffer when mutation or
//! long-lived storage requires it.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_image::native::Image;

/// Borrow contiguous Coeus tensor data with a statically checked rank.
///
/// Returns `(voxels, shape)` where `voxels` borrows the tensor's contiguous
/// storage and `shape` is `[usize; D]`.
///
/// # Errors
/// Returns an error when the tensor rank does not match `D`, or when the tensor
/// is a non-contiguous view. Non-contiguous views require an explicit materialize
/// step by the caller so this helper never hides an O(N) copy behind a borrow.
pub fn extract_slice<T, B, const D: usize>(
    tensor: &Tensor<T, B>,
) -> anyhow::Result<(&[T], [usize; D])>
where
    T: coeus_core::Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    let dims = checked_dims::<D>(tensor.shape())?;
    if !tensor.is_contiguous() {
        anyhow::bail!(
            "coeus tensor ops: extract_slice requires contiguous layout, got shape={:?} strides={:?}",
            tensor.shape(),
            tensor.strides()
        );
    }
    Ok((tensor.as_slice(), dims))
}

/// Copy Coeus tensor data into an owned vector with a statically checked rank.
///
/// Use [`extract_slice`] for read-only kernels that can finish while the tensor
/// borrow is live.
///
/// # Errors
/// Returns an error under the same conditions as [`extract_slice`].
pub fn extract_vec<T, B, const D: usize>(
    tensor: &Tensor<T, B>,
) -> anyhow::Result<(Vec<T>, [usize; D])>
where
    T: coeus_core::Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    let (slice, dims) = extract_slice::<T, B, D>(tensor)?;
    Ok((slice.to_vec(), dims))
}

/// Borrow contiguous Coeus image data with a statically checked rank.
///
/// This image-level helper delegates to [`extract_slice`], keeping tensor rank
/// and contiguity validation in one implementation.
///
/// # Errors
/// Returns an error when the wrapped tensor is not rank `D` or is a
/// non-contiguous view.
pub fn extract_image_slice<T, B, const D: usize>(
    image: &Image<T, B, D>,
) -> anyhow::Result<(&[T], [usize; D])>
where
    T: coeus_core::Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    extract_slice(image.data())
}

/// Copy Coeus image data into an owned vector with a statically checked rank.
///
/// Use [`extract_image_slice`] for read-only kernels that can finish while the
/// image borrow is live.
///
/// # Errors
/// Returns an error under the same conditions as [`extract_image_slice`].
pub fn extract_image_vec<T, B, const D: usize>(
    image: &Image<T, B, D>,
) -> anyhow::Result<(Vec<T>, [usize; D])>
where
    T: coeus_core::Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    let (slice, dims) = extract_image_slice(image)?;
    Ok((slice.to_vec(), dims))
}

/// Construct a Coeus tensor from a flat buffer after validating shape length.
///
/// # Errors
/// Returns an error when the checked product of `dims` overflows or does not
/// match `vals.len()`.
pub fn rebuild<T, B, const D: usize>(
    vals: Vec<T>,
    dims: [usize; D],
    backend: &B,
) -> anyhow::Result<Tensor<T, B>>
where
    T: coeus_core::Scalar,
    B: ComputeBackend,
{
    let expected = checked_numel(&dims)?;
    if vals.len() != expected {
        anyhow::bail!(
            "coeus tensor ops: data length {} does not match shape {:?} product {}",
            vals.len(),
            dims,
            expected
        );
    }
    Ok(Tensor::from_slice_on(dims, &vals, backend))
}

/// Construct a Coeus image from a flat buffer, preserving source metadata.
///
/// # Errors
/// Returns an error when the checked product of `dims` overflows, does not
/// match `vals.len()`, or the rebuilt tensor rank does not match the image rank.
pub fn rebuild_image<T, B, const D: usize>(
    vals: Vec<T>,
    dims: [usize; D],
    src: &Image<T, B, D>,
    backend: &B,
) -> anyhow::Result<Image<T, B, D>>
where
    T: coeus_core::Scalar,
    B: ComputeBackend,
{
    let tensor = rebuild(vals, dims, backend)?;
    Image::new(tensor, *src.origin(), *src.spacing(), *src.direction())
}

fn checked_dims<const D: usize>(shape: &[usize]) -> anyhow::Result<[usize; D]> {
    shape.try_into().map_err(|_| {
        anyhow::anyhow!(
            "coeus tensor ops: expected rank {}, got rank {} shape={:?}",
            D,
            shape.len(),
            shape
        )
    })
}

fn checked_numel(dims: &[usize]) -> anyhow::Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            anyhow::anyhow!("coeus tensor ops: shape {:?} product overflows usize", dims)
        })
    })
}
