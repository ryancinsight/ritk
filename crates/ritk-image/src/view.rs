//! Borrowed, layout-aware host views over voxel data.
//!
//! The fourth shape of the access question in [`crate::access`], and the one
//! that costs nothing: borrow the voxels *as laid out* rather than as a flat
//! logical sequence. [`crate::access`]'s three families all answer "give me a
//! flat row-major host slice", which forces a copy whenever the tensor is
//! strided or offset. A view answers "give me indexed access to the voxels",
//! which a stride and an offset express directly — so it never copies, and it
//! never fails on a layout the flat contract would reject.
//!
//! The view type is [`leto::ArrayView`], not a RITK type. Coeus stores the
//! layout and `coeus_leto::to_leto_view` already converts a Coeus layout into
//! a leto one; leto owns the borrowed-array vocabulary and the whole iterator
//! family over it (`axis_iter`, `lanes`, `slice`, plus `Tiles`, `Windows` and
//! `ExactChunks` built from a view's layout and data). Defining a RITK view
//! type would fork that vocabulary for no capability, so this module is the
//! adapter and nothing more.

use anyhow::anyhow;
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;
use leto::ArrayView;

use crate::types::Image;

/// Borrow a rank-`N` view over a tensor's host storage.
///
/// Zero-copy for every CPU-addressable tensor, whatever its layout: the
/// stride and offset ride in the returned view's layout instead of being
/// normalized away by a materializing copy. Contrast
/// [`Tensor::to_contiguous`], which allocates and copies the whole buffer
/// whenever the layout is strided *or* merely offset — `is_contiguous()` looks
/// only at strides, so a row slice of a contiguous tensor reports contiguous
/// and still costs a full copy.
///
/// # Errors
///
/// Returns an error when the tensor rank exceeds `N`, or when the layout's
/// footprint does not fit the storage. A rank below `N` is left-padded with
/// size-1 axes.
pub fn tensor_view<T, B, const N: usize>(
    tensor: &Tensor<T, B>,
) -> anyhow::Result<ArrayView<'_, T, N>>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    // The whole storage, via the storage trait rather than `Tensor::as_slice`:
    // the tensor method pre-applies the offset and asserts contiguity, both of
    // which the layout already encodes and the leto view consumes directly.
    let storage = tensor.storage().as_slice();

    coeus_leto::to_leto_view::<T, N>(tensor.layout(), storage).map_err(|error| {
        anyhow!(
            "cannot view tensor as rank {N}: {error} (shape={:?}, strides={:?})",
            tensor.shape(),
            tensor.strides()
        )
    })
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Borrow the voxels as a rank-`D` view.
    ///
    /// The zero-copy access path. `Image::new` validates that the tensor rank
    /// equals `D`, so the rank conversion cannot pad or truncate here.
    ///
    /// # Errors
    ///
    /// Returns an error only when the layout's footprint exceeds the storage —
    /// see [`tensor_view`].
    #[inline]
    pub fn view(&self) -> anyhow::Result<ArrayView<'_, T, D>> {
        tensor_view::<T, B, D>(&self.data)
    }
}

#[cfg(test)]
#[path = "tests_view.rs"]
mod tests;
