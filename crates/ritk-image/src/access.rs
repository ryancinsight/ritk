//! Host-side access to an [`Image`]'s voxel data.
//!
//! Two accessors, one per distinct behaviour. [`Image::data_slice`] borrows a
//! contiguous host slice or fails; [`Image::data_cow_on`] never fails, borrowing
//! when the layout permits and materialising a compact copy when it does not.
//! A caller that needs ownership writes `.into_owned()`, which is where the copy
//! belongs: visible at the site that pays for it, rather than hidden behind a
//! name. Kept apart from the type definition because choosing between the two is
//! a performance decision the caller makes, not part of an image's identity.

use anyhow::anyhow;
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};

use crate::types::Image;

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
    /// Returns an error when the tensor is not row-major contiguous.
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
    B: ComputeBackend,
{
    /// Host image data in logical row-major order, borrowing when the tensor
    /// is already contiguous and materializing a compact copy otherwise.
    ///
    /// The layout-independent host-extraction surface format writers and
    /// boundary code need (ADR 0002 cutover prerequisite): unlike
    /// [`Self::data_slice`], it never fails on a strided view — it pays the
    /// copy exactly when the layout requires one (`Cow::Owned`), and is
    /// zero-copy otherwise (`Cow::Borrowed`). Mirrors the Coeus `Image`'s
    /// `data_slice() -> Cow` contract.
    ///
    /// Callers needing an owned `Vec<T>` call `.into_owned()`; extraction
    /// succeeds for every valid image, so no fallible owning form exists.
    #[must_use]
    pub fn data_cow_on(&self, backend: &B) -> std::borrow::Cow<'_, [T]> {
        self.data.host_cow_on(backend)
    }
}

#[cfg(test)]
#[path = "tests_access.rs"]
mod tests;
