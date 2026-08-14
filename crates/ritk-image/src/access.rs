//! Host-side access to an [`Image`]'s voxel data.
//!
//! Three shapes of one question: can this tensor be read as a flat host slice,
//! and what does it cost. `data_slice` borrows or fails, `data_cow_*` borrows
//! when the layout permits and materialises when it does not, and `data_vec_*`
//! always owns. Kept apart from the type definition because choosing between
//! them is a performance decision the caller makes, not part of an image's
//! identity.

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
    /// `data_slice() -> Cow` contract. (`B: Default` follows from
    /// `Tensor::to_contiguous_on`'s own bound.)
    #[must_use]
    pub fn data_cow_on(&self, backend: &B) -> std::borrow::Cow<'_, [T]> {
        self.data.host_cow_on(backend)
    }

    /// Owned host image data in logical row-major order (layout-independent).
    ///
    /// Thin wrapper over [`Self::data_cow_on`] for callers that need a `Vec`
    /// (the `Image` type's `try_data_vec` equivalent).
    #[must_use]
    pub fn data_vec_on(&self, backend: &B) -> Vec<T> {
        self.data_cow_on(backend).into_owned()
    }

    /// Copy logical row-major image data into an owned host buffer.
    ///
    /// # Errors
    ///
    /// This canonical Coeus image contract materializes backend storage and
    /// non-contiguous views, so extraction succeeds for every valid image.
    pub fn try_data_vec_on(&self, backend: &B) -> anyhow::Result<Vec<T>> {
        Ok(self.data.to_vec_on(backend))
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
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

    /// [`Self::try_data_vec_on`] on `B::default()`.
    pub fn try_data_vec(&self) -> anyhow::Result<Vec<T>> {
        self.try_data_vec_on(&B::default())
    }
}

#[cfg(test)]
#[path = "tests_access.rs"]
mod tests;
