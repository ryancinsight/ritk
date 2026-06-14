//! Fast host-side `f32` extraction capability for CPU tensor backends.
//!
//! # Motivation
//! The portable `Tensor::into_data()` path materializes a `TensorData` buffer.
//! On the NdArray backend this costs roughly 20 ms per million voxels — far more
//! than a memcpy — because it allocates and round-trips through a byte buffer.
//! The backend's own contiguous host array is reachable in ~0.08 ms/Mvoxel via
//! `into_primitive()` + `as_slice_memory_order()`, but that path is
//! backend-specific and cannot be expressed against the generic `Backend` trait.
//!
//! # Design
//! [`HostExtract`] is a capability trait in the spirit of the canonical
//! `ComputeBackend`/`ExecutionPolicy` seams: CPU backends that can hand out a
//! contiguous `&[f32]` host view advertise it, and generic algorithms that need
//! throughput request it with a `B: HostExtract` bound.  Backends without a
//! host-addressable buffer (e.g. a future GPU backend) simply do not implement
//! it, so those algorithms fall back to the portable `into_data()` path at their
//! own call sites.

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_ndarray::{NdArray, NdArrayTensor};

use crate::Image;

/// Tensor backends that expose a contiguous `&[f32]` host view without the
/// `into_data()` materialization.
pub trait HostExtract: Backend {
    /// Invoke `f` with a borrowed contiguous `&[f32]` view of `tensor`'s data,
    /// using the fastest host-access path the backend supports.
    fn with_host_f32<const D: usize, R>(
        tensor: &Tensor<Self, D>,
        f: impl FnOnce(&[f32]) -> R,
    ) -> R;

    /// Extract `tensor`'s data as an owned `Vec<f32>` (a single host copy).
    #[inline]
    fn host_f32_vec<const D: usize>(tensor: &Tensor<Self, D>) -> Vec<f32> {
        Self::with_host_f32(tensor, <[f32]>::to_vec)
    }
}

impl HostExtract for NdArray<f32> {
    #[inline]
    fn with_host_f32<const D: usize, R>(
        tensor: &Tensor<Self, D>,
        f: impl FnOnce(&[f32]) -> R,
    ) -> R {
        // `clone()` is an Arc refcount bump (O(1)); `into_primitive()` unwraps to
        // the backend array without copying; `as_slice_memory_order()` borrows
        // the contiguous buffer (O(1)).  Burn's NdArray always builds
        // C-contiguous arrays, so the non-contiguous branch is unreachable in
        // practice but kept for soundness.
        match tensor.clone().into_primitive() {
            TensorPrimitive::Float(NdArrayTensor::F32(arc_array)) => {
                if let Some(slice) = arc_array.as_slice_memory_order() {
                    f(slice)
                } else {
                    let owned: Vec<f32> = arc_array.iter().copied().collect();
                    f(&owned)
                }
            }
            _ => unreachable!("NdArray<f32> float primitive is always NdArrayTensor::F32"),
        }
    }
}

impl HostExtract for Autodiff<NdArray<f32>> {
    #[inline]
    fn with_host_f32<const D: usize, R>(
        tensor: &Tensor<Self, D>,
        f: impl FnOnce(&[f32]) -> R,
    ) -> R {
        // Strip the autodiff tape (also an Arc-cheap operation) and reuse the
        // inner NdArray host path.
        <NdArray<f32> as HostExtract>::with_host_f32(&tensor.clone().inner(), f)
    }
}

impl<B: HostExtract, const D: usize> Image<B, D> {
    /// Extract the image data as a `Vec<f32>` via the backend's fast host path
    /// (zero-copy borrow + one copy), bypassing the `into_data()` materialization
    /// that [`Image::try_data_vec`](crate::Image::try_data_vec) uses.
    #[inline]
    pub fn data_vec_fast(&self) -> Vec<f32> {
        B::host_f32_vec(self.data())
    }

    /// Call `f` with a borrowed contiguous `&[f32]` view of the image data via
    /// the backend's fast host path (no allocation on contiguous CPU backends).
    #[inline]
    pub fn with_data_slice_fast<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        B::with_host_f32(self.data(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn ndarray_host_vec_matches_into_data() {
        let device = Default::default();
        let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
        let t = Tensor::<B, 2>::from_data(
            TensorData::new(values.clone(), [4, 6]),
            &device,
        );
        assert_eq!(B::host_f32_vec(&t), values);
        let sum = B::with_host_f32(&t, |s| s.iter().sum::<f32>());
        assert_eq!(sum, values.iter().sum::<f32>());
    }

    #[test]
    fn autodiff_host_vec_matches_inner() {
        type AB = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let values: Vec<f32> = vec![1.0, -2.0, 3.5, 4.0];
        let t = Tensor::<AB, 1>::from_data(TensorData::new(values.clone(), [4]), &device);
        assert_eq!(AB::host_f32_vec(&t), values);
    }
}
