//! WGPU dispatch-limit constants and the chunked-row-apply helper.
//!
//! WGPU limits individual dispatch dimensions to 65 535 invocations.
//! All point-batch and tensor-row operations in `ritk-core` that may
//! exceed this limit must route through `apply_row_chunks`.
//!
//! # Constants
//! * [`WGPU_CHUNK_SIZE`]    — safe ceiling for standard 2-D/3-D operations.
//! * [`WGPU_CHUNK_SIZE_4D`] — reduced ceiling for 4-D B-spline kernels that
//!   issue more sub-dispatches per point than the 3-D path.

use coeus_core::{ComputeBackend, CpuAddressableStorageMut, Scalar};
use coeus_ops::cat;
use coeus_tensor::Tensor;

/// Maximum rows dispatched per shader invocation for 2-D/3-D operations.
///
/// Half the WGPU hard-limit (65 535) provides a conservative safe margin.
pub const WGPU_CHUNK_SIZE: usize = 32_768;

/// Maximum rows dispatched per shader invocation for the 4-D B-spline path.
///
/// The 4-D kernel issues more sub-dispatches per point than the 3-D path,
/// so a smaller ceiling is required to stay inside the WGPU limit.
pub const WGPU_CHUNK_SIZE_4D: usize = 16_384;

/// Apply `op` to `tensor` in chunks of at most `chunk_size` rows,
/// concatenating partial results into a single output tensor.
///
/// When `tensor.shape()[0] <= chunk_size` the operation is applied once
/// without any intermediate allocations.
///
/// # Arguments
/// * `tensor`     — Input tensor whose first axis is the row dimension.
/// * `chunk_size` — Maximum rows per invocation.
/// * `op`         — Function applied to each sub-tensor.
///
/// # Invariants
/// * `op` must preserve all dimensions except possibly the first.
/// * The first dimension of the output equals the first dimension of the input.
#[inline]
pub fn apply_row_chunks<T, B, F>(
    tensor: Tensor<T, B>,
    chunk_size: usize,
    op: F,
) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    F: Fn(Tensor<T, B>) -> Tensor<T, B>,
{
    let n = tensor.shape()[0];
    if n <= chunk_size {
        op(tensor)
    } else {
        let num_chunks = n.div_ceil(chunk_size);
        let mut chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n);
            let sliced = tensor.slice(&[(start, end)]);
            chunks.push(op(sliced));
        }
        let refs: Vec<&Tensor<T, B>> = chunks.iter().collect();
        cat(&refs, 0)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the WGPU chunked-row-apply helpers.
    use super::*;
    use coeus_core::SequentialBackend;

    type TestBackend = SequentialBackend;

    fn build_3d(n: usize) -> Tensor<f32, TestBackend> {
        let data: Vec<f32> = (0..n * 2 * 3).map(|i| i as f32).collect();
        let backend = TestBackend;
        Tensor::<f32, TestBackend>::from_slice_on([n, 2, 3], &data, &backend)
    }

    /// Double every element using an elementwise multiply kernel.
    fn scale_by_2(t: Tensor<f32, TestBackend>) -> Tensor<f32, TestBackend> {
        let b = TestBackend;
        let n = t.shape().iter().product::<usize>();
        let shape: Vec<usize> = t.shape().to_vec();
        let twos = Tensor::<f32, TestBackend>::from_slice_on(shape, &vec![2.0; n], &b);
        coeus_ops::mul(&t, &twos, &b)
    }

    #[test]
    fn apply_row_chunks_applies_once_below_limit() {
        let data = build_3d(4);
        let expected: Vec<f32> = data.as_slice().iter().map(|&v| v * 2.0).collect();

        let result = apply_row_chunks(data, 16, scale_by_2);

        assert_eq!(result.shape(), &[4, 2, 3]);
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn apply_row_chunks_preserves_uneven_row_order() {
        let data = build_3d(20);
        let expected: Vec<f32> = data.as_slice().iter().map(|&v| v * 2.0).collect();

        let result = apply_row_chunks(data, 8, scale_by_2);

        assert_eq!(result.shape(), &[20, 2, 3]);
        assert_eq!(result.as_slice(), expected.as_slice());
    }
}
