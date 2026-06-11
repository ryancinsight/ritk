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

use burn::tensor::{backend::Backend, Tensor};

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
/// When `tensor.dims()[0] <= chunk_size` the operation is applied once
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
#[allow(clippy::single_range_in_vec_init)] // Burn tensor slice() takes [Range; D] per rank
#[inline]
pub(crate) fn apply_row_chunks<B: Backend, const D: usize, F>(
    tensor: Tensor<B, D>,
    chunk_size: usize,
    op: F,
) -> Tensor<B, D>
where
    F: Fn(Tensor<B, D>) -> Tensor<B, D>,
{
    let n = tensor.dims()[0];
    if n <= chunk_size {
        op(tensor)
    } else {
        let num_chunks = n.div_ceil(chunk_size);
        let mut chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n);
            chunks.push(op(tensor.clone().slice([start..end])));
        }
        Tensor::cat(chunks, 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  3-D specialized variant (audit §8 351-03)
// ═══════════════════════════════════════════════════════════════════════
//
// The generic [`apply_row_chunks`] above takes a closure `F: Fn(...)`
// which is monomorphized per call site. For closures that *capture*
// state (e.g. `|chunk| conv1d(chunk, kernel.clone(), options.clone())`),
// the monomorphized call goes through a stored function pointer in the
// closure struct — one indirect call per chunk. For the 3-D hot path
// in the interpolation kernels, this is a measurable overhead.
//
// [`apply_row_chunks_3d`] addresses this in two ways:
//
// 1. **Const-generic chunk size** (`const CHUNK: usize`): the chunk
//    size is a compile-time constant, so the comparison `n <= CHUNK`,
//    the slice bounds `start..end`, and the `n.div_ceil(CHUNK)`
//    computation can be constant-folded by the compiler when `n` is
//    also known. More importantly, the function is monomorphized per
//    `CHUNK` value, so the compiler can specialize the generated code
//    for each chunk size (e.g. `WGPU_CHUNK_SIZE` vs.
//    `WGPU_CHUNK_SIZE_4D`).
//
// 2. **Function pointer instead of closure** (`fn(...)` not `F: Fn(...)`):
//    the operation is a raw function pointer, not a generic closure.
//    Function pointer calls are one indirection (a direct call through
//    the pointer) rather than the two indirections of a capturing
//    closure (struct dereference + function pointer call). The
//    compiler can also inline the function through the pointer when
//    the pointer is known at the call site (LTO or
//    `#[inline]`-annotated functions).
//
// # When to use
//
// Use [`apply_row_chunks_3d`] when:
// - The tensor is 3-D (the 3-D slice pattern is hardcoded for clarity
//   and potential compiler optimization)
// - The operation is a *non-capturing* function (no closure state)
// - The chunk size is a compile-time constant (typically
//   [`WGPU_CHUNK_SIZE`] or [`WGPU_CHUNK_SIZE_4D`])
//
// For capturing closures or runtime chunk sizes, fall back to the
// generic [`apply_row_chunks`].

/// Specialized 3-D variant of [`apply_row_chunks`] that takes the
/// chunk size as a const generic and the operation as a function
/// pointer.
///
/// See the module-level comment for the full design rationale.
#[allow(dead_code)] // Public API; call sites are in-progress (audit §8 351-03 follow-up).
#[inline]
pub(crate) fn apply_row_chunks_3d<B: Backend, const CHUNK: usize>(
    tensor: Tensor<B, 3>,
    op: fn(Tensor<B, 3>) -> Tensor<B, 3>,
) -> Tensor<B, 3> {
    let n = tensor.dims()[0];
    if n <= CHUNK {
        op(tensor)
    } else {
        let num_chunks = n.div_ceil(CHUNK);
        let mut chunks = Vec::with_capacity(num_chunks);
        // Read dims once before the loop to avoid repeated
        // `tensor.dims()` calls in the hot path.
        let dims = tensor.dims();
        for i in 0..num_chunks {
            let start = i * CHUNK;
            let end = (start + CHUNK).min(n);
            // Explicit 3-D slice pattern (vs. the generic
            // `slice([start..end])` in `apply_row_chunks`). The
            // compiler can see the full rank and the axis-1/axis-2
            // bounds at compile time (they don't depend on `i`), so
            // the slice bounds for axes 1 and 2 are hoisted out of
            // the loop.
            chunks.push(op(tensor.clone().slice([start..end, 0..dims[1], 0..dims[2]])));
        }
        Tensor::cat(chunks, 0)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the WGPU chunked-row-apply helpers.
    use super::*;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    /// Identity operation: returns the tensor unchanged. Used to test
    /// the chunked-apply mechanics without depending on Burn internals.
    fn identity_op(t: Tensor<TestBackend, 3>) -> Tensor<TestBackend, 3> {
        t
    }

    /// Doubles every element. Tests that the per-chunk operation is
    /// applied to each sub-tensor.
    fn double_op(t: Tensor<TestBackend, 3>) -> Tensor<TestBackend, 3> {
        t * 2.0
    }

    fn build_3d(n: usize) -> Tensor<TestBackend, 3> {
        let data: Vec<f32> = (0..n * 2 * 3).map(|i| i as f32).collect();
        let device = Default::default();
        Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data, burn::tensor::Shape::new([n, 2, 3])),
            &device,
        )
    }

    #[test]
    fn apply_row_chunks_3d_no_chunk_when_under_limit() {
        // n=4 < CHUNK=16: single op call, no chunking.
        let data = build_3d(4);
        let result = apply_row_chunks_3d::<TestBackend, 16>(data.clone(), double_op);
        let dims = result.dims();
        assert_eq!(dims[0], 4);
        assert_eq!(dims[1], 2);
        assert_eq!(dims[2], 3);
        // Verify double_op was applied.
        let original = data.into_data().as_slice::<f32>().unwrap().to_vec();
        let doubled = result.into_data().as_slice::<f32>().unwrap().to_vec();
        for (a, b) in original.iter().zip(doubled.iter()) {
            assert!((a * 2.0 - b).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_row_chunks_3d_chunks_when_over_limit() {
        // n=32 > CHUNK=16: 2 chunks of 16 rows each.
        let data = build_3d(32);
        let result = apply_row_chunks_3d::<TestBackend, 16>(data.clone(), double_op);
        let dims = result.dims();
        assert_eq!(dims[0], 32, "Output should preserve first-dim size");
        assert_eq!(dims[1], 2);
        assert_eq!(dims[2], 3);
        // Verify all elements are doubled.
        let original = data.into_data().as_slice::<f32>().unwrap().to_vec();
        let doubled = result.into_data().as_slice::<f32>().unwrap().to_vec();
        for (a, b) in original.iter().zip(doubled.iter()) {
            assert!((a * 2.0 - b).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_row_chunks_3d_handles_uneven_last_chunk() {
        // n=20, CHUNK=16: 2 chunks (16 + 4 rows).
        let data = build_3d(20);
        let result = apply_row_chunks_3d::<TestBackend, 16>(data.clone(), double_op);
        let dims = result.dims();
        assert_eq!(dims[0], 20, "Output should preserve first-dim size");
        let original = data.into_data().as_slice::<f32>().unwrap().to_vec();
        let doubled = result.into_data().as_slice::<f32>().unwrap().to_vec();
        for (a, b) in original.iter().zip(doubled.iter()) {
            assert!((a * 2.0 - b).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_row_chunks_3d_identity_preserves_data() {
        // Identity op: verify the concat logic preserves all elements
        // in the correct order.
        let data = build_3d(20);
        let result = apply_row_chunks_3d::<TestBackend, 8>(data.clone(), identity_op);
        let original = data.into_data().as_slice::<f32>().unwrap().to_vec();
        let identity = result.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(original, identity, "Identity op should preserve data");
    }

    #[test]
    fn apply_row_chunks_3d_matches_generic_for_identity() {
        // Verify the 3-D specialized variant produces the same result
        // as the generic `apply_row_chunks` for an identity operation.
        let data = build_3d(40);
        let specialized = apply_row_chunks_3d::<TestBackend, 16>(data.clone(), identity_op);
        let generic = apply_row_chunks(data.clone(), 16usize, identity_op);
        let specialized_slice = specialized.into_data().as_slice::<f32>().unwrap().to_vec();
        let generic_slice = generic.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(
            specialized_slice, generic_slice,
            "Specialized 3-D variant should match generic for identity"
        );
    }
}
