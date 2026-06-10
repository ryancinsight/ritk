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
