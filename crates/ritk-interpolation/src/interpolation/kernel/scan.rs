//! Shared `Interpolator::interpolate` driver.
//!
//! Owns the prologue every kernel repeats verbatim — the rank assertion, the
//! `[N, rank]` index-shape check and the two `to_contiguous()` calls — plus the
//! per-point loop. Each kernel supplies only its rank predicate and message and
//! the per-point closure.

use crate::interpolation::shared::compute_strides;
use coeus_core::{Backend, CpuAddressableStorage};
use coeus_tensor::Tensor;

/// Drive the shared `interpolate` prologue and per-point scan.
///
/// `supported_rank` and `rank_message` encode the kernel-specific rank
/// restriction. `point` is invoked once per output sample as
/// `point(data_slice, coords, shape, strides)`, where `data_slice` is the
/// row-major host buffer of `data`, `coords` is the innermost-first coordinate
/// row for the sample, `shape` is the row-major data shape, and `strides` its
/// row-major strides.
pub(super) fn scan<B, F>(
    data: &Tensor<f32, B>,
    indices: Tensor<f32, B>,
    supported_rank: impl Fn(usize) -> bool,
    rank_message: &str,
    mut point: F,
) -> Tensor<f32, B>
where
    B: Backend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    F: FnMut(&[f32], &[f32], &[usize], &[usize]) -> f32,
{
    let shape = data.shape().to_vec();
    let rank = shape.len();
    assert!(supported_rank(rank), "{rank_message}");

    let idx_shape = indices.shape();
    assert_eq!(idx_shape.len(), 2, "indices must be a 2D tensor [N, rank]");
    let n_points = idx_shape[0];
    let idx_rank = idx_shape[1];
    assert_eq!(idx_rank, rank, "indices rank must match data rank");

    let data_contig = data.to_contiguous();
    let data_slice = data_contig.as_slice();
    let idx_contig = indices.to_contiguous();
    let idx_slice = idx_contig.as_slice();

    let strides = compute_strides(&shape);
    let mut results = vec![0.0f32; n_points];
    for i in 0..n_points {
        let coords = &idx_slice[i * rank..(i + 1) * rank];
        results[i] = point(data_slice, coords, &shape, &strides);
    }

    Tensor::from_slice([n_points], &results)
}
