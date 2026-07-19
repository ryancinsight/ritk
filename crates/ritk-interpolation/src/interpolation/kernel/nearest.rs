//! Nearest neighbor interpolation implementation.
//!
//! This module provides nearest neighbor interpolation for 1D, 2D, 3D, and 4D data
//! using Coeus dynamic tensors. The implementation extracts data to the host,
//! performs the interpolation in CPU loops, and returns a Coeus tensor.

use super::BoundsPolicy;
use crate::interpolation::shared::OutOfBoundsMode;
use coeus_core::{Backend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_core::interpolation::Interpolator;
use serde::{Deserialize, Serialize};

/// Nearest Neighbor Interpolator.
///
/// Performs nearest neighbor interpolation (rounds to nearest integer coordinate).
///
/// When [`BoundsPolicy::Extend`] (the default), out-of-bounds samples clamp to the nearest
/// edge voxel. When [`BoundsPolicy::ZeroPad`], out-of-bounds samples return `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NearestNeighborInterpolator {
    /// Boundary handling policy. Default: `Extend`.
    pub bounds_policy: BoundsPolicy,
}

impl NearestNeighborInterpolator {
    /// Create a new nearest neighbor interpolator with edge-clamping (default).
    pub fn new() -> Self {
        Self {
            bounds_policy: BoundsPolicy::Extend,
        }
    }

    /// Create a nearest neighbor interpolator that returns `0.0` for OOB samples.
    pub fn new_zero_pad() -> Self {
        Self {
            bounds_policy: BoundsPolicy::ZeroPad,
        }
    }

    /// Builder-style setter for the bounds policy.
    pub fn with_bounds_policy(mut self, policy: BoundsPolicy) -> Self {
        self.bounds_policy = policy;
        self
    }
}

impl Default for NearestNeighborInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for NearestNeighborInterpolator
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn interpolate(&self, data: &Tensor<f32, B>, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        let mode = self.bounds_policy.as_out_of_bounds_mode();
        let shape = data.shape().to_vec();
        let rank = shape.len();
        assert!(
            (1..=4).contains(&rank),
            "Nearest-neighbor interpolation only supports 1D-4D data"
        );

        let idx_shape = indices.shape();
        assert_eq!(idx_shape.len(), 2, "indices must be a 2D tensor [N, rank]");
        let n_points = idx_shape[0];
        let idx_rank = idx_shape[1];
        assert_eq!(idx_rank, rank, "indices rank must match data rank");

        let data_contig = data.to_contiguous();
        let data_slice = data_contig.as_slice();
        let idx_contig = indices.to_contiguous();
        let idx_slice = idx_contig.as_slice();

        let mut results = vec![0.0f32; n_points];
        let strides = compute_strides(&shape);

        for i in 0..n_points {
            let coords = &idx_slice[i * rank..(i + 1) * rank];
            results[i] = interpolate_point(data_slice, &shape, &strides, coords, mode);
        }

        Tensor::from_slice([n_points], &results)
    }
}

/// Compute row-major strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Clamp a coordinate to the valid index range for an axis.
fn clamp_index(idx: f32, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let max = (size - 1) as f32;
    let clamped = idx.clamp(0.0, max);
    clamped as usize
}

/// Nearest-neighbor interpolate a single point.
fn interpolate_point(
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    coords: &[f32],
    mode: super::OutOfBoundsMode,
) -> f32 {
    let rank = shape.len();
    let zero_pad = mode == OutOfBoundsMode::ZeroPad;

    // `coords` columns are innermost-first ([x, y, z]), while `shape` is
    // row-major ([z, y, x]); map axis `d` to coordinate `rank - 1 - d`.
    let mut offset = 0usize;
    for d in 0..rank {
        let size = shape[d];
        let coord = coords[rank - 1 - d];
        let rounded = (coord + 0.5).floor();
        let idx = if zero_pad {
            if rounded < 0.0 || rounded >= size as f32 {
                return 0.0;
            }
            rounded as usize
        } else {
            clamp_index(rounded, size)
        };
        offset += idx * strides[d];
    }

    data[offset]
}
