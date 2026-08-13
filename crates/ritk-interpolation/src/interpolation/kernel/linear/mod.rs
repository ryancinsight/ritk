//! Linear interpolation implementation.
//!
//! This module provides linear interpolation for 1D, 2D, 3D, and 4D data
//! using Coeus dynamic tensors. The implementation extracts data to the host,
//! performs the interpolation in CPU loops, and returns a Coeus tensor.

use super::BoundsPolicy;
use crate::interpolation::shared::OutOfBoundsMode;
use coeus_core::{Backend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_core::interpolation::Interpolator;
use serde::{Deserialize, Serialize};

/// Linear Interpolator.
///
/// Performs linear interpolation natively on the CPU.
/// When [`BoundsPolicy::Extend`] (the default), out-of-bounds coordinates are clamped to the
/// nearest edge voxel. When [`BoundsPolicy::ZeroPad`], out-of-bounds samples return `0.0`,
/// which prevents spurious correlation peaks in MI-based registration metrics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearInterpolator {
    /// Boundary handling policy. Default: `Extend`.
    pub bounds_policy: BoundsPolicy,
}

impl LinearInterpolator {
    /// Create a new linear interpolator with edge-clamping (default behaviour).
    pub fn new() -> Self {
        Self {
            bounds_policy: BoundsPolicy::Extend,
        }
    }

    /// Create a linear interpolator that returns `0.0` for out-of-bounds samples.
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

impl Default for LinearInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for LinearInterpolator
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn interpolate(&self, data: &Tensor<f32, B>, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        let mode = self.bounds_policy.as_out_of_bounds_mode();
        let shape = data.shape().to_vec();
        let rank = shape.len();
        assert!(
            (1..=MAX_RANK).contains(&rank),
            "Linear interpolation only supports 1D-{MAX_RANK}D data"
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

/// Highest data rank the linear kernel accepts.
///
/// Bounds both the entry check and the per-axis stack scratch, so the two
/// cannot drift apart. Four covers 3-D volumes with a time or channel axis;
/// beyond that the `2^rank` corner loop stops being a linear interpolation in
/// any practical sense.
const MAX_RANK: usize = 4;

/// Clamp a coordinate to the valid index range for an axis.
fn clamp_index(idx: f32, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let max = (size - 1) as f32;
    let clamped = idx.clamp(0.0, max);
    clamped as usize
}

/// The two bracketing indices and the interpolation weight along one axis.
///
/// Held together rather than in three parallel arrays because the corner loop
/// reads all three of an axis at once.
#[derive(Clone, Copy, Default)]
struct AxisBracket {
    lower: usize,
    upper: usize,
    /// Fractional distance from `lower` towards `upper`, in `[0, 1]`.
    weight: f32,
}

/// Linearly interpolate a single point.
fn interpolate_point(
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    coords: &[f32],
    mode: OutOfBoundsMode,
) -> f32 {
    let rank = shape.len();
    let zero_pad = mode == OutOfBoundsMode::ZeroPad;

    // Compute lower/upper integer indices and weights for each axis.
    // `coords` columns are innermost-first ([x, y, z]), while `shape` is
    // row-major ([z, y, x]); map axis `d` to coordinate `rank - 1 - d`.
    //
    // Stack scratch, not `vec!`: this runs once per output sample, so a
    // heap allocation here is one per voxel of a resampled volume. `MAX_RANK`
    // is the same bound `interpolate` already enforces on entry.
    let mut axes = [AxisBracket::default(); MAX_RANK];

    for (d, axis) in axes.iter_mut().take(rank).enumerate() {
        let size = shape[d];
        let coord = coords[rank - 1 - d];
        let floor = coord.floor();
        let frac = coord - floor;
        let floor_c = floor as isize;
        let upper_c = floor_c + 1;

        if zero_pad {
            if floor_c < 0 || floor_c > (size - 1) as isize {
                return 0.0;
            }
            axis.lower = floor_c.clamp(0, size as isize - 1) as usize;
            axis.upper = upper_c.clamp(0, size as isize - 1) as usize;
        } else {
            axis.lower = clamp_index(floor, size);
            axis.upper = clamp_index(upper_c as f32, size);
        }
        axis.weight = frac.clamp(0.0, 1.0);
    }
    let axes = &axes[..rank];

    // Iterate over all 2^rank corners and accumulate weighted values.
    let mut result = 0.0f32;
    let corners = 1usize << rank;
    for corner in 0..corners {
        let mut offset = 0usize;
        let mut weight = 1.0f32;
        for (d, axis) in axes.iter().enumerate() {
            let is_upper = (corner >> d) & 1 == 1;
            let idx = if is_upper { axis.upper } else { axis.lower };
            offset += idx * strides[d];
            weight *= if is_upper {
                axis.weight
            } else {
                1.0 - axis.weight
            };
        }
        result += data[offset] * weight;
    }

    result
}
