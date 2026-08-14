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
use ritk_image::tensor_view;
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
        let rank = data.ndim();
        assert!(
            (1..=MAX_RANK).contains(&rank),
            "Linear interpolation only supports 1D-{MAX_RANK}D data"
        );

        let idx_shape = indices.shape();
        assert_eq!(idx_shape.len(), 2, "indices must be a 2D tensor [N, rank]");
        assert_eq!(idx_shape[1], rank, "indices rank must match data rank");

        // Rank dispatch happens once, here, and monomorphizes the whole kernel
        // body below it: the per-axis loops become fixed-trip, the scratch
        // becomes exactly `[_; N]`, and no per-sample work inspects the rank.
        match rank {
            1 => sample_all::<1, B>(data, &indices, mode),
            2 => sample_all::<2, B>(data, &indices, mode),
            3 => sample_all::<3, B>(data, &indices, mode),
            4 => sample_all::<4, B>(data, &indices, mode),
            _ => unreachable!("the entry assertion bounds rank to 1..={MAX_RANK}"),
        }
    }
}

/// Sample every point of a `[point_count, N]` index tensor against rank-`N` data.
///
/// Both operands are borrowed through [`tensor_view`], so an arbitrarily
/// strided or offset input costs no copy: the layout is carried in the view's
/// strides and consumed by the offset arithmetic below. The previous
/// `to_contiguous()` pair materialized the entire volume *and* the entire index
/// tensor before reading either.
fn sample_all<const N: usize, B>(
    data: &Tensor<f32, B>,
    indices: &Tensor<f32, B>,
    mode: OutOfBoundsMode,
) -> Tensor<f32, B>
where
    B: Backend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let data_view = tensor_view::<f32, B, N>(data)
        .expect("invariant: CpuAddressableStorage data of rank N views as rank N");
    let index_view = tensor_view::<f32, B, 2>(indices)
        .expect("invariant: CpuAddressableStorage indices of rank 2 view as rank 2");

    let shape = data_view.shape();
    let data_strides = data_view.strides();
    let data_base = data_view.offset();
    let values = data_view.data();

    let point_count = index_view.shape()[0];
    let index_strides = index_view.strides();
    let index_base = index_view.offset();
    let index_values = index_view.data();

    let results = (0..point_count)
        .map(|point| {
            let row = index_base as isize + index_strides[0] * point as isize;
            let coords: [f32; N] = std::array::from_fn(|axis| {
                index_values[(row + index_strides[1] * axis as isize) as usize]
            });
            interpolate_point::<N>(values, shape, data_strides, data_base, coords, mode)
        })
        .collect::<Vec<_>>();

    Tensor::from_slice([point_count], &results)
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

/// Linearly interpolate a single point of rank-`N` data.
///
/// `strides` and `base` come from the source tensor's layout rather than from
/// a row-major recomputation, so the kernel reads a strided or offset volume
/// in place. `data` is the whole storage slice and `base` its logical origin.
fn interpolate_point<const N: usize>(
    data: &[f32],
    shape: [usize; N],
    strides: [isize; N],
    base: usize,
    coords: [f32; N],
    mode: OutOfBoundsMode,
) -> f32 {
    let zero_pad = mode == OutOfBoundsMode::ZeroPad;

    // Compute lower/upper integer indices and weights for each axis.
    // `coords` columns are innermost-first ([x, y, z]), while `shape` is
    // row-major ([z, y, x]); map axis `d` to coordinate `N - 1 - d`.
    //
    // Stack scratch rather than `vec!`: this runs once per output sample, and
    // the vectors were measured at three real allocations per call — 300000
    // for 100000 points, not elided. Removing them is speed-neutral on one
    // thread (the allocator fast path is not this kernel's bottleneck), so the
    // reason to keep them off the heap is allocator traffic under parallel
    // resampling, which a single-threaded benchmark cannot show.
    //
    // Three flat arrays, not one array of structs: the struct-of-arrays layout
    // measured 2x faster than packing the three fields together, so the
    // original layout stays. The buffers are `[_; N]` rather than `[_; MAX_RANK]`
    // because the rank dispatch in `interpolate` monomorphizes this body, so
    // they are exactly as long as the loops that fill them.
    let mut lower = [0usize; N];
    let mut upper = [0usize; N];
    let mut weights = [0.0f32; N];

    for d in 0..N {
        let size = shape[d];
        let coord = coords[N - 1 - d];
        let floor = coord.floor();
        let frac = coord - floor;
        let floor_c = floor as isize;
        let upper_c = floor_c + 1;

        if zero_pad {
            if floor_c < 0 || floor_c > (size - 1) as isize {
                return 0.0;
            }
            lower[d] = floor_c.clamp(0, size as isize - 1) as usize;
            upper[d] = upper_c.clamp(0, size as isize - 1) as usize;
        } else {
            lower[d] = clamp_index(floor, size);
            upper[d] = clamp_index(upper_c as f32, size);
        }
        weights[d] = frac.clamp(0.0, 1.0);
    }

    // Iterate over all 2^N corners and accumulate weighted values.
    let mut result = 0.0f32;
    let corners = 1usize << N;
    for corner in 0..corners {
        let mut offset = base as isize;
        let mut weight = 1.0f32;
        for d in 0..N {
            let is_upper = (corner >> d) & 1 == 1;
            let index = if is_upper { upper[d] } else { lower[d] };
            offset += index as isize * strides[d];
            weight *= if is_upper {
                weights[d]
            } else {
                1.0 - weights[d]
            };
        }
        result += data[offset as usize] * weight;
    }

    result
}
