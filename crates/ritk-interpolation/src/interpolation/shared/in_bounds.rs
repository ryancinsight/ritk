//! In-bounds mask computation for out-of-bounds zero-padding.
//!
//! Shared out-of-bounds handling policy for interpolation kernels.

use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::{Backend, Tensor};

/// Compute a `{0.0, 1.0}` in-bounds mask for voxel indices.
///
/// Returns an `[N]` float tensor: `1.0` for in-bounds coordinates and `0.0`
/// otherwise. Index columns are innermost-first (`[x, y, z]`) while `shape`
/// remains row-major (`[z, y, x]`).
#[must_use]
pub fn compute_oob_mask<B: Backend>(
    indices: &Tensor<f32, B>,
    shape: &[usize],
) -> Tensor<f32, B>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let idx_shape = indices.shape();
    assert_eq!(idx_shape.len(), 2, "indices must be rank-2 [N, D]");
    let n = idx_shape[0];
    let rank = idx_shape[1];
    assert_eq!(rank, shape.len(), "index rank must match image rank");

    let indices = indices.to_contiguous();
    let values = indices.as_slice();
    let mut mask = vec![1.0; n];
    for sample in 0..n {
        for axis in 0..rank {
            let coordinate = values[sample * rank + (rank - 1 - axis)];
            let floor = coordinate.floor() as isize;
            if floor < 0 || floor >= shape[axis] as isize {
                mask[sample] = 0.0;
                break;
            }
        }
    }
    Tensor::from_slice([n], &mask)
}

/// Out-of-bounds handling policy for interpolation kernels.
///
/// - `ZeroPad`: coordinates outside the image extent produce output 0.
/// - `Clamp`: coordinates outside the image extent are clamped to the
///   nearest valid boundary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfBoundsMode {
    /// Return 0 for out-of-bounds coordinates.
    ZeroPad,
    /// Clamp out-of-bounds coordinates to the nearest valid boundary.
    Clamp,
}
