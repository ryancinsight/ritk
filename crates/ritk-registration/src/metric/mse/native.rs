//! Coeus-native MSE engine (`Image<f32, B, 3>` substrate).
//!
//! Atlas migration (burn → coeus): the register-engine parallel path for the
//! Mean Squared Error metric. The Burn-generic [`super::MeanSquaredError`]
//! surface stays unchanged (its consumers remain on Burn until their own
//! cutover); this module ADDS the native substrate alongside so registration's
//! eventual `Image<B>` → native cutover is unblocked.
//!
//! The resample path (fixed grid → native batch transforms → native affine →
//! native trilinear) is the shared [`super::super::native_resample`] substrate,
//! identical to the native NGF engine; only the reduction differs — here the
//! mean of the squared fixed/resampled-moving difference.
//!
//! 3-D only: the register engine operates on volumes, and the native trilinear
//! kernel is 3-D.

use super::super::native_resample::{fixed_world_points, resample_moving_at_world};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use ritk_transform::transform::affine::AtlasAffineTransform;

/// `MSE = (1/N) · Σ (Fixed(x) − Moving(T(x)))²` of `moving` resampled through
/// `transform` onto the `fixed` grid. The Coeus-native sister of
/// `MeanSquaredError::forward`.
pub fn mse_value_native<B>(
    fixed: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
    transform: &AtlasAffineTransform<B, 3>,
) -> f32
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let fixed_world = fixed_world_points(fixed);
    let f = fixed.data_vec();
    let m = resample_moving_at_world(&fixed_world, moving, transform);

    debug_assert_eq!(
        f.len(),
        m.len(),
        "invariant: resampled moving has one value per fixed voxel"
    );
    let n = f.len();
    let sum_sq: f32 = f
        .iter()
        .zip(m.iter())
        .map(|(&fv, &mv)| {
            let d = mv - fv;
            d * d
        })
        .sum();
    sum_sq / n as f32
}

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests_native;
