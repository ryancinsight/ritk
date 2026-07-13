//! Coeus-native NCC engine (`Image<f32, B, 3>` substrate).
//!
//! Atlas migration (burn → coeus): the register-engine parallel path for the
//! zero-normalized cross correlation. The Burn-generic
//! [`super::NormalizedCrossCorrelation`] surface stays unchanged (its consumers
//! remain on Burn until their own cutover); this module ADDS the native
//! substrate alongside so registration's eventual `Image<B>` → native cutover
//! is unblocked.
//!
//! The resample path is the shared [`super::super::native_resample`] substrate;
//! only the reduction differs — the identical single-pass five-moment ZNCC
//! (Lewis 1995) as the Burn engine, evaluated on the resampled host values.
//!
//! 3-D only: the register engine operates on volumes, and the native trilinear
//! kernel is 3-D.

use super::super::native_resample::{fixed_world_points, resample_moving_at_world};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use ritk_transform::transform::affine::AtlasAffineTransform;

/// Variance clamp guaranteeing a finite denominator for constant/identical
/// inputs — identical to the Burn engine's `clamp_min(epsilon)`.
const NCC_EPS: f32 = 1e-10;

/// Zero-normalized cross correlation framed as a minimization loss (`−NCC`) of
/// `moving` resampled through `transform` onto the `fixed` grid. The Coeus-native
/// sister of [`super::NormalizedCrossCorrelation::forward`].
///
/// Single pass over the `N` voxels accumulates the five raw moments
/// `ΣF, ΣM, ΣF², ΣM², ΣFM`; the central-moment reduction
/// `NCC = (ΣFM − ΣF·ΣM/N) / √((ΣF² − ΣF²/N)·(ΣM² − ΣM²/N))` then follows
/// (Lewis 1995), with each variance clamped to [`NCC_EPS`]. Returns `−NCC ∈ [−1, 1]`.
pub fn ncc_loss_native<B>(
    fixed: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
    transform: &AtlasAffineTransform<B, 3>,
) -> f32
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let fixed_world = fixed_world_points(fixed);
    let f = fixed.data_vec();
    let m = resample_moving_at_world(&fixed_world, moving, transform);

    debug_assert_eq!(
        f.len(),
        m.len(),
        "invariant: resampled moving has one value per fixed voxel"
    );
    let n = f.len() as f32;

    let (mut s_f, mut s_m, mut s_ff, mut s_mm, mut s_fm) = (0.0f32, 0.0, 0.0, 0.0, 0.0);
    for (&fv, &mv) in f.iter().zip(m.iter()) {
        s_f += fv;
        s_m += mv;
        s_ff += fv * fv;
        s_mm += mv * mv;
        s_fm += fv * mv;
    }

    let num = s_fm - s_f * s_m / n;
    let var_f = (s_ff - s_f * s_f / n).max(NCC_EPS);
    let var_m = (s_mm - s_m * s_m / n).max(NCC_EPS);
    let ncc = num / (var_f * var_m).sqrt();

    -ncc
}

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests_native;
