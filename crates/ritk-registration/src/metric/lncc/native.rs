//! Coeus-native LNCC engine (`Image<f32, B, 3>` substrate).
//!
//! Atlas migration (burn → coeus): the register-engine parallel path for the
//! Local Normalized Cross Correlation. The Burn-generic
//! [`super::LocalNormalizedCrossCorrelation`] surface stays unchanged (its
//! consumers remain on Burn until their own cutover); this module ADDS the
//! native substrate alongside so registration's eventual `Image<B>` → native
//! cutover is unblocked.
//!
//! The moving resample is the shared [`super::super::native_resample`] substrate;
//! the local means/variances/covariance are the burn-free separable Gaussian
//! [`ritk_filter::gaussian::gaussian_smooth_flat_3d`] (the flat-buffer sister of
//! the Burn `conv1d` path both metrics use). All arithmetic runs on flat host
//! buffers — no Burn tensor, no native `Image` reconstruction. The reduction is
//! identical to the Burn engine (Cachier et al. 2003): local covariance over the
//! geometric mean of local variances, negated and averaged.
//!
//! 3-D only: the register engine operates on volumes, and both the native
//! trilinear kernel and the flat Gaussian are 3-D.

use super::super::native_resample::{fixed_world_points, resample_moving_at_world};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_filter::gaussian::gaussian_smooth_flat_3d;
use ritk_filter::GaussianSigma;
use ritk_image::native::Image;
use ritk_transform::transform::affine::AtlasAffineTransform;

/// Local mean `μ = K∗I` and local variance `v = max(K∗I² − μ², 0)` of the flat
/// z-major volume `vals` (with its precomputed square `sq`), via the shared
/// burn-free separable Gaussian — the flat-buffer sister of the Burn
/// `compute_local_stats`.
fn local_stats(
    vals: &[f32],
    sq: &[f32],
    dims: [usize; 3],
    sigmas: [GaussianSigma; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>) {
    let mean = gaussian_smooth_flat_3d(vals, dims, sigmas, spacing);
    let mean_sq = gaussian_smooth_flat_3d(sq, dims, sigmas, spacing);
    let var: Vec<f32> = mean_sq
        .iter()
        .zip(mean.iter())
        .map(|(&ms, &mu)| (ms - mu * mu).max(0.0))
        .collect();
    (mean, var)
}

/// LNCC framed as a minimization loss (`−mean(LNCC)`) of `moving` resampled
/// through `transform` onto the `fixed` grid, with local windows defined by a
/// Gaussian of `kernel_sigma`. The Coeus-native sister of
/// [`super::LocalNormalizedCrossCorrelation::forward`].
///
/// `LNCC = cov / (√(v_F · v_M) + ε)` per voxel, where `μ, v, cov` are the local
/// Gaussian-weighted mean/variance/covariance (Cachier et al. 2003); the returned
/// value is `−mean(LNCC)`. `epsilon` matches the Burn engine default (`1e-5`).
pub fn lncc_loss_native<B>(
    fixed: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
    transform: &AtlasAffineTransform<B, 3>,
    kernel_sigma: GaussianSigma,
    epsilon: f32,
) -> f32
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let f = fixed.data_vec();
    let fixed_world = fixed_world_points(fixed);
    let m = resample_moving_at_world(&fixed_world, moving, transform);

    let dims = fixed.shape();
    let sp = fixed.spacing();
    let spacing = [sp[0], sp[1], sp[2]];
    let sigmas = [kernel_sigma; 3];

    let f_sq: Vec<f32> = f.iter().map(|&v| v * v).collect();
    let m_sq: Vec<f32> = m.iter().map(|&v| v * v).collect();
    let (mean_f, var_f) = local_stats(&f, &f_sq, dims, sigmas, spacing);
    let (mean_m, var_m) = local_stats(&m, &m_sq, dims, sigmas, spacing);

    // Cross term: cov = K∗(F·M) − μ_F·μ_M.
    let fm: Vec<f32> = f.iter().zip(m.iter()).map(|(&a, &b)| a * b).collect();
    let mean_fm = gaussian_smooth_flat_3d(&fm, dims, sigmas, spacing);

    let n = f.len();
    let mut acc = 0.0f32;
    for i in 0..n {
        let cov = mean_fm[i] - mean_f[i] * mean_m[i];
        let denom = (var_f[i] * var_m[i]).sqrt() + epsilon;
        acc += cov / denom;
    }

    -(acc / n as f32)
}

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests_native;
