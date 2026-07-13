//! Coeus-native NGF engine (`Image<f32, B, 3>` substrate).
//!
//! Atlas migration (burn → coeus): the register-engine parallel path for the
//! Normalized Gradient Fields metric. The Burn-generic [`super::fixed_prep`]
//! /[`super::NormalizedGradientField`] surface stays unchanged (its consumers —
//! [`crate::ngf_rigid`], cli/python — remain on Burn until their own cutover);
//! this module ADDS the native substrate alongside so registration's eventual
//! `Image<B>` → native cutover is unblocked.
//!
//! The metric arithmetic is unchanged: the fixed/moving gradient fields, the
//! edge-noise scale `η`, and the weighted squared-normalized-dot reduction all
//! flow through the same host-slice functions in [`super::scalar`]. Only the
//! substrate that produces the resampled moving volume differs — Coeus batch
//! point transforms ([`Image::index_to_world_native`]/
//! [`Image::world_to_index_native`], differential-verified bit-faithful to the
//! Burn `index_to_world_tensor`/`world_to_index_tensor`), the native affine
//! [`AtlasAffineTransform`], and the native
//! [`trilinear_interpolation`](ritk_interpolation::native::trilinear_interpolation)
//! kernel replace the Burn grid/tensor/`LinearInterpolator` path.
//!
//! 3-D only: the register engine operates on volumes, and the native trilinear
//! kernel is 3-D. This mirrors [`super::fixed_prep::NgfFixedPrep`]'s dense path
//! (`sampling: None`); the stochastic-sample estimator has no native consumer
//! yet and is left on Burn (recorded as a residual gap).

use super::super::native_resample::{fixed_world_points, resample_moving_at_world};
use super::scalar::{
    compute_gradient_field, row_major_strides, weighted_eta2, weighted_ngf_from_fixed,
};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use ritk_transform::transform::affine::AtlasAffineTransform;

/// Precomputed native fixed-image NGF state for repeated transform evaluations.
///
/// The Coeus-native sister of [`super::fixed_prep::NgfFixedPrep`]'s dense path:
/// building it ONCE removes the fixed-grid generation, the index→world mapping,
/// the fixed host read, the fixed gradient field, and `η_F` from the optimiser's
/// hot loop. Each [`eval`](Self::eval) resamples only the moving image and
/// computes its gradient.
pub struct NgfFixedPrepNative<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// World coordinates of every fixed-grid voxel, `[N, 3]` axis-major, flat.
    fixed_world: Vec<f32>,
    shape: [usize; 3],
    stride: [usize; 3],
    gf: Vec<[f32; 3]>,
    eta_f2: f32,
    mask: Option<Vec<bool>>,
    weights: Option<Vec<f32>>,
    _backend: std::marker::PhantomData<B>,
}

impl<B> NgfFixedPrepNative<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Precompute the native fixed-image state for `fixed`, restricted to `mask`
    /// and scaled by `weights` (both row-major, fixed-image C-order).
    pub fn new(
        fixed: &Image<f32, B, 3>,
        mask: Option<&[bool]>,
        weights: Option<&[f32]>,
    ) -> Self {
        let shape = fixed.shape();
        let stride = row_major_strides(&shape);

        let fixed_world = fixed_world_points(fixed);

        let f = fixed.data_vec();
        let gf = compute_gradient_field(&f, &shape, &stride);
        let eta_f2 = weighted_eta2(&gf, mask, weights);

        Self {
            fixed_world,
            shape,
            stride,
            gf,
            eta_f2,
            mask: mask.map(<[bool]>::to_vec),
            weights: weights.map(<[f32]>::to_vec),
            _backend: std::marker::PhantomData,
        }
    }

    /// `NGF ∈ [0, 1]` of `moving` resampled through `transform` onto the fixed
    /// grid, reusing the precomputed fixed state.
    pub fn eval(
        &self,
        moving: &Image<f32, B, 3>,
        transform: &AtlasAffineTransform<B, 3>,
    ) -> f32 {
        let m = self.resample(moving, transform);
        weighted_ngf_from_fixed(
            &self.gf,
            self.eta_f2,
            &m,
            &self.shape,
            &self.stride,
            self.mask.as_deref(),
            self.weights.as_deref(),
        )
    }

    /// Resample `moving` through `transform` at the fixed-grid world points,
    /// returning the `N` interpolated host values in fixed C-order.
    fn resample(&self, moving: &Image<f32, B, 3>, transform: &AtlasAffineTransform<B, 3>) -> Vec<f32> {
        resample_moving_at_world(&self.fixed_world, moving, transform)
    }
}

/// One-shot native NGF: `NGF ∈ [0, 1]` of `moving` resampled through `transform`
/// onto the `fixed` grid over the `true` voxels of `mask` (or all if `None`),
/// each masked voxel scaled by `weights` (or 1). The Coeus-native sister of
/// [`super::NormalizedGradientField::ngf_value_weighted`]; the registration hot
/// loop builds [`NgfFixedPrepNative`] ONCE instead.
pub fn ngf_value_native<B>(
    fixed: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
    transform: &AtlasAffineTransform<B, 3>,
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> f32
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    NgfFixedPrepNative::new(fixed, mask, weights).eval(moving, transform)
}

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests_native;
