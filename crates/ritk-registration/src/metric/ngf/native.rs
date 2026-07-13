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

use super::scalar::{
    compute_gradient_field, row_major_strides, weighted_eta2, weighted_ngf_from_fixed,
};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_image::native::Image;
use ritk_interpolation::native::trilinear_interpolation;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;

/// Innermost-first (`col 0 = x = axis D-1`) row-major index grid over `shape`,
/// as flat `[N·3]` `f32` — the same column/row convention as
/// `ritk_image::grid::generate_grid` (Burn), reproduced here because no native
/// grid generator exists yet (that op is owned upstream by `ritk-image`; until
/// it lands, this local generator keeps the native NGF path self-contained).
fn fixed_index_grid(shape: [usize; 3]) -> Vec<f32> {
    let total: usize = shape.iter().product();
    let mut grid = Vec::with_capacity(total * 3);
    let mut idx = [0usize; 3];
    for _ in 0..total {
        // Innermost dimension first: col 0 = x = idx[D-1].
        for d in (0..3).rev() {
            grid.push(idx[d] as f32);
        }
        // Increment innermost first → row-major iteration (rows match flat data).
        for d in (0..3).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    grid
}

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
        let n: usize = shape.iter().product();
        let stride = row_major_strides(&shape);

        // Fixed grid indices (innermost-first) → world (axis-major) via the native
        // batch transform (bit-faithful to the Burn `index_to_world_tensor`).
        let idx = Tensor::<f32, B>::from_slice([n, 3], &fixed_index_grid(shape));
        let fixed_world = fixed.index_to_world_native(&idx).as_slice().to_vec();

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
        let n = self.shape.iter().product::<usize>();

        // Fixed world points (axis-major) → transformed moving-space world points.
        let world_img = Image::<f32, B, 2>::from_flat(
            self.fixed_world.clone(),
            [n, 3],
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
        )
        .expect("fixed world points carry a valid [N, 3] rank-2 layout");
        let moving_world = transform
            .transform_points(&world_img)
            .expect("affine transform of [N, 3] world points")
            .data_vec();

        // Transformed world (axis-major) → moving continuous indices (innermost-
        // first: col 0 = x = axis 2).
        let moving_world_t = Tensor::<f32, B>::from_slice([n, 3], &moving_world);
        let moving_idx = moving.world_to_index_native(&moving_world_t);
        let mi = moving_idx.as_slice();

        // Trilinear grid layout `[1, 3, N, 1, 1]`, channels (z, y, x) = axes
        // (0, 1, 2). The native kernel's channel `k` samples image axis `k`, so
        // channel `k` reads the index column for axis `k` = innermost-first
        // column `2 - k`.
        let mut grid = vec![0.0f32; 3 * n];
        for p in 0..n {
            grid[p] = mi[p * 3 + 2]; // channel 0 = z (axis 0) ← col 2
            grid[n + p] = mi[p * 3 + 1]; // channel 1 = y (axis 1) ← col 1
            grid[2 * n + p] = mi[p * 3]; // channel 2 = x (axis 2) ← col 0
        }

        let [d, h, w] = moving.shape();
        let moving_flat = moving.data_vec();
        trilinear_interpolation::<f32>(&moving_flat, 1, 1, d, h, w, &grid, n, 1, 1)
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
