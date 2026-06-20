use super::scalar::*;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use ritk_image::{grid, Image};
use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_transform::Transform;

/// Precomputed fixed-image NGF state for repeated transform evaluations across a
/// registration. Building this ONCE removes from the optimiser's hot loop: the
/// fixed-grid generation, the index→world mapping, the fixed host read, the fixed
/// gradient field, and `η_F`. Each [`eval`](Self::eval) then only resamples the
/// moving image and computes its gradient — roughly halving the per-evaluation
/// gradient work and eliminating the largest repeated allocations.
pub(crate) struct NgfFixedPrep<B: Backend, const D: usize> {
    /// World coordinates of every fixed-grid voxel, `[N, D]` (constant).
    fixed_points: Tensor<B, 2>,
    shape: [usize; D],
    stride: [usize; D],
    /// Precomputed fixed gradient field and its edge-noise scale.
    gf: Vec<[f32; D]>,
    eta_f2: f32,
    mask: Option<Vec<bool>>,
    weights: Option<Vec<f32>>,
    interpolator: LinearInterpolator,
    /// When `Some`, [`eval`](Self::eval) estimates NGF on this fixed random voxel
    /// subset instead of the full grid — orders of magnitude fewer resample +
    /// gradient operations per evaluation (elastix-style stochastic sampling).
    sampling: Option<SamplePlan<B, D>>,
}

/// A fixed random subset of mask voxels for stochastic NGF evaluation. Holds the
/// world coordinates of each sample's axis-neighbours (so the moving gradient is
/// a gather + finite difference), the precomputed fixed gradient/weight/denominator
/// per sample, and the fixed edge-noise scale over the subset.
struct SamplePlan<B: Backend, const D: usize> {
    /// World coords of `[lo₀, hi₀, …, lo_{D-1}, hi_{D-1}]` per sample, row-major,
    /// shape `[2D·S, D]`. Out-of-bounds neighbours reuse the sample's own index.
    points: Tensor<B, 2>,
    /// Fixed gradient at each sample voxel.
    gf: Vec<[f32; D]>,
    /// Per-voxel weight at each sample (mask × optional center-Gaussian).
    weight: Vec<f64>,
    /// Per-axis finite-difference denominator (in-bounds neighbour count) per sample.
    denom: Vec<[f32; D]>,
    /// `η_F²` over the sample subset.
    eta_f2: f32,
    /// Number of samples `S`.
    count: usize,
}

impl<B: Backend, const D: usize> NgfFixedPrep<B, D> {
    /// Precompute the fixed-image state for `fixed`, restricted to `mask` and
    /// scaled by `weights` (both row-major, fixed-image C-order).
    pub(crate) fn new(fixed: &Image<B, D>, mask: Option<&[bool]>, weights: Option<&[f32]>) -> Self {
        let device = fixed.data().device();
        let shape = fixed.shape();
        let n: usize = shape.iter().product();
        let fixed_indices = grid::generate_grid(shape, &device);
        let fixed_points = fixed.index_to_world_tensor(fixed_indices);
        let f: Vec<f32> = fixed
            .data()
            .clone()
            .reshape([n])
            .into_data()
            .to_vec()
            .expect("fixed image to f32 host vec");
        let stride = row_major_strides(&shape);
        let gf = compute_gradient_field(&f, &shape, &stride);
        let eta_f2 = weighted_eta2(&gf, mask, weights);
        Self {
            fixed_points,
            shape,
            stride,
            gf,
            eta_f2,
            mask: mask.map(<[bool]>::to_vec),
            weights: weights.map(<[f32]>::to_vec),
            interpolator: LinearInterpolator::new(),
            sampling: None,
        }
    }

    /// As [`new`](Self::new), but [`eval`](Self::eval) estimates NGF on a fixed
    /// deterministic subset of `sample_count` mask voxels (spread by uniform
    /// stride over the masked, positive-weight voxels). The subset is fixed for
    /// reproducibility; the estimate's variance is bounded by the sample count.
    pub(crate) fn new_sampled(
        fixed: &Image<B, D>,
        mask: Option<&[bool]>,
        weights: Option<&[f32]>,
        sample_count: usize,
    ) -> Self {
        let device = fixed.data().device();
        let shape = fixed.shape();
        let n: usize = shape.iter().product();
        let stride = row_major_strides(&shape);

        // Candidate voxels: masked with positive weight.
        let candidates: Vec<usize> = (0..n)
            .filter(|&f| voxel_weight(f, mask, weights) > 0.0)
            .collect();
        if candidates.is_empty() || sample_count == 0 {
            return Self::new(fixed, mask, weights); // fall back to the dense path
        }
        let s = sample_count.min(candidates.len());
        let step = (candidates.len() / s).max(1);
        let samples: Vec<usize> = candidates.iter().copied().step_by(step).take(s).collect();

        // One host read of the fixed image — the only O(N) operation. The full
        // fixed grid and full gradient field are NOT built (the dominant cost at
        // fine pyramid levels); only the sampled gradients and the sampled
        // neighbour world coordinates are materialised.
        let f: Vec<f32> = fixed
            .data()
            .clone()
            .reshape([n])
            .into_data()
            .to_vec()
            .expect("fixed image to f32 host vec");

        let comp = |flat: usize, a: usize| (flat / stride[a]) % shape[a];
        // For each sample: the fixed gradient, weight, per-axis denominator, and
        // the 2D neighbour VOXEL multi-indices (OOB → the sample itself, matching
        // `grad_at`'s one-sided difference) laid out as `[lo₀, hi₀, …]` rows.
        let mut idx_multi: Vec<f32> = Vec::with_capacity(samples.len() * 2 * D * D);
        let mut denom: Vec<[f32; D]> = Vec::with_capacity(samples.len());
        let mut gf_s: Vec<[f32; D]> = Vec::with_capacity(samples.len());
        let mut weight: Vec<f64> = Vec::with_capacity(samples.len());
        let mut g = [0.0_f32; D];
        for &flat in &samples {
            let mut d = [0.0_f32; D];
            for (a, da) in d.iter_mut().enumerate() {
                let idx_a = comp(flat, a);
                let has_lo = idx_a > 0;
                let has_hi = idx_a + 1 < shape[a];
                let lo = if has_lo { flat - stride[a] } else { flat };
                let hi = if has_hi { flat + stride[a] } else { flat };
                // Column order matches `generate_grid`: innermost-first (col 0 = x =
                // axis D-1), the convention `index_to_world_tensor` consumes.
                for b in (0..D).rev() {
                    idx_multi.push(comp(lo, b) as f32);
                }
                for b in (0..D).rev() {
                    idx_multi.push(comp(hi, b) as f32);
                }
                *da = (usize::from(has_lo) + usize::from(has_hi)).max(1) as f32;
            }
            denom.push(d);
            grad_at(&f, flat, &shape, &stride, &mut g);
            gf_s.push(g);
            weight.push(voxel_weight(flat, mask, weights));
        }

        // Index→world for only the sampled neighbours (`[2D·S, D]`).
        let rows = samples.len() * 2 * D;
        let idx = Tensor::<B, 2>::from_data(TensorData::new(idx_multi, [rows, D]), &device);
        let points = fixed.index_to_world_tensor(idx);
        let eta_f2 = weighted_eta2_slice(&gf_s, &weight);

        Self {
            // Dense-path fields are unused when `sampling` is `Some`.
            fixed_points: Tensor::<B, 2>::zeros([1, D], &device),
            shape,
            stride,
            gf: Vec::new(),
            eta_f2: 1.0,
            mask: None,
            weights: None,
            interpolator: LinearInterpolator::new(),
            sampling: Some(SamplePlan {
                points,
                gf: gf_s,
                weight,
                denom,
                eta_f2,
                count: s,
            }),
        }
    }

    /// `NGF ∈ [0, 1]` of `moving` resampled through `transform` onto the fixed
    /// grid, reusing the precomputed fixed state. Uses the stochastic-sample
    /// estimate when a [`SamplePlan`] is present, else the dense full-grid metric.
    pub(crate) fn eval(&self, moving: &Image<B, D>, transform: &impl Transform<B, D>) -> f32 {
        match &self.sampling {
            Some(plan) => self.eval_sampled(moving, transform, plan),
            None => {
                let m = self.resample(moving, transform, self.fixed_points.clone());
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
        }
    }

    /// Resample `moving` through `transform` at the given fixed-grid world
    /// `points` (`[K, D]`), returning the `K` interpolated host values.
    fn resample(
        &self,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        points: Tensor<B, 2>,
    ) -> Vec<f32> {
        let moving_points = transform.transform_points(points);
        let moving_indices = moving.world_to_index_tensor(moving_points);
        self.interpolator
            .interpolate(moving.data(), moving_indices)
            .into_data()
            .to_vec()
            .expect("resampled moving image to f32 host vec")
    }

    /// Stochastic NGF estimate over the [`SamplePlan`] subset: resample only the
    /// `2D·S` neighbour points, finite-difference the moving gradient per sample,
    /// and accumulate the weighted squared normalized dot product against the
    /// precomputed fixed gradients. Identical formula to the dense path, evaluated
    /// on `S` voxels instead of `N`.
    fn eval_sampled(
        &self,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        plan: &SamplePlan<B, D>,
    ) -> f32 {
        let vals = self.resample(moving, transform, plan.points.clone());
        // Per-sample moving gradient from its `[lo, hi]` neighbour pairs.
        let mut gm_all: Vec<[f32; D]> = Vec::with_capacity(plan.count);
        let (mut s_eta, mut wsum) = (0.0_f64, 0.0_f64);
        for si in 0..plan.count {
            let base = si * 2 * D;
            let mut gm = [0.0_f32; D];
            for (a, gma) in gm.iter_mut().enumerate() {
                let lo = vals[base + 2 * a];
                let hi = vals[base + 2 * a + 1];
                *gma = (hi - lo) / plan.denom[si][a];
            }
            s_eta += plan.weight[si] * f64::from(magnitude(&gm));
            wsum += plan.weight[si];
            gm_all.push(gm);
        }
        if wsum <= 0.0 {
            return 0.0;
        }
        let eta_m2 = ((s_eta / wsum).powi(2)).max(1e-12) as f32;

        let mut acc = 0.0_f64;
        for (si, gm) in gm_all.iter().enumerate().take(plan.count) {
            let g = &plan.gf[si];
            let dot: f32 = (0..D).map(|a| g[a] * gm[a]).sum();
            let na2: f32 = (0..D).map(|a| g[a] * g[a]).sum::<f32>() + plan.eta_f2;
            let nb2: f32 = (0..D).map(|a| gm[a] * gm[a]).sum::<f32>() + eta_m2;
            acc += plan.weight[si] * f64::from((dot * dot) / (na2 * nb2));
        }
        (acc / wsum) as f32
    }
}
