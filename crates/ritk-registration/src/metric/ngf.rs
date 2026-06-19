//! Normalized Gradient Fields (NGF) metric for multi-modal registration.
//!
//! # Theorem: edge-orientation similarity (Haber & Modersitzki 2006)
//!
//! Cross-modal pairs (CT↔MRI) lack a functional intensity relationship, so
//! intensity metrics (MI, NCC) can be weak — e.g. a near-uniform CT brain
//! interior gives almost no mutual-information signal, and a rotation about the
//! centroid barely perturbs the joint histogram. NGF instead aligns the
//! **orientation** of image gradients, which co-locate across modalities even
//! where intensities do not (a skull/ventricle boundary is an edge in *both* CT
//! and MRI). For fixed `F` and moving `M` resampled onto the fixed grid,
//!
//! ```text
//! NGF(F, M) = (1/N) · Σ_x  (∇F·∇M)² / ((|∇F|² + η_F²)(|∇M|² + η_M²))
//! ```
//!
//! Each term is `1` when the gradients are parallel **or anti-parallel** (so a
//! bright-CT / dark-MR edge still scores `1` — the squared dot product is
//! sign-invariant) and `0` where either side is flat. `η` is the edge-noise
//! scale (the mean masked gradient magnitude, per Haber & Modersitzki), which
//! suppresses flat-region noise. `NGF ∈ [0, 1]`; higher is better aligned, so the
//! metric returns `−NGF` as a minimization loss.
//!
//! This is a **gradient-free** metric (the gradients are spatial image gradients,
//! not autodiff gradients of the transform): it returns a scalar for the
//! derivative-free optimizers (CMA-ES, coordinate descent) that cross-modal rigid
//! registration uses, where intensity-MI hill-climbing from identity is unreliable.
//! Pre-masking the images (e.g. to a brain mask) focuses NGF on the shared rigid
//! structure, since flat masked-out regions contribute ~0.

use super::trait_::Metric;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use ritk_image::grid;
use ritk_image::Image;
use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_transform::Transform;

/// Normalized Gradient Fields metric (Haber & Modersitzki 2006).
///
/// Returns `−NGF ∈ [−1, 0]` as a loss to be minimized. Robust for cross-modal
/// (CT↔MRI) alignment where intensity MI/NCC are weak. See the [module docs](self).
pub struct NormalizedGradientField;

impl NormalizedGradientField {
    /// Create a new NGF metric (linear interpolation of the moving image, held by
    /// the per-registration `NgfFixedPrep`).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for NormalizedGradientField {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for NormalizedGradientField {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        let device = fixed.data().device();
        let ngf = self.ngf_value(fixed, moving, transform, None);
        // −NGF as a minimization loss.
        Tensor::<B, 1>::from_data(TensorData::new(vec![-ngf], [1]), &device)
    }

    fn name(&self) -> &'static str {
        "NormalizedGradientField"
    }
}

impl NormalizedGradientField {
    /// Resample `moving` onto the `fixed` grid through `transform`, then return
    /// `NGF ∈ [0, 1]` over the `true` voxels of `mask` (or all if `None`). The
    /// masked path is used by the cross-modal rigid registration; the unmasked
    /// path backs [`Metric::forward`].
    pub(crate) fn ngf_value<B: Backend, const D: usize>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        mask: Option<&[bool]>,
    ) -> f32 {
        self.ngf_value_weighted(fixed, moving, transform, mask, None)
    }

    /// As [`ngf_value`](Self::ngf_value), but each masked voxel's contribution is
    /// scaled by `weights[flat]` (row-major, same length as the fixed image).
    /// A brain-centroid Gaussian weight (see [`center_gaussian_weight_field`])
    /// down-weights the high-gradient skull/scalp rim — which otherwise dominates
    /// the uniform NGF average and lets the optimiser ignore deep structures
    /// (ventricles, deep gray) — so the metric becomes sensitive to the central
    /// anatomy where rigid alignment is anatomically defined.
    pub(crate) fn ngf_value_weighted<B: Backend, const D: usize>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        mask: Option<&[bool]>,
        weights: Option<&[f32]>,
    ) -> f32 {
        // One-shot path (trait `forward`): build the fixed context and evaluate
        // once. The registration hot loop instead builds [`NgfFixedPrep`] ONCE and
        // calls [`NgfFixedPrep::eval`] per transform, reusing the fixed work.
        NgfFixedPrep::new(fixed, mask, weights).eval(moving, transform)
    }
}

/// `NGF ∈ [0, 1]` of two co-gridded volumes `f`, `m` of (row-major) `shape`,
/// averaged over the `true` voxels of `mask` (or all voxels if `mask` is `None`).
///
/// Gradients always use real neighbours (the image is *not* zeroed outside the
/// mask), so the mask only restricts which voxels are *counted* — no artificial
/// mask-boundary edge is introduced. Masking to the brain+skull region is what
/// makes cross-modal NGF lock onto the shared rigid anatomy instead of the
/// scalp/scanner-bed/FOV edges. See the [module docs](self).
#[cfg(test)]
fn ngf_scalar<const D: usize>(
    f: &[f32],
    m: &[f32],
    shape: &[usize; D],
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> f32 {
    let n = f.len();
    if n == 0 || m.len() != n {
        return 0.0;
    }
    let stride = row_major_strides(shape);
    let gf = compute_gradient_field(f, shape, &stride);
    let eta_f2 = weighted_eta2(&gf, mask, weights);
    weighted_ngf_from_fixed(&gf, eta_f2, m, shape, &stride, mask, weights)
}

/// Row-major (C-order) strides for `shape` — last axis fastest.
fn row_major_strides<const D: usize>(shape: &[usize; D]) -> [usize; D] {
    let mut stride = [1usize; D];
    for a in (0..D.saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * shape[a + 1];
    }
    stride
}

/// Per-voxel central-difference gradient field of `data` (one `[f32; D]` per
/// voxel). Computed once per image and, for the fixed image, reused across every
/// transform evaluation.
fn compute_gradient_field<const D: usize>(
    data: &[f32],
    shape: &[usize; D],
    stride: &[usize; D],
) -> Vec<[f32; D]> {
    let mut g = vec![[0.0_f32; D]; data.len()];
    for (flat, gv) in g.iter_mut().enumerate() {
        grad_at(data, flat, shape, stride, gv);
    }
    g
}

/// Per-voxel weight: 0 outside the mask, else `weights[flat]` (or 1 when no
/// weight field is supplied — recovering the uniform Haber–Modersitzki mean).
fn voxel_weight(flat: usize, mask: Option<&[bool]>, weights: Option<&[f32]>) -> f64 {
    if mask.is_some_and(|mk| !mk[flat]) {
        return 0.0;
    }
    weights.map_or(1.0, |w| f64::from(w[flat]).max(0.0))
}

/// `η² = (weighted-mean masked gradient magnitude)²` — the edge-noise scale
/// (Haber & Modersitzki), clamped away from zero.
fn weighted_eta2<const D: usize>(
    grads: &[[f32; D]],
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> f32 {
    let (mut s, mut ws) = (0.0_f64, 0.0_f64);
    for (flat, g) in grads.iter().enumerate() {
        let w = voxel_weight(flat, mask, weights);
        if w == 0.0 {
            continue;
        }
        s += w * f64::from(magnitude(g));
        ws += w;
    }
    if ws <= 0.0 {
        1e-12
    } else {
        ((s / ws).powi(2)).max(1e-12) as f32
    }
}

/// `η²` over an explicit gradient/weight subset (the stochastic-sampling path).
fn weighted_eta2_slice<const D: usize>(grads: &[[f32; D]], weight: &[f64]) -> f32 {
    let (mut s, mut ws) = (0.0_f64, 0.0_f64);
    for (g, &w) in grads.iter().zip(weight) {
        s += w * f64::from(magnitude(g));
        ws += w;
    }
    if ws <= 0.0 {
        1e-12
    } else {
        ((s / ws).powi(2)).max(1e-12) as f32
    }
}

/// Weighted mean squared normalized gradient dot product, given the PRECOMPUTED
/// fixed gradient field `gf`/`eta_f2` and the moving host volume `m`. The moving
/// gradient and `η_M` are computed here (they vary with the transform); the fixed
/// side is supplied so a registration reuses it across every evaluation.
fn weighted_ngf_from_fixed<const D: usize>(
    gf: &[[f32; D]],
    eta_f2: f32,
    m: &[f32],
    shape: &[usize; D],
    stride: &[usize; D],
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> f32 {
    let n = gf.len();
    if m.len() != n {
        return 0.0;
    }
    let mut gm = [0.0_f32; D];

    // Pass 1: η_M² from the (weighted) moving gradient magnitude.
    let (mut s, mut ws) = (0.0_f64, 0.0_f64);
    for flat in 0..n {
        let w = voxel_weight(flat, mask, weights);
        if w == 0.0 {
            continue;
        }
        grad_at(m, flat, shape, stride, &mut gm);
        s += w * f64::from(magnitude(&gm));
        ws += w;
    }
    if ws <= 0.0 {
        return 0.0;
    }
    let eta_m2 = ((s / ws).powi(2)).max(1e-12) as f32;

    // Pass 2: weighted squared normalized gradient dot product over the mask.
    let mut acc = 0.0_f64;
    for (flat, g) in gf.iter().enumerate() {
        let w = voxel_weight(flat, mask, weights);
        if w == 0.0 {
            continue;
        }
        grad_at(m, flat, shape, stride, &mut gm);
        let dot: f32 = (0..D).map(|a| g[a] * gm[a]).sum();
        let na2: f32 = (0..D).map(|a| g[a] * g[a]).sum::<f32>() + eta_f2;
        let nb2: f32 = (0..D).map(|a| gm[a] * gm[a]).sum::<f32>() + eta_m2;
        acc += w * f64::from((dot * dot) / (na2 * nb2));
    }
    (acc / ws) as f32
}

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

/// Build a brain-centroid Gaussian weight field over `mask` for the weighted NGF
/// metric. `w(x) = exp(−‖x − c‖² / (2σ²))` for masked voxels (0 elsewhere), with
/// `c` the mask centroid and `σ = sigma_frac · r_rms`, where `r_rms` is the
/// mask's root-mean-square radius — all distances in PHYSICAL units via `spacing`
/// so anisotropic voxels (e.g. 0.4×0.4×3 mm) are weighted correctly.
///
/// `sigma_frac` controls how sharply the periphery is suppressed: smaller →
/// tighter central focus. With the RMS radius as the scale, `sigma_frac ≈ 0.7`
/// places σ near ⅓ of the outer brain radius (the value the multimodal-edge
/// literature uses to emphasise deep structure over the skull rim).
#[must_use]
pub fn center_gaussian_weight_field<const D: usize>(
    shape: &[usize; D],
    mask: Option<&[bool]>,
    spacing: &[f64; D],
    sigma_frac: f64,
) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut stride = [1usize; D];
    for a in (0..D.saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * shape[a + 1];
    }
    let included = |flat: usize| mask.is_none_or(|mk| mk[flat]);
    let phys = |flat: usize, a: usize| ((flat / stride[a]) % shape[a]) as f64 * spacing[a];

    // Centroid (physical) and RMS radius over the mask.
    let mut c = [0.0_f64; D];
    let mut cnt = 0.0_f64;
    for flat in 0..n {
        if !included(flat) {
            continue;
        }
        for (a, ca) in c.iter_mut().enumerate() {
            *ca += phys(flat, a);
        }
        cnt += 1.0;
    }
    if cnt < 1.0 {
        return vec![1.0; n];
    }
    for ca in &mut c {
        *ca /= cnt;
    }
    let mut r2sum = 0.0_f64;
    for flat in 0..n {
        if !included(flat) {
            continue;
        }
        r2sum += (0..D).map(|a| (phys(flat, a) - c[a]).powi(2)).sum::<f64>();
    }
    let r_rms = (r2sum / cnt).sqrt().max(f64::EPSILON);
    let two_sigma2 = 2.0 * (sigma_frac * r_rms).powi(2);

    let mut w = vec![0.0_f32; n];
    for (flat, wf) in w.iter_mut().enumerate() {
        if !included(flat) {
            continue;
        }
        let r2 = (0..D).map(|a| (phys(flat, a) - c[a]).powi(2)).sum::<f64>();
        *wf = (-r2 / two_sigma2).exp() as f32;
    }
    w
}

/// Central-difference spatial gradient of `data` at flat index `flat`, written
/// into `out` (one component per axis). Borders use a one-sided difference.
fn grad_at<const D: usize>(
    data: &[f32],
    flat: usize,
    shape: &[usize; D],
    stride: &[usize; D],
    out: &mut [f32; D],
) {
    for a in 0..D {
        let idx_a = (flat / stride[a]) % shape[a];
        let has_lo = idx_a > 0;
        let has_hi = idx_a + 1 < shape[a];
        let lo = if has_lo {
            data[flat - stride[a]]
        } else {
            data[flat]
        };
        let hi = if has_hi {
            data[flat + stride[a]]
        } else {
            data[flat]
        };
        let denom = (usize::from(has_lo) + usize::from(has_hi)).max(1) as f32;
        out[a] = (hi - lo) / denom;
    }
}

fn magnitude<const D: usize>(g: &[f32; D]) -> f32 {
    (0..D).map(|a| g[a] * g[a]).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, TensorData};
    use burn_ndarray::NdArray;
    use ritk_spatial::{Direction, Point, Spacing};
    use ritk_transform::TranslationTransform;

    type B = NdArray<f32>;

    fn image2d(data: Vec<f32>, shape: [usize; 2]) -> Image<B, 2> {
        let device = Default::default();
        let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0]),
            Spacing::new([1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn vertical_edge(w: usize, h: usize, at: usize, sign: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = if x < at { 0.0 } else { sign };
            }
        }
        v
    }

    /// Cross-modal sign invariance: a co-located edge with OPPOSITE contrast
    /// (bright→dark vs dark→bright) scores exactly the same as an identical-
    /// contrast edge — the squared gradient dot product makes a bright-CT /
    /// dark-MR boundary register just like a same-sign one.
    #[test]
    fn ngf_is_sign_invariant() {
        let (w, h) = (8usize, 8usize);
        let f = vertical_edge(w, h, 4, 1.0);
        let same = ngf_scalar(&f, &f, &[h, w], None, None);
        let opposite = ngf_scalar(&f, &vertical_edge(w, h, 4, -1.0), &[h, w], None, None);
        assert!(same > 0.0, "self-NGF should be positive, got {same}");
        assert!(
            (same - opposite).abs() < 1e-4,
            "opposite contrast must score equal: same {same} vs opposite {opposite}"
        );
    }

    /// NGF of perpendicular edges (uncorrelated orientation) is well below that of
    /// aligned edges — the property that lets NGF recover a rotation that intensity
    /// MI cannot.
    #[test]
    fn aligned_beats_perpendicular() {
        let (w, h) = (8usize, 8usize);
        let vert = vertical_edge(w, h, 4, 1.0); // gradient in x
        let mut horiz = vec![0.0f32; w * h]; // gradient in y
        for y in 0..h {
            for x in 0..w {
                horiz[y * w + x] = if y < 4 { 0.0 } else { 1.0 };
            }
        }
        let aligned = ngf_scalar(&vert, &vert, &[h, w], None, None);
        let perpendicular = ngf_scalar(&vert, &horiz, &[h, w], None, None);
        assert!(
            aligned > perpendicular + 0.1,
            "aligned {aligned} should exceed perpendicular {perpendicular}"
        );
    }

    /// Center weighting makes a CENTRAL edge mismatch dominate the metric over a
    /// stronger PERIPHERAL one — the skull-domination fix. Fixed and moving agree
    /// at the periphery but disagree centrally; the uniform NGF barely drops
    /// (periphery dominates), while the center-Gaussian-weighted NGF drops sharply
    /// because the central disagreement now carries the weight.
    #[test]
    fn center_weight_emphasizes_central_mismatch() {
        let (w, h) = (32usize, 32usize);
        // Fixed: strong peripheral vertical edges (cols 2 and 29) + a central one (col 16).
        let mut f = vec![0.0f32; w * h];
        let mut m = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let periph = if x == 2 || x == 29 { 1.0 } else { 0.0 };
                f[y * w + x] = periph + if x == 16 { 1.0 } else { 0.0 };
                // Moving matches the periphery but the central edge is displaced
                // (col 20 instead of 16) — a purely central disagreement.
                m[y * w + x] = periph + if x == 20 { 1.0 } else { 0.0 };
            }
        }
        let shape = [h, w];
        let spacing = [1.0_f64, 1.0_f64];
        let uniform = ngf_scalar(&f, &m, &shape, None, None);
        let wfield = center_gaussian_weight_field(&shape, None, &spacing, 0.4);
        let weighted = ngf_scalar(&f, &m, &shape, None, Some(&wfield));
        // The central mismatch costs MORE under center weighting: weighted NGF
        // (similarity) is strictly lower than the periphery-dominated uniform NGF.
        assert!(
            weighted < uniform - 1e-3,
            "center weighting should penalise the central mismatch: \
             weighted {weighted} vs uniform {uniform}"
        );
        // Weight field is a valid Gaussian: positive at the centre, ~0 at corners.
        let center = wfield[(h / 2) * w + w / 2];
        assert!(center > 0.9, "center weight {center} should be near 1");
        assert!(wfield[0] < center, "corner weight {} < center", wfield[0]);
    }

    /// End-to-end through the `Metric` trait: registering the moving edge onto the
    /// fixed edge (identity) gives a lower loss than a translation that pulls the
    /// edges apart. The edge varies along the d1 (x/column) axis; with an identity
    /// direction the corresponding WORLD component is index 1, so the displacing
    /// translation is `[0, dx]` (world component 1 = x).
    #[test]
    fn metric_loss_lower_when_aligned() {
        let (w, h) = (16usize, 16usize);
        let img = image2d(vertical_edge(w, h, 8, 1.0), [h, w]);
        let metric = NormalizedGradientField::new();
        let device = Default::default();
        let loss = |dx: f32| {
            let t = TranslationTransform::<B, 2>::new(Tensor::from_data(
                TensorData::new(vec![0.0_f32, dx], [2]),
                &device,
            ));
            metric
                .forward(&img, &img, &t)
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0]
        };
        let aligned = loss(0.0);
        let shifted = loss(4.0);
        assert!(
            aligned < 0.0,
            "aligned loss should be negative, got {aligned}"
        );
        assert!(
            aligned < shifted,
            "aligned loss {aligned} should be below shifted {shifted}"
        );
    }

    /// The stochastic-sample NGF path computes the SAME value as the dense path
    /// when the subset is complete (every voxel sampled) — verifying the
    /// neighbour-gather + finite-difference gradient and η over the sample subset
    /// reproduce the dense metric. A strided half-subset stays a close estimate.
    #[test]
    fn sampled_ngf_matches_dense() {
        let (w, h) = (24usize, 24usize);
        let img = image2d(vertical_edge(w, h, 12, 1.0), [h, w]);
        let device = Default::default();
        let ident = TranslationTransform::<B, 2>::new(Tensor::from_data(
            TensorData::new(vec![0.0_f32, 0.0], [2]),
            &device,
        ));
        let dense = NgfFixedPrep::<B, 2>::new(&img, None, None).eval(&img, &ident);
        let full = NgfFixedPrep::<B, 2>::new_sampled(&img, None, None, w * h).eval(&img, &ident);
        let half =
            NgfFixedPrep::<B, 2>::new_sampled(&img, None, None, w * h / 2).eval(&img, &ident);
        assert!(
            dense > 0.0,
            "dense self-NGF should be positive, got {dense}"
        );
        assert!(
            (dense - full).abs() < 1e-3,
            "full-subset sampled {full} must equal dense {dense}"
        );
        assert!(
            (dense - half).abs() < 0.15 * dense,
            "half-subset sampled {half} should approximate dense {dense}"
        );
    }
}
