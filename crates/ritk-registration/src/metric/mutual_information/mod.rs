//! Mutual Information metric implementations.
//!
//! Provides a single, mathematically unified implementation for Standard MI,
//! Mattes MI, and Normalized MI, strictly avoiding redundant density
//! estimations while enforcing Single Source of Truth for histogram operations.
//!
//! # Theorem: Mattes Mutual Information
//!
//! **Theorem** (Mattes et al. 2003, *IEEE Trans. Med. Imaging* 22:986):
//! Given N randomly sampled voxels xᵢ from fixed image A and moving image B∘T:
//! ```text
//! I_Mattes(A, B; T) = Σ_{a,b} p̂(a,b) · log( p̂(a,b) / (p̂_A(a) · p̂_B(b)) )
//! ```
//! where the joint density is estimated by cubic B-spline Parzen windows:
//! ```text
//! p̂(a,b) = (1/N) Σᵢ β³((a − A(xᵢ)) / hₐ) · β³((b − B(T(xᵢ))) / h_b)
//! ```
//!
//! # Cubic B-spline Kernel
//!
//! ```text
//! β³(u) = { 2/3 − |u|² + |u|³/2,     |u| < 1
//!          { (2 − |u|)³ / 6,          1 ≤ |u| < 2
//!          { 0,                        |u| ≥ 2
//! ```
//!
//! Analytic gradient w.r.t. transform parameters θ is computed automatically
//! via Burn's autodiff backend.
//!
//! # References
//!
//! - Mattes, D., et al. (2003). PET-CT image registration in the chest using
//!   free-form deformations. *IEEE Trans. Med. Imaging* 22(1):120–128.
//! - Viola, P., & Wells, W. M. (1997). Alignment by maximization of mutual
//!   information. *Int. J. Comput. Vis.* 24(2):137–154.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::interpolation::LinearInterpolator;
use ritk_core::transform::Transform;
use std::sync::{Arc, Mutex};

use crate::metric::histogram::cache::WFixedCache;
use crate::metric::{histogram::ParzenJointHistogram, Metric};

mod variant;
pub use variant::{MutualInformationVariant, NormalizationMethod};

/// Unified Mutual Information Metric.
///
/// Computes mutual information between two images using differentiable soft histogramming.
/// Consolidates Standard, Mattes, and Normalized variants into a single SSOT architecture.
///
/// MI(X, Y) = H(X) + H(Y) - H(X, Y)
///
/// Returns negative metric as loss (to be minimized).
#[derive(Clone, Debug)]
pub struct MutualInformation<B: Backend> {
    variant: MutualInformationVariant,
    histogram_calculator: ParzenJointHistogram<B>,
    sampling_percentage: Option<f32>,
    interpolator: LinearInterpolator,
    fixed_mask_points: Option<Tensor<B, 2>>,
    /// 350-P1-03: per-`MutualInformation` cache for the fixed-image Parzen weight
    /// matrix `W_fixed^T [num_bins, N]`. Reused across iterations of the same
    /// multi-resolution level when the (fixed image, n) pair is unchanged.
    ///
    /// On the first iteration of a level, [`Metric::forward`] lazily populates
    /// this via [`ParzenJointHistogram::compute_image_joint_histogram`] +
    /// `extract_w_fixed_t_cache`. Subsequent [`Metric::forward_with_cache`]
    /// calls hit the cache and skip the O(N × num_bins) Parzen weight
    /// recomputation. Shared across clones via `Arc`.
    cached_w_fixed_t: Arc<Mutex<Option<WFixedCache<B>>>>,
}

impl<B: Backend> MutualInformation<B> {
    /// Create a new Mutual Information metric with explicit parameters.
    ///
    /// # Arguments
    /// * `variant` - Variant of MI (Standard, Mattes, Normalized)
    /// * `num_bins` - Number of histogram bins
    /// * `min_intensity` - Minimum expected intensity
    /// * `max_intensity` - Maximum expected intensity
    /// * `parzen_sigma` - Parzen window sigma (for Mattes, this should be `bin_width`)
    pub fn new(
        variant: MutualInformationVariant,
        num_bins: usize,
        min_intensity: f32,
        max_intensity: f32,
        parzen_sigma: f32,
        device: &B::Device,
    ) -> Self {
        assert!(num_bins >= 4, "num_bins must be ≥ 4");
        Self {
            variant,
            histogram_calculator: ParzenJointHistogram::new(
                num_bins,
                min_intensity,
                max_intensity,
                parzen_sigma,
                device,
            ),
            sampling_percentage: None,
            interpolator: LinearInterpolator::new_zero_pad(),
            fixed_mask_points: None,
            cached_w_fixed_t: Arc::new(Mutex::new(None)),
        }
    }

    /// Create with separate intensity ranges for fixed and moving images (elastix-style).
    ///
    /// Each image axis of the joint histogram uses its own min/max/sigma,
    /// giving each image the full histogram resolution. This is particularly
    /// beneficial for cross-modal registration (CT/MRI) where the two images
    /// have very different intensity ranges: with a shared range, the narrower
    /// image wastes a fraction of its bins; with separate ranges, both images
    /// use all `num_bins` bins.
    pub fn new_with_separate_ranges(
        variant: MutualInformationVariant,
        num_bins: usize,
        fixed_min: f32,
        fixed_max: f32,
        moving_min: f32,
        moving_max: f32,
        device: &B::Device,
    ) -> Self {
        let mut s = Self::new(
            variant,
            num_bins,
            fixed_min,
            fixed_max,
            (fixed_max - fixed_min).max(1e-6) / num_bins as f32,
            device,
        );
        s.histogram_calculator = s
            .histogram_calculator
            .with_separate_moving_range(moving_min, moving_max);
        s
    }

    /// Helper to create Mattes Mutual Information with its characteristic parameterization.
    /// Mattes specifies `parzen_sigma = bin_width`.
    pub fn new_mattes(
        num_bins: usize,
        min_intensity: f32,
        max_intensity: f32,
        device: &B::Device,
    ) -> Self {
        let bin_width = (max_intensity - min_intensity).max(1e-6) / num_bins as f32;
        Self::new(
            MutualInformationVariant::Mattes,
            num_bins,
            min_intensity,
            max_intensity,
            bin_width,
            device,
        )
    }

    /// Create with default Mattes parameters (50 bins, [0, 255]).
    pub fn mattes_default(device: &B::Device) -> Self {
        Self::new_mattes(50, 0.0, 255.0, device).with_sampling(0.20)
    }

    /// Create with default Standard parameters (32 bins, [0, 255]).
    pub fn standard_default(device: &B::Device) -> Self {
        Self::new(
            MutualInformationVariant::Standard,
            32,
            0.0,
            255.0,
            1.0,
            device,
        )
    }

    /// Create with default Normalized parameters (JointEntropy, 32 bins).
    pub fn normalized_default(device: &B::Device) -> Self {
        Self::new(
            MutualInformationVariant::Normalized(NormalizationMethod::JointEntropy),
            32,
            0.0,
            255.0,
            1.0,
            device,
        )
    }

    /// Supply pre-selected foreground world-space sample points for the fixed image.
    ///
    /// When set, these replace stochastic uniform sampling in [`Metric::forward`]:
    /// only the provided world positions contribute to the joint histogram.
    /// This restricts MI computation to the brain region (standard ANTs strategy),
    /// eliminating background-dominated histogram bins that cause spurious MI peaks.
    ///
    /// The points tensor must have shape `[N, D]` and be on the same device as the
    /// images passed to `forward`.  It should be generated from the same pyramid
    /// level as the images — call `extract_foreground_world_points` in the CMA-ES
    /// registration pipeline to produce these from a downsampled brain mask.
    ///
    /// When `None` (the default), the existing stochastic uniform sampling path is
    /// used (controlled by `with_sampling`).
    pub fn with_fixed_mask_points(mut self, points: Tensor<B, 2>) -> Self {
        self.fixed_mask_points = Some(points);
        self
    }

    /// Set stochastic sampling fraction ∈ (0, 1].
    pub fn with_sampling(mut self, percentage: f32) -> Self {
        let clamped = percentage.clamp(1e-4, 1.0);
        if clamped < 1.0 {
            self.sampling_percentage = Some(clamped);
        } else {
            self.sampling_percentage = None;
        }
        self
    }

    #[inline]
    pub fn num_bins(&self) -> usize {
        self.histogram_calculator.num_bins
    }

    /// Convert a joint histogram `[num_bins, num_bins]` into the MI / NMI loss
    /// tensor, applying the configured variant routing (Standard / Mattes /
    /// Normalized).
    ///
    /// 350-P1-03: factored out of [`Metric::forward`] so the same loss
    /// computation is reused by both `forward` (cache-miss path) and
    /// `forward_with_cache` (cache-hit fast path). The histogram itself
    /// carries no autodiff state into this method — the autodiff path is
    /// already established by either `compute_image_joint_histogram` or
    /// `compute_image_joint_histogram_with_w_fixed` at the call site.
    ///
    /// 350-P1-04 attempt: tried to drop the `.clone()` on `joint_hist` and
    /// `p_xy` (the audit §2.4 hot-path allocations), but Burn's `Tensor::sum`
    /// and `Tensor::sum_dim` consume `self` (move semantics, not `&self`),
    /// so the original `.clone()` calls are required to reuse the tensors
    /// downstream. The clones stay; the audit §2.4 entry is left as a known
    /// minor allocation. P1-04 closed as **N/A** (no code change).
    fn compute_mi_loss(&self, joint_hist: Tensor<B, 2>) -> Tensor<B, 1> {
        // Normalize to joint PDF p̂(x,y)
        // The `.clone()` on `joint_hist` is required: Burn's `Tensor::sum`
        // consumes `self`, so we must clone before `.sum()` to keep
        // `joint_hist` usable for the `/` below.
        let sum = joint_hist.clone().sum();
        let eps = 1e-10_f32;
        let p_xy = joint_hist / (sum.unsqueeze_dim(1) + eps);

        // Compute marginals p̂(x), p̂(y) — also requires `.clone()` for the
        // same reason: `Tensor::sum_dim(dim)` consumes `self`, so we clone
        // `p_xy` to extract both marginals from the same joint PDF.
        let p_x = p_xy.clone().sum_dim(1).squeeze::<1>();
        let p_y = p_xy.clone().sum_dim(0).squeeze::<1>();

        // Entropy computations
        // H(X) = -Σ p(x) log p(x)
        // Sprint 354 ARCH-354-02: delegate to the shared free function so the
        // entropy formula lives in one place (metric/entropy.rs) instead of
        // being a method on ParzenJointHistogram.
        let h_x = crate::metric::entropy::entropy(p_x);
        let h_y = crate::metric::entropy::entropy(p_y);

        // H(X,Y) = -Σ p(x,y) log p(x,y)
        let log_p_xy = (p_xy.clone() + eps).log();
        let h_xy = p_xy.mul(log_p_xy).sum().neg();

        // Variant routing
        match self.variant {
            MutualInformationVariant::Standard | MutualInformationVariant::Mattes => {
                // MI(X,Y) = H(X) + H(Y) - H(X,Y)
                let mi = h_x + h_y - h_xy;
                // Return negative MI (loss)
                mi.neg()
            }
            MutualInformationVariant::Normalized(method) => {
                let nmi = match method {
                    NormalizationMethod::JointEntropy => (h_x.clone() + h_y.clone()) / (h_xy + eps),
                    NormalizationMethod::AverageEntropy => {
                        let mi = h_x.clone() + h_y.clone() - h_xy;
                        (mi * 2.0) / (h_x + h_y + eps)
                    }
                    NormalizationMethod::MinEntropy => {
                        let mi = h_x.clone() + h_y.clone() - h_xy;
                        let sum_h = h_x.clone() + h_y.clone();
                        let diff_h = (h_x - h_y).abs();
                        let min_h = (sum_h - diff_h) * 0.5;
                        mi / (min_h + eps)
                    }
                    NormalizationMethod::MaxEntropy => {
                        let mi = h_x.clone() + h_y.clone() - h_xy;
                        let sum_h = h_x.clone() + h_y.clone();
                        let diff_h = (h_x - h_y).abs();
                        let max_h = (sum_h + diff_h) * 0.5;
                        mi / (max_h + eps)
                    }
                };
                // Return negative NMI (loss)
                nmi.neg()
            }
        }
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for MutualInformation<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Joint Histogram built strictly through shared Parzen infrastructure
        let joint_hist = if let Some(ref pts) = self.fixed_mask_points {
            self.histogram_calculator.compute_masked_joint_histogram(
                fixed,
                pts.clone(),
                moving,
                transform,
                &self.interpolator,
                None, // No caching — mask points are constant across iterations;
                      // callers (e.g., CMA-ES optimizer) may pass Some(key)
                      // directly to enable caching.
            )
        } else {
            self.histogram_calculator.compute_image_joint_histogram(
                fixed,
                moving,
                transform,
                &self.interpolator,
                self.sampling_percentage,
            )
        };

        let result = self.compute_mi_loss(joint_hist);

        // 2. 350-P1-03: Populate the per-instance W_fixed^T cache (one-time per
        //    level) so subsequent `forward_with_cache` calls can skip the
        //    O(N × num_bins) Parzen weight recomputation.
        //
        //    Only the non-mask, non-sampling path benefits from W_fixed^T reuse
        //    (the sampling and mask paths use different point sets per call,
        //    so the matrix is not actually constant). For these paths, the
        //    cache stays empty and `forward_with_cache` falls back to `forward`.
        if self.fixed_mask_points.is_none() && self.sampling_percentage.is_none() {
            let n = fixed.shape().iter().product::<usize>();
            let already_populated = {
                let cache_guard = self
                    .cached_w_fixed_t
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                cache_guard
                    .as_ref()
                    .map(|c| c.matches(fixed, n))
                    .unwrap_or(false)
            };
            if !already_populated {
                if let Some(new_cache) = self.histogram_calculator.extract_w_fixed_t_cache(fixed, n)
                {
                    *self
                        .cached_w_fixed_t
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(new_cache);
                }
            }
        }

        result
    }

    /// 350-P1-03: cache-aware fast path. Consults the per-instance W_fixed^T
    /// cache first; on a (fingerprint, n) hit, uses
    /// `compute_image_joint_histogram_with_w_fixed` to skip the
    /// O(N × num_bins) Parzen weight recomputation. On miss, falls back to
    /// `forward` (which populates the cache as a side effect for the next call).
    ///
    /// Only the non-mask, non-sampling path supports cache reuse — for the
    /// other paths this method is a thin pass-through to `forward`.
    ///
    /// Expected savings on a 256³ Mattes MI volume: ~3.2 GB / ~400 ms per
    /// level per `docs/audit_optimization_sprint_350.md` §2.3.
    fn forward_with_cache(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // Only the non-mask, non-sampling path supports W_fixed^T cache reuse.
        if self.fixed_mask_points.is_some() || self.sampling_percentage.is_some() {
            return self.forward(fixed, moving, transform);
        }

        let n = fixed.shape().iter().product::<usize>();

        // Check the per-instance cache.
        let cached_w_fixed_t = {
            let cache_guard = self
                .cached_w_fixed_t
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache_guard
                .as_ref()
                .filter(|c| c.matches(fixed, n))
                .map(|c| c.w_fixed_t.clone())
        };

        if let Some(w_fixed_t) = cached_w_fixed_t {
            // Cache hit — use the new fast path: skip the Parzen weight
            // recomputation, reuse the precomputed W_fixed^T matrix.
            let joint_hist = self
                .histogram_calculator
                .compute_image_joint_histogram_with_w_fixed(
                    fixed,
                    moving,
                    transform,
                    &self.interpolator,
                    &w_fixed_t,
                    n,
                );
            self.compute_mi_loss(joint_hist)
        } else {
            // Cache miss — fall through to forward(), which populates the cache
            // as a side effect for the next iteration.
            self.forward(fixed, moving, transform)
        }
    }

    fn name(&self) -> &'static str {
        match self.variant {
            MutualInformationVariant::Standard => "Mutual Information",
            MutualInformationVariant::Mattes => "Mattes Mutual Information",
            MutualInformationVariant::Normalized(_) => "Normalized Mutual Information",
        }
    }
}

#[cfg(test)]
mod tests_mutual_information;
