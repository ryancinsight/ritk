//! Backend-dispatched joint histogram computation.
//!
//! When the `direct-parzen` feature is enabled, the **first** dispatch method
//! (`compute_joint_histogram_dispatch`) extracts tensor data to host memory and
//! calls the direct sparse-loop path (~6Ã— faster on CPU). The **cached** dispatch
//! methods provide two paths:
//!
//! - `compute_joint_histogram_from_cache_dispatch` â€” always uses the tensor matmul
//!   path to preserve the autodiff gradient tape (needed by RSGD).
//! - `compute_joint_histogram_from_cache_sparse_dispatch` â€” uses the sparse
//!   W_fixed^T representation for derivative-free backends (CMA-ES), eliminating
//!   the `0..num_bins` inner scan and the `if w_f > 0.0` branch (~3Ã— faster than
//!   the dense cache path on CPU).
//!
//! # Design
//!
//! Burn's trait system doesn't support runtime backend-type checks. Instead,
//! we use compile-time feature flags. When `direct-parzen` is enabled, the
//! non-cached path extracts tensor data to host and calls the direct loop.
//! This is safe for derivative-free backends (CMA-ES uses `B::InnerBackend`).
//!
//! The `from_cache` tensor path must preserve the autodiff tape because RSGD
//! passes `moving_values` that carry transform-parameter gradients. Calling
//! `into_data()` there would sever the tape and produce zero gradients.
//! The W_fixed^T cache still provides the key performance benefit (fixed-image
//! Parzen weights computed only once per RSGD run).
//!
//! The sparse dispatch path is called only from the chunked compute_image path
//! when a sparse W_fixed^T cache is available. It extracts moving values to
//! host memory (safe for CMA-ES which uses `B::InnerBackend`) and calls the
//! sparse inner loop.
//!
//! # SSOT
//!
//! All sigmaÂ² conversions now go through `ParzenConfig::from_intensity_sigma`
//! (SSOT-319-02). The former `sigma_sq_in_bins` standalone function has been
//! removed â€” its 10+ call sites across `dispatch.rs`, `compute.rs`,
//! `compute_image/mod.rs`, `masked/mod.rs`, and test files now call
//! `ParzenConfig::from_intensity_sigma` directly.

use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use std::borrow::Cow;

use super::ParzenJointHistogram;

/// Normalize intensities to `[0, num_bins - 1]` and extract as a `Cow<'_, [f32]>`.
///
/// This is the data-extraction bridge: it applies the same normalization as
/// the tensor path (`val * scale + offset`, clamped) but returns a host-side
/// slice suitable for the direct computation functions. Since normalization
/// always creates new data, this always returns `Cow::Owned`. The API contract
/// is established so a future Burn release exposing `Tensor::as_slice()` can
/// return `Cow::Borrowed` on the no-normalization path.
#[cfg(feature = "direct-parzen")]
pub(in crate::metric::histogram) fn normalize_and_extract<B: Backend>(
    values: &Tensor<f32, B>,
    min_intensity: f32,
    max_intensity: f32,
    num_bins: usize,
) -> Cow<'static, [f32]> {
    let num_bins_f = (num_bins - 1) as f32;
    let scale = num_bins_f / (max_intensity - min_intensity);
    let offset = -min_intensity * scale;
    let normalized = (values.clone() * scale + offset).clamp(0.0, num_bins_f);
    let data = normalized.into_data();
    Cow::Owned(data.as_slice::<f32>().expect("f32 data").to_vec())
}

/// Extract an optional OOB mask tensor to a host-side slice (DRY-326-03).
///
/// Both `compute_joint_histogram_dispatch` and
/// `compute_joint_histogram_from_cache_sparse_dispatch` shared the same
/// 5-line extraction pattern. This helper is the SSOT for converting
/// `Option<&Tensor<f32, B>>` â†’ `Option<Cow<'_, [f32]>>` for the direct path.
/// Always returns `Cow::Owned` today; can switch to `Cow::Borrowed` when
/// Burn exposes stable slice accessors.
#[cfg(feature = "direct-parzen")]
#[inline]
fn extract_oob_mask<B: Backend>(oob_mask: Option<&Tensor<f32, B>>) -> Option<Cow<'static, [f32]>> {
    oob_mask.map(|m| {
        Cow::Owned(
            m.clone()
                .into_data()
                .as_slice::<f32>()
                .expect("f32 data")
                .to_vec(),
        )
    })
}

impl<B: Backend> ParzenJointHistogram<B> {
    /// Compute joint histogram with backend-dispatched optimization.
    ///
    /// When `direct-parzen` is enabled, this attempts to use the direct
    /// sparse-loop computation path for non-autodiff backends. The method
    /// signature is backend-generic so it can be called from the same
    /// `ParzenJointHistogram<B>` regardless of backend.
    ///
    /// For autodiff backends, the gradient tape must be preserved, so the
    /// tensor matmul path is always used. For pure NdArray backends, the
    /// direct path avoids all `[N, num_bins]` intermediate allocations.
    #[cfg(feature = "direct-parzen")]
    pub(crate) fn compute_joint_histogram_dispatch(
        &self,
        fixed: &Tensor<f32, B>,
        moving: &Tensor<f32, B>,
        oob_mask: Option<&Tensor<f32, B>>,
    ) -> Tensor<f32, B> {
        let [_n] = fixed.dims();
        let num_bins = self.num_bins;
        let device = fixed.device();

        let fix_cfg = self.fixed_sigma_cfg();
        let mov_cfg = self.moving_sigma_cfg();

        // Extract normalized data for the direct path
        let fixed_norm =
            normalize_and_extract(fixed, self.min_intensity, self.max_intensity, num_bins);
        let moving_norm = normalize_and_extract(
            moving,
            self.moving_min_intensity.unwrap_or(self.min_intensity),
            self.moving_max_intensity.unwrap_or(self.max_intensity),
            num_bins,
        );

        // Extract OOB mask if present (DRY-326-03)
        let oob_vec = extract_oob_mask(oob_mask);
        let oob_slice: Option<&[f32]> = oob_vec.as_deref();

        let hist_data = super::direct::compute_joint_histogram_direct(
            &fixed_norm,
            &moving_norm,
            num_bins,
            fix_cfg.sigma_sq(),
            mov_cfg.sigma_sq(),
            oob_slice,
            Some(&self.histogram_pool),
        );

        Tensor::from_data(hist_data, &device)
    }

    /// Compute joint histogram from cached W_fixed^T with backend dispatch.
    ///
    /// Despite the `direct-parzen` feature being enabled, this method always uses
    /// the tensor matmul path rather than the sparse-loop host extraction.
    ///
    /// **Rationale**: `moving_values` may carry an autodiff gradient tape (RSGD
    /// registration path with `Autodiff<B>` backend). Calling `into_data()` on it
    /// severs the tape, causing zero gradients and preventing convergence. The
    /// fixed-image cache (`w_fixed_transposed`) already provides the key perf
    /// benefit: Parzen weights for the constant fixed image are computed only once.
    /// The sparse cache path (`compute_joint_histogram_from_cache_sparse`) is only safe
    /// for derivative-free backends (CMA-ES uses `B::InnerBackend`), which take the
    /// `compute_joint_histogram_from_cache_sparse_dispatch` path instead.
    #[cfg(feature = "direct-parzen")]
    pub(crate) fn compute_joint_histogram_from_cache_dispatch(
        &self,
        w_fixed_transposed: &Tensor<f32, B>,
        moving_values: &Tensor<f32, B>,
        oob_mask: Option<&Tensor<f32, B>>,
    ) -> Tensor<f32, B> {
        // Always use the tensor path so autodiff gradient tape is preserved.
        self.compute_joint_histogram_from_cache(w_fixed_transposed, moving_values, oob_mask)
    }

    /// Compute joint histogram from a sparse W_fixed^T cache (direct CPU path).
    ///
    /// This is the sparse-optimized variant for derivative-free backends (CMA-ES).
    /// It extracts moving values to host memory and calls
    /// `compute_joint_histogram_from_cache_sparse`, which iterates only over
    /// the ~7 non-zero fixed-image bins per sample instead of all `num_bins`.
    /// Eliminates the `if w_f > 0.0` branch and the strided memory access
    /// pattern of the dense cache path.
    ///
    /// **Only safe for derivative-free backends** â€” calling `into_data()` on
    /// `moving_values` severs any autodiff tape. The tensor-path
    /// `compute_joint_histogram_from_cache_dispatch` must be used for RSGD.
    #[cfg(feature = "direct-parzen")]
    pub(crate) fn compute_joint_histogram_from_cache_sparse_dispatch(
        &self,
        sparse_w_fixed: &[(super::direct::SparseSampleCache, f32)],
        moving_values: &Tensor<f32, B>,
        oob_mask: Option<&Tensor<f32, B>>,
    ) -> Tensor<f32, B> {
        let num_bins = self.num_bins;
        let device = moving_values.device();

        let mov_min = self.moving_min_intensity.unwrap_or(self.min_intensity);
        let mov_max = self.moving_max_intensity.unwrap_or(self.max_intensity);
        let mov_sigma = self.moving_parzen_sigma.unwrap_or(self.parzen_sigma);

        let mov_cfg = super::direct::ParzenConfig::from_intensity_sigma(
            mov_sigma, mov_min, mov_max, num_bins,
        );

        let moving_norm = normalize_and_extract(moving_values, mov_min, mov_max, num_bins);

        // Extract OOB mask if present (DRY-326-03)
        let oob_vec = extract_oob_mask(oob_mask);
        let oob_slice: Option<&[f32]> = oob_vec.as_deref();

        let hist_data = super::direct::compute_joint_histogram_from_cache_sparse(
            sparse_w_fixed,
            &moving_norm,
            num_bins,
            mov_cfg.sigma_sq(),
            oob_slice,
            Some(&self.histogram_pool),
        );

        Tensor::from_data(hist_data, &device)
    }

    /// Fallback: always use the tensor matmul path.
    ///
    /// This is used when the `direct-parzen` feature is disabled, ensuring
    /// the standard tensor-based computation works with any backend.
    #[cfg(not(feature = "direct-parzen"))]
    pub(crate) fn compute_joint_histogram_dispatch(
        &self,
        fixed: &Tensor<f32, B>,
        moving: &Tensor<f32, B>,
        oob_mask: Option<&Tensor<f32, B>>,
    ) -> Tensor<f32, B> {
        self.compute_joint_histogram(fixed, moving, oob_mask)
    }

    #[cfg(not(feature = "direct-parzen"))]
    pub(crate) fn compute_joint_histogram_from_cache_dispatch(
        &self,
        w_fixed_transposed: &Tensor<f32, B>,
        moving_values: &Tensor<f32, B>,
        oob_mask: Option<&Tensor<f32, B>>,
    ) -> Tensor<f32, B> {
        self.compute_joint_histogram_from_cache(w_fixed_transposed, moving_values, oob_mask)
    }
}
