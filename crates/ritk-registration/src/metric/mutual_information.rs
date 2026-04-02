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
use std::marker::PhantomData;

use crate::metric::{histogram::ParzenJointHistogram, Metric};

/// Normalization method for Normalized Mutual Information (NMI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Normalize by joint entropy: (H(X) + H(Y)) / H(X,Y)
    JointEntropy,
    /// Normalize by average of marginal entropies: 2 * MI / (H(X) + H(Y))
    AverageEntropy,
    /// Normalize by minimum: MI / min(H(X), H(Y))
    MinEntropy,
    /// Normalize by maximum: MI / max(H(X), H(Y))
    MaxEntropy,
}

/// Variant of Mutual Information to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutualInformationVariant {
    /// Standard Mutual Information (Viola-Wells).
    Standard,
    /// Mattes Mutual Information (Cubic B-Spline approximation via parameterization).
    Mattes,
    /// Normalized Mutual Information.
    Normalized(NormalizationMethod),
}

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
    _phantom: PhantomData<B>,
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
    ) -> Self {
        assert!(num_bins >= 4, "num_bins must be ≥ 4");
        Self {
            variant,
            histogram_calculator: ParzenJointHistogram::new(
                num_bins,
                min_intensity,
                max_intensity,
                parzen_sigma,
            ),
            sampling_percentage: None,
            interpolator: LinearInterpolator::new(),
            _phantom: PhantomData,
        }
    }

    /// Helper to create Mattes Mutual Information with its characteristic parameterization.
    /// Mattes specifies `parzen_sigma = bin_width`.
    pub fn new_mattes(num_bins: usize, min_intensity: f32, max_intensity: f32) -> Self {
        let bin_width = (max_intensity - min_intensity).max(1e-6) / num_bins as f32;
        Self::new(
            MutualInformationVariant::Mattes,
            num_bins,
            min_intensity,
            max_intensity,
            bin_width,
        )
    }

    /// Create with default Mattes parameters (50 bins, [0, 255]).
    pub fn mattes_default() -> Self {
        Self::new_mattes(50, 0.0, 255.0).with_sampling(0.20)
    }

    /// Create with default Standard parameters (32 bins, [0, 255]).
    pub fn standard_default() -> Self {
        Self::new(MutualInformationVariant::Standard, 32, 0.0, 255.0, 1.0)
    }

    /// Create with default Normalized parameters (JointEntropy, 32 bins).
    pub fn normalized_default() -> Self {
        Self::new(
            MutualInformationVariant::Normalized(NormalizationMethod::JointEntropy),
            32,
            0.0,
            255.0,
            1.0,
        )
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
}

impl<B: Backend, const D: usize> Metric<B, D> for MutualInformation<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Joint Histogram built strictly through shared Parzen infrastructure
        let joint_hist = self.histogram_calculator.compute_image_joint_histogram(
            fixed,
            moving,
            transform,
            &self.interpolator,
            self.sampling_percentage,
        );

        // 2. Normalize to joint PDF p̂(x,y)
        let sum = joint_hist.clone().sum();
        let eps = 1e-10_f32;
        let p_xy = joint_hist / (sum.unsqueeze_dim(1) + eps);

        // 3. Compute marginals p̂(x), p̂(y)
        let p_x = p_xy.clone().sum_dim(1).squeeze::<1>();
        let p_y = p_xy.clone().sum_dim(0).squeeze::<1>();

        // 4. Entropy computations
        // H(X) = -Σ p(x) log p(x)
        let h_x = self.histogram_calculator.compute_entropy(p_x);
        let h_y = self.histogram_calculator.compute_entropy(p_y);

        // H(X,Y) = -Σ p(x,y) log p(x,y)
        let log_p_xy = (p_xy.clone() + eps).log();
        let h_xy = p_xy.mul(log_p_xy).sum().neg();

        // 5. Variant routing
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

    fn name(&self) -> &'static str {
        match self.variant {
            MutualInformationVariant::Standard => "Mutual Information",
            MutualInformationVariant::Mattes => "Mattes Mutual Information",
            MutualInformationVariant::Normalized(_) => "Normalized Mutual Information",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_mutual_information_consolidation() {
        let max_intensity = 255.0;
        let num_bins = 50;

        // Mattes specific configuration check
        let m_mattes = MutualInformation::<B>::new_mattes(num_bins, 0.0, max_intensity);
        let expected_sigma = max_intensity / 50.0;

        assert_eq!(MutualInformationVariant::Mattes, m_mattes.variant);
        assert!((m_mattes.histogram_calculator.parzen_sigma - expected_sigma).abs() < 1e-6);

        // Check correct name routing
        assert_eq!(
            <MutualInformation<B> as Metric<B, 3>>::name(&m_mattes),
            "Mattes Mutual Information"
        );

        let m_std = MutualInformation::<B>::standard_default();
        assert_eq!(
            <MutualInformation<B> as Metric<B, 3>>::name(&m_std),
            "Mutual Information"
        );

        let m_nmi = MutualInformation::<B>::normalized_default();
        assert_eq!(
            <MutualInformation<B> as Metric<B, 3>>::name(&m_nmi),
            "Normalized Mutual Information"
        );
    }

    #[test]
    fn test_sampling_clamp() {
        let m = MutualInformation::<B>::standard_default().with_sampling(1.5);
        assert_eq!(m.sampling_percentage, None); // Clamps to 1.0, which disables stochastic branch

        let m = MutualInformation::<B>::standard_default().with_sampling(-0.5);
        assert_eq!(m.sampling_percentage, Some(1e-4)); // Clamps to 1e-4
    }
}
