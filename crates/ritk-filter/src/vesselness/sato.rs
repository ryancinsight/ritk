//! Sato multi-scale line filter for curvilinear structure detection.
//!
//! # Mathematical Specification
//!
//! **Reference:** Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
//! Koller, T., Gerig, G. & Kikinis, R. (1998). Three-dimensional multi-scale line
//! filter for segmentation and visualization of curvilinear structures in medical
//! images. *Medical Image Analysis* 2(2):143–168.
//!
//! At each Gaussian scale σ the normalised Hessian `H_σ = σ² · H(I_σ)` is
//! computed, where `I_σ = G_σ ∗ I`.  The three eigenvalues
//! `λ₁, λ₂, λ₃` are sorted by absolute value so that `|λ₁| ≤ |λ₂| ≤ |λ₃|`.
//!
//! For a bright tubular structure on a dark background the expected pattern is:
//!
//! ```text
//!   λ₁ ≈ 0,  λ₂ < 0,  λ₃ < 0   (two strongly negative, one near zero)
//! ```
//!
//! **Line response function** (Sato 1998, eq. 5–7):
//!
//! If `λ₂ < 0` AND `λ₃ < 0`:
//!
//! ```text
//!   V(λ₁,λ₂,λ₃) = |λ₃| · (λ₂/λ₃)^α · f(λ₁,λ₂)
//!
//!   where  f(λ₁,λ₂) = 1                                  if λ₁ ≤ 0
//!                     = exp(−λ₁² / (2·(α·λ₂)²))           if λ₁ > 0
//! ```
//!
//! `α` (default 0.5) controls cross-section anisotropy tolerance.
//! Higher α → more permissive to elliptical cross-sections.
//!
//! For dark tubes on a bright background set `bright_tubes = false`, which
//! inverts the sign convention: `λ₂ > 0` and `λ₃ > 0` are required and the
//! response is computed using `|λ₂|` and `|λ₃|` with the same formula.
//!
//! The final output is the **maximum** response over all scales σ.

use super::hessian::symmetric_3x3_eigenvalues;
use super::VesselPolarity;
use crate::recursive_gaussian::compute_hessian_iir;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for the Sato line filter.
#[derive(Debug, Clone)]
pub struct SatoConfig {
    /// Gaussian scale values σ (physical units, e.g. mm) at which to evaluate
    /// the filter. The output is the per-voxel maximum over all scales.
    pub scales: Vec<f64>,
    /// Cross-section anisotropy exponent. Controls how strongly the ratio
    /// `λ₂/λ₃` is penalised. Typical range: [0.5, 2.0]. Default: 0.5.
    pub alpha: f64,
    /// Vessel polarity: detect bright structures on a dark background
    /// or dark structures on a bright background.
    pub polarity: VesselPolarity,
}

impl Default for SatoConfig {
    fn default() -> Self {
        Self {
            scales: vec![1.0, 2.0, 4.0],
            alpha: 0.5,
            polarity: VesselPolarity::Bright,
        }
    }
}

/// Multi-scale Sato line filter.
///
/// Produces a line-probability map in `[0, ∞)` (not normalised to 1 because the
/// raw Hessian eigenvalue magnitudes carry scale information useful for
/// downstream thresholding).  The output has the same shape and spatial metadata
/// as the input image.
///
/// # Example
/// ```rust,ignore
/// let config = SatoConfig { scales: vec![1.0, 2.0], ..Default::default() };
/// let filter = SatoLineFilter::new(config);
/// let line_map = filter.apply(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct SatoLineFilter {
    /// Algorithm configuration.
    pub config: SatoConfig,
}

impl SatoLineFilter {
    /// Construct with explicit configuration.
    pub fn new(config: SatoConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to a 3-D image.
    ///
    /// The input tensor must have `f32` element type.
    ///
    /// # Errors
    /// Returns an error if the image tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let vals = &vals_vec;

        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];

        let response = compute_sato_multiscale(vals, dims, spacing, &self.config);

        Ok(rebuild(response, dims, image))
    }

    /// Coeus-native sister of [`SatoLineFilter::apply`].
    ///
    /// Runs the identical multi-scale Sato line response (Hessian eigen-analysis
    /// via recursive-Gaussian derivatives) through the shared
    /// `compute_sato_multiscale` host core on the image's contiguous host
    /// buffer, so the result is bitwise-identical to the Coeus path. No Coeus
    /// tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            compute_sato_multiscale(vals, dims, spacing, &self.config)
        })
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Compute per-voxel maximum Sato line response over all scales.
fn compute_sato_multiscale(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    config: &SatoConfig,
) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    let mut max_response = vec![0.0_f32; n];

    for &sigma in &config.scales {
        // Compute Hessian via second-order Deriche IIR recursion —
        // matching ITK HessianRecursiveGaussianImageFilter.
        let hessian = compute_hessian_iir(data, dims, spacing, sigma);

        // Per-voxel line response (scale-normalised by σ²).
        let sigma2 = (sigma * sigma) as f32;
        let hessian_ref = &hessian;
        moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
            &mut max_response,
            4096,
            |chunk_idx, slice| {
                let start_idx = chunk_idx * 4096;
                for (offset, max_val) in slice.iter_mut().enumerate() {
                    let i = start_idx + offset;
                    let h = hessian_ref[i];
                    // Scale-normalise Hessian (σ² · H_σ convention).
                    let h_scaled = [
                        h[0] * sigma2,
                        h[1] * sigma2,
                        h[2] * sigma2,
                        h[3] * sigma2,
                        h[4] * sigma2,
                        h[5] * sigma2,
                    ];
                    let eigs = symmetric_3x3_eigenvalues(h_scaled);
                    let v = sato_response(eigs, config.alpha, config.polarity);
                    if v > *max_val {
                        *max_val = v;
                    }
                }
            },
        );
    }

    max_response
}

/// Compute the Sato line response for a single voxel given its three Hessian
/// eigenvalues in arbitrary order.
///
/// # Algorithm
/// 1. Sort eigenvalues by absolute value: `|λ₁| ≤ |λ₂| ≤ |λ₃|`.
/// 2. For bright tubes: require `λ₂ < 0` and `λ₃ < 0`.
///    For dark tubes:  require `λ₂ > 0` and `λ₃ > 0`.
///    Inversion for dark tubes: negate all eigenvalues before the test.
/// 3. Compute `V = |λ₃| · (λ₂/λ₃)^α · f(λ₁,λ₂)`.
#[inline]
fn sato_response(eigenvalues: [f32; 3], alpha: f64, polarity: VesselPolarity) -> f32 {
    // Sort by absolute value (bubble-sort on 3 elements; branchless-friendly).
    let mut e = eigenvalues;
    if e[0].abs() > e[1].abs() {
        e.swap(0, 1);
    }
    if e[1].abs() > e[2].abs() {
        e.swap(1, 2);
    }
    if e[0].abs() > e[1].abs() {
        e.swap(0, 1);
    }
    // Now |e[0]| ≤ |e[1]| ≤ |e[2]|  (λ₁, λ₂, λ₃).
    let [lam1, lam2, lam3] = e;

    // For dark tubes invert all signs so that the bright-tube gate applies.
    let (l1, l2, l3) = match polarity {
        VesselPolarity::Bright => (lam1, lam2, lam3),
        VesselPolarity::Dark => (-lam1, -lam2, -lam3),
    };

    // Gate: both l2 and l3 must be strictly negative.
    if l2 >= 0.0 || l3 >= 0.0 {
        return 0.0;
    }

    // Avoid numerical instability: |l3| must be non-trivial.
    let abs_l3 = l3.abs();
    if abs_l3 < f32::EPSILON {
        return 0.0;
    }

    // Ratio λ₂/λ₃ ∈ (0, 1] (both negative → positive ratio).
    let ratio = l2 / l3; // ∈ (0,1] since |l2| ≤ |l3| and both negative.

    // Shape anisotropy term: (λ₂/λ₃)^α
    let shape_term = (ratio as f64).powf(alpha) as f32;

    // Perpendicular modulation: f(λ₁, λ₂)
    let perp_term = if l1 <= 0.0 {
        1.0_f32
    } else {
        // exp(−λ₁² / (2·(α·λ₂)²))
        let denom = 2.0 * (alpha * l2 as f64) * (alpha * l2 as f64);
        if denom < 1e-30 {
            0.0
        } else {
            (-(l1 as f64 * l1 as f64) / denom).exp() as f32
        }
    };

    abs_l3 * shape_term * perp_term
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_sato.rs"]
mod tests;

#[cfg(test)]
mod tests_native {
    use super::{SatoConfig, SatoLineFilter};
    use crate::native_support::{assert_coeus_matches_coeus, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    #[test]
    fn matches_coeus() {
        let vals: Vec<f32> = (0..210).map(|i| ((i * 5) % 23) as f32).collect();
        assert_coeus_matches_coeus(
            vals,
            [5, 6, 7],
            |img| {
                SatoLineFilter::new(SatoConfig::default())
                    .apply(img)
                    .expect("burn sato")
            },
            |img, backend| SatoLineFilter::new(SatoConfig::default()).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_constant_field_zero_response() {
        // The Hessian of a constant field is zero → all eigenvalues zero →
        // vesselness response is exactly zero everywhere.
        let img = make_native_image(vec![9.0f32; 343], [7, 7, 7]);
        let out = SatoLineFilter::new(SatoConfig::default())
            .apply_native(&img, &SequentialBackend)
            .expect("native sato");
        for &v in &native_vals(&out) {
            assert_eq!(v, 0.0, "constant field must give zero line response");
        }
    }
}
