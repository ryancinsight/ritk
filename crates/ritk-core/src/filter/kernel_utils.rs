//! Kernel construction utilities for filter implementations.
//!
//! This module owns the generic 1-D Gaussian kernel builder that is shared
//! across CPU, GPU, and registration filter paths.  It is kept in `ritk-core`
//! (rather than `ritk-filter`) to avoid a circular dependency:
//! `ritk-filter` → `ritk-core` ← `ritk-filter`.
//!
//! # Exports
//!
//! - [`gaussian_kernel`]: normalised symmetric 1-D Gaussian kernel.

use num_traits::Float;
use std::ops::AddAssign;

// ── gaussian_kernel ─────────────────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel of type `T`.
///
/// The kernel is symmetric, centred, and sums to `T::one()` (probability-
/// preserving convolution). If `sigma <= T::zero()`, returns `vec![T::one()]`
/// (identity kernel). When `radius` is `None`, the radius defaults to
/// `⌈3σ⌉`, which captures >99.7% of the Gaussian mass.
///
/// # Formula
///
/// ```text
/// w_i = exp(−d² / (2σ²)) / Z,   d = i − radius,   Z = Σ w_i
/// ```
///
/// # Invariants
///
/// - `kernel.len() == 2 * radius + 1`
/// - `kernel[radius]` is the peak value
/// - `kernel[i] == kernel[len − 1 − i]` (symmetry)
/// - `Σ kernel[i] == 1.0` within floating-point rounding
///
/// # Type parameters
///
/// `T: Float + AddAssign + Default` — supports `f32` and `f64`. Monomorphisation
/// emits zero-cost specialisations identical to hand-written concrete versions.
///
/// # Evidence tier
///
/// Property-tested: normalisation, symmetry, peak-at-centre, length (see tests
/// in `filter::kernel_utils`, `level_set::helpers`, `level_set::geodesic_active_contour`).
pub fn gaussian_kernel<T>(sigma: T, radius: Option<usize>) -> Vec<T>
where
    T: Float + AddAssign + Default,
{
    let zero = T::zero();
    let one = T::one();

    if sigma <= zero {
        return vec![one];
    }

    let r = radius.unwrap_or_else(|| (T::from(3.0).unwrap() * sigma).ceil().to_usize().unwrap());
    // SAFETY: 2.0 is exactly representable in every IEEE-754 float type.
    let two_sigma2 = T::from(2.0).unwrap() * sigma * sigma; // 2σ²
    let len = 2 * r + 1;

    let mut kernel = Vec::with_capacity(len);
    let mut sum = zero;
    for i in 0..len {
        let d = T::from(i).unwrap() - T::from(r).unwrap();
        let w = (-d * d / two_sigma2).exp();
        kernel.push(w);
        sum += w;
    }

    let inv_sum = one / sum;
    for w in &mut kernel {
        *w = *w * inv_sum;
    }

    kernel
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // ── gaussian_kernel ────────────────────────────────────────────────

    /// Kernel sums to 1.0 (f64 variant).
    #[test]
    fn gaussian_kernel_f64_sums_to_one() {
        let kernel = super::gaussian_kernel(2.0_f64, None);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "kernel sum = {sum}");
    }

    /// Kernel sums to 1.0 (f32 variant).
    #[test]
    fn gaussian_kernel_f32_sums_to_one() {
        let kernel = super::gaussian_kernel(2.0_f32, None);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
    }

    /// Zero sigma returns identity kernel.
    #[test]
    fn gaussian_kernel_zero_sigma_is_identity() {
        let kernel = super::gaussian_kernel(0.0_f64, None);
        assert_eq!(kernel, vec![1.0_f64]);
    }

    /// Explicit radius overrides the default.
    #[test]
    fn gaussian_kernel_explicit_radius() {
        let kernel = super::gaussian_kernel(1.0_f64, Some(5));
        assert_eq!(kernel.len(), 11); // 2 * 5 + 1
    }

    /// Centre-to-adjacent ratio verifies the exponent denominator is exactly 2σ².
    ///
    /// # Derivation
    /// For d=1 from centre: w₁/w₀ = exp(-1 / (2σ²)).
    /// With σ=2.0: expected = exp(-1/8) ≈ 0.882497.
    /// The previous defect (`1 + σ²` = 5) would produce exp(-1/5) ≈ 0.818731 — a ~7% error.
    #[test]
    fn gaussian_kernel_exponent_denominator_is_two_sigma_squared() {
        let sigma = 2.0_f64;
        let kernel = super::gaussian_kernel(sigma, Some(4));
        let centre = 4_usize; // r = 4, centre = index 4
        let expected_ratio = (-1.0_f64 / (2.0 * sigma * sigma)).exp();
        let actual_ratio = kernel[centre - 1] / kernel[centre];
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-12,
            "ratio kernel[r-1]/kernel[r] = {actual_ratio:.9}, expected exp(-1/(2σ²)) = {expected_ratio:.9}"
        );
    }

    /// Kernel is symmetric.
    #[test]
    fn gaussian_kernel_is_symmetric() {
        let kernel = super::gaussian_kernel(2.0_f64, None);
        let n = kernel.len();
        for i in 0..n {
            assert!(
                (kernel[i] - kernel[n - 1 - i]).abs() < 1e-15,
                "asymmetry at i={i}: {} vs {}",
                kernel[i],
                kernel[n - 1 - i]
            );
        }
    }

    /// Peak is at the centre.
    #[test]
    fn gaussian_kernel_peak_at_centre() {
        let kernel = super::gaussian_kernel(1.0_f64, Some(3));
        let center = kernel.len() / 2;
        for (i, &w) in kernel.iter().enumerate() {
            if i != center {
                assert!(
                    kernel[center] >= w,
                    "centre {} < kernel[{i}] = {w}",
                    kernel[center]
                );
            }
        }
    }
}
