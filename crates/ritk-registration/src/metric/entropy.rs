//! Free-function entropy wrappers (Sprint 354, ARCH-354-02).
//!
//! Centralizes the Shannon-entropy computation `H(X) = -Î£ p log p` as a
//! free function so it can be reused across `MutualInformation`,
//! `NCC`-style marginal computations, and any future metric that needs
//! it, without going through the `ParzenJointHistogram` impl method.
//!
//! # Why a free function?
//!
//! The historical `ParzenJointHistogram::compute_entropy` is a method
//! on a struct that carries histogram-shape context. But the entropy
//! formula itself is independent of bin count or windowing â€” it only
//! needs a probability vector. Exposing it as a free function lets:
//!
//! 1. `MutualInformation` call it without holding a `ParzenJointHistogram`
//!    reference (cleaner SoC).
//! 2. Future metrics reuse it without the histogram-calculator boilerplate.
//! 3. Test the entropy formula in isolation.
//!
//! # Formula
//!
//! ```text
//! H(X) = -Î£_i p_i Â· log(p_i + Îµ)
//! ```
//!
//! The `+ Îµ` (default `1e-10`) guards `log(0) = -âˆž` for empty bins. It
//! is a tiny perturbation on `p_i = 0` and a negligible perturbation
//! on the `p_i` that actually carry mass.

use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

/// Default epsilon for `log(0)` protection in entropy computation.
pub const DEFAULT_ENTROPY_EPS: f32 = 1e-10;

/// Compute the Shannon entropy of a probability distribution.
///
/// # Arguments
/// * `p` - A `[N]` tensor of non-negative values that sum to ~1.0.
///   Inputs are **not** re-normalized; callers must pass a valid PMF.
///
/// # Returns
/// A scalar `Tensor<f32, B>` (shape `[1]`) holding `H(X) = -Î£ p log(p + Îµ)`.
///
/// # Formula
/// ```text
/// H(X) = -Î£_i p_i Â· log(p_i + Îµ)
/// ```
///
/// # Example
/// ```ignore
/// use ritk_registration::metric::entropy::entropy;
/// let p = Tensor::<f32, B>::from_floats([0.5, 0.5], &device);
/// let h = entropy(p); // â†’ ln(2) â‰ˆ 0.693
/// ```
#[inline]
pub fn entropy<B: Backend>(p: Tensor<f32, B>) -> Tensor<f32, B> {
    entropy_with_eps(p, DEFAULT_ENTROPY_EPS)
}

/// Compute the Shannon entropy with an explicit epsilon.
///
/// Prefer [`entropy`] unless you need to tune the `log(0)` guard.
#[inline]
pub fn entropy_with_eps<B: Backend>(p: Tensor<f32, B>, eps: f32) -> Tensor<f32, B> {
    let log_p = (p.clone() + eps).log();
    p.mul(log_p).sum().neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type B = SequentialBackend;

    /// Uniform distribution over N bins has entropy ln(N).
    #[test]
    fn uniform_distribution_entropy_is_log_n() {
        let device = B;
        let n = 4;
        let p_vec = vec![0.25_f32; 4];
        let p = Tensor::<f32, B>::from_slice_on([n], &p_vec, &device);
        let h = entropy(p);
        let v = h.as_slice()[0];
        let expected = (n as f32).ln();
        assert!(
            (v - expected).abs() < 1e-5,
            "H(uniform[4]) must equal ln(4) â‰ˆ {expected}, got {v}"
        );
    }

    /// Deterministic distribution (one-hot) has zero entropy.
    #[test]
    fn deterministic_distribution_entropy_is_zero() {
        let device = Default::default();
        let p_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let p = Tensor::<f32, B>::from_data(::new(p_vec, [4]), &device);
        let h = entropy(p);
        let v = h.into_data().as_slice::<f32>().unwrap()[0];
        assert!(v.abs() < 1e-5, "H([1, 0, 0, 0]) must be 0.0, got {v}");
    }

    /// `entropy` and `entropy_with_eps` agree for the default epsilon.
    #[test]
    fn entropy_with_default_eps_matches() {
        let device = Default::default();
        let p = Tensor::<f32, B>::from_data(::new(vec![0.5_f32, 0.3, 0.2], [3]), &device);
        let h1 = entropy(p.clone());
        let h2 = entropy_with_eps(p, DEFAULT_ENTROPY_EPS);
        let v1 = h1.into_data().as_slice::<f32>().unwrap()[0];
        let v2 = h2.into_data().as_slice::<f32>().unwrap()[0];
        assert!((v1 - v2).abs() < 1e-9);
    }
}
