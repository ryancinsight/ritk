//! Mutual information estimators via histogram binning.
//!
//! # Definitions
//!
//! I(X;Y) = H(X) + H(Y) − H(X,Y)   (Shannon 1948)
//!
//! NMI(X,Y) = (H(X) + H(Y)) / H(X,Y)   (Studholme et al. 1999)

use anyhow::{bail, Result};

use super::entropy::{joint_entropy, marginal_entropy};

/// Standard bivariate mutual information I(X;Y) = H(X) + H(Y) − H(X,Y).
///
/// Returns `max(I, 0.0)` — negative values are numerical artefacts from
/// finite-bin histograms where I(X;Y) ≈ 0.
///
/// # Errors
/// Propagates errors from [`joint_entropy`] and [`marginal_entropy`].
pub fn mutual_information(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    if a.len() != b.len() {
        bail!("channel lengths differ: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    let h_a = marginal_entropy(a, num_bins)?;
    let h_b = marginal_entropy(b, num_bins)?;
    let h_ab = joint_entropy(a, b, num_bins)?;
    Ok((h_a + h_b - h_ab).max(0.0))
}

/// Normalized mutual information NMI(X,Y) = (H(X) + H(Y)) / H(X,Y).
///
/// Returns `1.0` when H(X,Y) < ε (both signals are identical constants).
/// NMI ∈ [1.0, 2.0]: value 1 means independent; value 2 means identical.
///
/// # Errors
/// Propagates errors from [`joint_entropy`] and [`marginal_entropy`].
pub fn normalized_mutual_information(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    if a.len() != b.len() {
        bail!("channel lengths differ: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    let h_a = marginal_entropy(a, num_bins)?;
    let h_b = marginal_entropy(b, num_bins)?;
    let h_ab = joint_entropy(a, b, num_bins)?;
    if h_ab < f64::EPSILON {
        return Ok(1.0);
    }
    Ok((h_a + h_b) / h_ab)
}
