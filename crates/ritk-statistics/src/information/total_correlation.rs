//! Total Correlation (multivariate mutual information).
//!
//! # Definition (Watanabe 1960)
//!
//! TC(X₁,...,Xₙ) = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ)
//!
//! For n = 2: TC(X,Y) = I(X;Y).
//! TC = 0 iff all channels are mutually independent.
//!
//! # Complexity
//!
//! Joint histogram requires `num_bins^n` entries.
//! Enforced limit: `num_bins^n ≤ 4_194_304` (~32 MB at f64).
//!
//! # References
//!
//! - Watanabe, S. (1960). IBM J. Research and Development, 4(1), 66–82.
//! - Studholme, C., et al. (1999). Pattern Recognition, 32(1), 71–86.

use anyhow::Result;

use super::entropy::{joint_entropy_n, marginal_entropy};

/// TC(X₁,...,Xₙ) = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ).
///
/// Returns `max(TC, 0.0)` — negative values are numerical artefacts from
/// finite-bin histograms near the independence boundary.
///
/// # Arguments
/// - `channels`: equal-length `f32` slices, n ≥ 1.
/// - `num_bins`: 2 ≤ B ≤ 64; `B^n ≤ 4_194_304`.
///
/// # Errors
/// Propagates errors from [`joint_entropy_n`] and [`marginal_entropy`].
pub fn total_correlation(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    let sum_h: f64 = channels
        .iter()
        .map(|ch| marginal_entropy(ch, num_bins))
        .try_fold(0.0_f64, |acc, r| r.map(|v| acc + v))?;
    let h_joint = joint_entropy_n(channels, num_bins)?;
    Ok((sum_h - h_joint).max(0.0))
}
