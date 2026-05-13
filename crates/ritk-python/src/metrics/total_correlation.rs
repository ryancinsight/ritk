//! Total Correlation (Multivariate Mutual Information) over N aligned images.
//!
//! Delegates to `ritk_core::statistics::information::total_correlation`.
//! See that module for the mathematical definition (Watanabe 1960) and
//! complexity constraints (B^n ≤ 4_194_304).

use anyhow::Result;
use ritk_core::statistics::information::total_correlation as core_tc;

/// Total correlation C(X₁,...,Xₙ) = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ).
///
/// Delegates to `ritk_core::statistics::information::total_correlation`.
pub(super) fn total_correlation_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_tc(channels, num_bins)
}
