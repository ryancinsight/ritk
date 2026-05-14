//! O-Information and Dual Total Correlation (multivariate synergy/redundancy).
//!
//! # Definitions
//!
//! ## Dual Total Correlation (Han 1978)
//!
//! DTC(X₁,...,Xₙ) = Σᵢ H(X₁,...,Xₙ\Xᵢ) − (n−1)·H(X₁,...,Xₙ)
//!
//! Derivation:
//!   H(Xᵢ | {Xⱼ}_{j≠i}) = H(X₁,...,Xₙ) − H({Xⱼ}_{j≠i})
//!   DTC = H(X₁,...,Xₙ) − Σᵢ H(Xᵢ | {Xⱼ}_{j≠i})
//!       = H(X₁,...,Xₙ) − Σᵢ [H(X₁,...,Xₙ) − H({Xⱼ}_{j≠i})]
//!       = Σᵢ H({Xⱼ}_{j≠i}) − (n−1)·H(X₁,...,Xₙ)
//!
//! DTC ≥ 0 always (cf. conditional entropy chain rule).
//! For n = 2: DTC(X,Y) = H(X) + H(Y) − H(X,Y) = I(X;Y) = TC(X,Y).
//!
//! ## O-Information (Rosas et al. 2019)
//!
//! Ω(X₁,...,Xₙ) = TC(X₁,...,Xₙ) − DTC(X₁,...,Xₙ)
//!              = Σᵢ H(Xᵢ) − Σᵢ H({Xⱼ}_{j≠i}) + (n−2)·H(X₁,...,Xₙ)
//!
//! Interpretation:
//!   Ω > 0: system is redundancy-dominated (more redundant shared information).
//!   Ω < 0: system is synergy-dominated (more synergistic emergent information).
//!   Ω = 0: balanced (or independent).
//!
//! For n = 3: Ω(X,Y,Z) = II(X;Y;Z) (O-information generalises interaction information).
//!
//! # Complexity
//!
//! Each (n−1)-way sub-histogram requires num_bins^(n−1) entries.
//! The existing 4_194_304-entry limit in `joint_entropy_n` applies automatically.
//!
//! # References
//!
//! - Han, T. S. (1978). *Inform. Control*, 36(2), 133–156.
//! - Rosas, F. E., et al. (2019). *Phys. Rev. E*, 100(3), 032305.

use anyhow::Result;

use super::entropy::marginal_entropy;
use super::total_correlation::total_correlation;

/// DTC(X₁,...,Xₙ) = Σᵢ H(X₁,...,Xₙ\Xᵢ) − (n−1)·H(X₁,...,Xₙ).
///
/// Returns `max(DTC, 0.0)` — negative values are numerical artefacts from
/// finite-bin histograms near the independence boundary.
///
/// # Arguments
/// - `channels`: equal-length `f32` slices, n ≥ 2.
/// - `num_bins`: 2 ≤ B ≤ 64; `B^(n-1) ≤ 4_194_304`.
///
/// # Errors
/// Returns an error when n < 2, channels are empty, lengths differ,
/// `num_bins < 2`, or a sub-histogram exceeds the 4_194_304 limit.
pub fn dual_total_correlation(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    let n = channels.len();
    if n < 2 {
        anyhow::bail!("dual_total_correlation requires at least 2 channels, got {n}");
    }
    
    // Performance/Memory optimization: build the full N-dimensional joint histogram ONCE.
    // Marginalizing this avoids O(N^2 * num_samples) complexity.
    let joint_hist = super::entropy::build_joint_hist_n(channels, num_bins)?;
    let h_joint = super::entropy::entropy_from_hist_pub(&joint_hist);
    
    let sum_h_minus_i: f64 = (0..n)
        .map(|i| {
            let sub_hist = super::entropy::marginalize_hist(&joint_hist, num_bins, n, i);
            super::entropy::entropy_from_hist_pub(&sub_hist)
        })
        .sum();
        
    Ok((sum_h_minus_i - (n - 1) as f64 * h_joint).max(0.0))
}

/// Ω(X₁,...,Xₙ) = TC(X₁,...,Xₙ) − DTC(X₁,...,Xₙ).
///
/// May be negative (synergy-dominated) or positive (redundancy-dominated).
///
/// # Arguments
/// - `channels`: equal-length `f32` slices, n ≥ 2.
/// - `num_bins`: 2 ≤ B ≤ 64; `B^(n-1) ≤ 4_194_304`.
///
/// # Errors
/// Propagates errors from [`total_correlation`] and [`dual_total_correlation`].
pub fn o_information(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    let n = channels.len();
    if n < 2 {
        anyhow::bail!("o_information requires at least 2 channels, got {n}");
    }
    // Ω = Σᵢ H(Xᵢ) − Σᵢ H(X₁,...,Xₙ\Xᵢ) + (n−2)·H(X₁,...,Xₙ)
    // Computed as TC − DTC to share the joint-entropy computation cost.
    let tc = total_correlation(channels, num_bins)?;
    let dtc = dual_total_correlation(channels, num_bins)?;
    Ok(tc - dtc)
}

/// O-Information computed from pre-computed TC and DTC (zero extra histogram work).
///
/// Useful when the caller already holds both values and needs Ω without recomputation.
///
/// # Arguments
/// - `tc`: Total Correlation (Watanabe 1960) — result of `total_correlation(...)`.
/// - `dtc`: Dual Total Correlation (Han 1978) — result of `dual_total_correlation(...)`.
#[inline]
pub fn o_information_from_tc_dtc(tc: f64, dtc: f64) -> f64 {
    tc - dtc
}

/// O-information via direct expansion (single-pass over all histograms).
///
/// Ω = Σᵢ H(Xᵢ) − Σᵢ H(X₁,...,Xₙ\Xᵢ) + (n−2)·H(X₁,...,Xₙ)
///
/// Equivalent to `o_information` but avoids the intermediate `total_correlation`
/// and `dual_total_correlation` calls when the caller only needs Ω and neither
/// TC nor DTC separately.
///
/// # Errors
/// Same conditions as `dual_total_correlation`.
pub fn o_information_direct(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    let n = channels.len();
    if n < 2 {
        anyhow::bail!("o_information_direct requires at least 2 channels, got {n}");
    }
    
    // Performance/Memory optimization: build the full N-dimensional joint histogram ONCE.
    let joint_hist = super::entropy::build_joint_hist_n(channels, num_bins)?;
    let h_joint = super::entropy::entropy_from_hist_pub(&joint_hist);
    
    let sum_h_marginal: f64 = channels
        .iter()
        .map(|ch| marginal_entropy(ch, num_bins))
        .try_fold(0.0_f64, |acc, r| r.map(|v| acc + v))?;
        
    let sum_h_minus_i: f64 = (0..n)
        .map(|i| {
            let sub_hist = super::entropy::marginalize_hist(&joint_hist, num_bins, n, i);
            super::entropy::entropy_from_hist_pub(&sub_hist)
        })
        .sum();
        
    Ok(sum_h_marginal - sum_h_minus_i + (n as f64 - 2.0) * h_joint)
}
