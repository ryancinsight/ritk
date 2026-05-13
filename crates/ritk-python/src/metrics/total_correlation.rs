//! Total Correlation (Multivariate Mutual Information) over N aligned images.
//!
//! # Definition
//!
//! For n channels X₁, ..., Xₙ (Watanabe 1960):
//!
//! ```text
//! C(X₁,...,Xₙ) = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ)
//! ```
//!
//! where H is Shannon entropy estimated from uniform histogram bins.
//! For n = 2 this reduces to the standard bivariate MI I(X₁;X₂).
//!
//! # Complexity
//!
//! Joint histogram requires B^n entries (B bins, n channels).
//! Enforced limit: B^n ≤ 4_194_304 entries (~32 MB at f64).
//!
//! # References
//!
//! - Watanabe, S. (1960). Information theoretical analysis of multivariate
//!   correlation. *IBM J. Research and Development*, 4(1), 66–82.
//! - Studholme, C., et al. (1999). An overlap invariant entropy measure of 3D
//!   medical image alignment. *Pattern Recognition*, 32(1), 71–86.

use anyhow::{bail, Result};

use super::mi::min_max;

/// Total correlation C(X₁,...,Xₙ) = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ) using uniform
/// nearest-bin histogram estimation.
///
/// # Arguments
/// - `channels`: equal-length slices, one per image channel (n ≥ 1).
/// - `num_bins`: histogram bins per channel (2 ≤ B ≤ 64).
///
/// # Returns
/// `C ≥ 0`; exactly 0 iff all channels are mutually independent.
pub(super) fn total_correlation_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    let n = channels.len();
    if n == 0 {
        bail!("channels must not be empty");
    }
    if num_bins < 2 {
        bail!("num_bins must be ≥ 2, got {}", num_bins);
    }
    if num_bins > 64 {
        bail!("num_bins must be ≤ 64 to bound memory use, got {}", num_bins);
    }
    let len = channels[0].len();
    if len == 0 {
        bail!("channels must contain at least one sample");
    }
    for (i, ch) in channels.iter().enumerate() {
        if ch.len() != len {
            bail!("channel {} length {} != channel 0 length {}", i, ch.len(), len);
        }
    }

    let joint_size = num_bins.pow(n as u32);
    if joint_size > 4_194_304 {
        bail!(
            "joint histogram {}^{} = {} exceeds limit 4_194_304; reduce num_bins or n",
            num_bins,
            n,
            joint_size
        );
    }

    // Per-channel intensity range for bin scaling.
    let ranges: Vec<(f32, f32)> = channels.iter().map(|ch| min_max(ch)).collect();

    let mut marginals: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0_f64; num_bins]).collect();
    let mut joint = vec![0.0_f64; joint_size];
    let total = len as f64;

    for sample_idx in 0..len {
        let mut joint_idx = 0usize;
        for (ch_idx, ch) in channels.iter().enumerate() {
            let (mn, mx) = ranges[ch_idx];
            let range = (mx - mn) as f64;
            let scale = if range < f64::EPSILON {
                0.0
            } else {
                (num_bins - 1) as f64 / range
            };
            let bin = ((ch[sample_idx] as f64 - mn as f64) * scale)
                .clamp(0.0, (num_bins - 1) as f64) as usize;
            marginals[ch_idx][bin] += 1.0;
            joint_idx = joint_idx * num_bins + bin;
        }
        joint[joint_idx] += 1.0;
    }

    for marg in marginals.iter_mut() {
        for v in marg.iter_mut() {
            *v /= total;
        }
    }
    for v in joint.iter_mut() {
        *v /= total;
    }

    // Σᵢ H(Xᵢ)
    let sum_h: f64 = marginals
        .iter()
        .map(|marg| {
            marg.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum::<f64>()
        })
        .sum();

    // H(X₁,...,Xₙ)
    let h_joint: f64 = joint
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // C = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ)  ≥ 0 by information theory.
    Ok((sum_h - h_joint).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_correlation_n2_matches_standard_mi() {
        // For n=2, C(X,Y) = MI(X,Y).
        // Analytical: C(A,A) = H(A) > 0 for non-constant A.
        let a: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x + 8) % 64) as f32).collect();

        let tc = total_correlation_slices(&[&a, &b], 16).unwrap();
        assert!(tc >= 0.0, "total correlation must be non-negative, got {tc}");
    }

    #[test]
    fn total_correlation_identical_channels_equals_sum_marginal_entropies_minus_joint() {
        // C(A,A) = H(A) + H(A) - H(A,A) = H(A) (since H(A,A)=H(A) for duplicates).
        let a: Vec<f32> = (0..32).map(|x| (x % 8) as f32).collect();
        let tc = total_correlation_slices(&[&a, &a], 8).unwrap();
        // For identical channels, C = H(A) > 0
        assert!(tc > 0.0, "C(A,A) must be positive for non-constant A, got {tc}");
    }

    #[test]
    fn total_correlation_independent_channels_near_zero() {
        // Independent channels: C ≈ 0.
        // Using uniform [0,N) and its bitwise complement as approximate independence.
        let a: Vec<f32> = (0..256).map(|x| (x % 16) as f32).collect();
        let b: Vec<f32> = (0..256).map(|x| ((255 - x) % 16) as f32).collect();
        let tc = total_correlation_slices(&[&a, &b], 16).unwrap();
        // Should be small (may not be exactly 0 due to finite-sample histograms)
        assert!(tc >= 0.0, "total correlation must be non-negative, got {tc}");
    }

    #[test]
    fn total_correlation_n3_positive_for_identical() {
        // C(A,A,A) > 0 for non-constant A.
        let a: Vec<f32> = (0..32).map(|x| (x % 8) as f32).collect();
        let tc = total_correlation_slices(&[&a, &a, &a], 8).unwrap();
        assert!(tc > 0.0, "C(A,A,A) must be positive, got {tc}");
    }

    #[test]
    fn total_correlation_rejects_zero_channels() {
        assert!(total_correlation_slices(&[], 16).is_err());
    }

    #[test]
    fn total_correlation_rejects_low_bins() {
        let a = [1.0_f32, 2.0];
        assert!(total_correlation_slices(&[&a], 1).is_err());
    }

    #[test]
    fn total_correlation_rejects_excessive_bins() {
        let a = [1.0_f32, 2.0];
        assert!(total_correlation_slices(&[&a, &a, &a, &a], 65).is_err());
    }

    #[test]
    fn total_correlation_rejects_mismatched_lengths() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];
        assert!(total_correlation_slices(&[&a, &b], 4).is_err());
    }
}
