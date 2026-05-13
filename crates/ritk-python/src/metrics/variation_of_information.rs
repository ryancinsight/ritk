//! Variation of Information (VI) between two images.
//!
//! # Definition (Meilă 2003)
//!
//! ```text
//! VI(X, Y) = H(X|Y) + H(Y|X)
//!          = H(X) + H(Y) − 2·I(X, Y)
//! ```
//!
//! VI is a proper metric on the space of partitions/probability distributions:
//! it satisfies non-negativity, symmetry, and the triangle inequality.
//! For identical images, VI = 0; for independent images, VI = H(X) + H(Y).
//!
//! # Reference
//!
//! Meilă, M. (2003). Comparing clusterings by the variation of information.
//! In *COLT*, pp. 173–187. Springer, Berlin.

use anyhow::{bail, Result};

use super::mi::min_max;

/// Variation of Information VI(X,Y) = H(X) + H(Y) − 2·I(X,Y).
///
/// Uses nearest-bin hard-assignment histogram estimation.
///
/// # Arguments
/// - `a`, `b`: equal-length f32 slices.
/// - `num_bins`: histogram bins per channel (≥ 2).
///
/// # Returns
/// `VI ≥ 0`; exactly 0 iff the two distributions are identical.
pub(super) fn variation_of_information_slices(
    a: &[f32],
    b: &[f32],
    num_bins: usize,
) -> Result<f64> {
    if a.len() != b.len() {
        bail!(
            "channel lengths differ: {} vs {}",
            a.len(),
            b.len()
        );
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    if num_bins < 2 {
        bail!("num_bins must be ≥ 2, got {}", num_bins);
    }

    let n = a.len();
    let (a_min, a_max) = min_max(a);
    let (b_min, b_max) = min_max(b);
    let a_range = (a_max - a_min) as f64;
    let b_range = (b_max - b_min) as f64;
    let scale_a = if a_range < f64::EPSILON {
        0.0
    } else {
        (num_bins - 1) as f64 / a_range
    };
    let scale_b = if b_range < f64::EPSILON {
        0.0
    } else {
        (num_bins - 1) as f64 / b_range
    };

    let mut joint = vec![0.0_f64; num_bins * num_bins];
    let mut hist_a = vec![0.0_f64; num_bins];
    let mut hist_b = vec![0.0_f64; num_bins];

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let ia = ((ai as f64 - a_min as f64) * scale_a)
            .clamp(0.0, (num_bins - 1) as f64) as usize;
        let ib = ((bi as f64 - b_min as f64) * scale_b)
            .clamp(0.0, (num_bins - 1) as f64) as usize;
        joint[ia * num_bins + ib] += 1.0;
        hist_a[ia] += 1.0;
        hist_b[ib] += 1.0;
    }

    let total = n as f64;
    for v in joint.iter_mut() {
        *v /= total;
    }
    for v in hist_a.iter_mut() {
        *v /= total;
    }
    for v in hist_b.iter_mut() {
        *v /= total;
    }

    let h_a: f64 = hist_a.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    let h_b: f64 = hist_b.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    let h_ab: f64 = joint.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    let mi = h_a + h_b - h_ab;

    // VI = H(X) + H(Y) − 2·I(X,Y)  ≥ 0 since I(X,Y) ≤ min(H(X),H(Y)).
    Ok((h_a + h_b - 2.0 * mi).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vi_identical_images_is_zero() {
        // VI(X,X) = H(X)+H(X)−2·I(X,X) = H(X)+H(X)−2·H(X) = 0.
        let a: Vec<f32> = (0..32).map(|x| (x % 8) as f32).collect();
        let vi = variation_of_information_slices(&a, &a, 8).unwrap();
        assert!(vi.abs() < 1e-10, "VI(X,X) must be 0, got {vi}");
    }

    #[test]
    fn vi_is_non_negative() {
        let a: Vec<f32> = (0..64).map(|x| (x % 16) as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x + 4) % 16) as f32).collect();
        let vi = variation_of_information_slices(&a, &b, 16).unwrap();
        assert!(vi >= 0.0, "VI must be non-negative, got {vi}");
    }

    #[test]
    fn vi_is_symmetric() {
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x * 3 + 1) % 8) as f32).collect();
        let vi_ab = variation_of_information_slices(&a, &b, 8).unwrap();
        let vi_ba = variation_of_information_slices(&b, &a, 8).unwrap();
        assert!(
            (vi_ab - vi_ba).abs() < 1e-12,
            "VI must be symmetric: VI(a,b)={vi_ab} != VI(b,a)={vi_ba}"
        );
    }

    #[test]
    fn vi_constant_vs_varying_equals_entropy_of_varying() {
        // I(A, constant) = 0 since H(constant)=0.
        // VI(A, constant) = H(A) + 0 − 2·0 = H(A).
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let b_const: Vec<f32> = vec![3.0_f32; 64];
        let vi = variation_of_information_slices(&a, &b_const, 8).unwrap();

        // H(A) with 8 equally likely values = ln(8)
        let h_a = (8.0_f64).ln();
        assert!(
            (vi - h_a).abs() < 0.05,
            "VI(A, constant) must ≈ H(A)={h_a:.4}, got {vi:.4}"
        );
    }

    #[test]
    fn vi_rejects_empty_inputs() {
        assert!(variation_of_information_slices(&[], &[], 8).is_err());
    }

    #[test]
    fn vi_rejects_mismatched_lengths() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];
        assert!(variation_of_information_slices(&a, &b, 4).is_err());
    }

    #[test]
    fn vi_rejects_too_few_bins() {
        let a = [1.0_f32, 2.0];
        assert!(variation_of_information_slices(&a, &a, 1).is_err());
    }
}
