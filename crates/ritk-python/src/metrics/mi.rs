//! Histogram-based Mutual Information variants.
//!
//! Delegates all three variants to `ritk_core::statistics::information`:
//! - "standard":   hard nearest-bin assignment — `mutual_information`
//! - "normalized": symmetric uncertainty 2·I/(H(A)+H(B)) ∈ [0,1] — `symmetric_uncertainty`
//! - "mattes":     bilinear soft-binning (Mattes 2003) — `mutual_information_mattes`

use anyhow::Result;
use ritk_core::statistics::information::{
    mutual_information as core_mi, mutual_information_mattes as core_mi_mattes,
    symmetric_uncertainty as core_su,
};

/// Histogram-based MI with configurable binning strategy.
///
/// `variant` must be one of `"mattes"`, `"standard"`, `"normalized"`.
pub(super) fn mi_slices(a: &[f32], b: &[f32], num_bins: usize, variant: &str) -> Result<f64> {
    match variant {
        "mattes" => core_mi_mattes(a, b, num_bins),
        "standard" => core_mi(a, b, num_bins),
        "normalized" => core_su(a, b, num_bins),
        _ => unreachable!("variant validated before mi_slices"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi_self_exceeds_constant() {
        // Analytical: MI(A,A) = H(A) > 0 for non-constant A.
        // MI(A, constant) = 0 since H(constant) = 0.
        let a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let b_const: Vec<f32> = vec![5.0_f32; 32];
        let mi_self = mi_slices(&a, &a, 16, "standard").unwrap();
        let mi_const = mi_slices(&a, &b_const, 16, "standard").unwrap();
        assert!(mi_self > 0.0, "MI(A,A) must be positive for non-constant A, got {mi_self}");
        assert!(
            mi_const.abs() < 1e-10,
            "MI(A,constant) must be 0, got {mi_const}"
        );
    }

    #[test]
    fn mi_normalized_variant_in_zero_one() {
        // SU = 2·I/(H(A)+H(B)) ∈ [0,1].
        let a: Vec<f32> = (0..64).map(|x| (x % 16) as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x + 4) % 16) as f32).collect();
        let su = mi_slices(&a, &b, 16, "normalized").unwrap();
        assert!(
            (0.0..=1.0).contains(&su),
            "symmetric uncertainty must be in [0,1], got {su}"
        );
    }

    #[test]
    fn mi_normalized_identical_is_one() {
        // SU(X,X) = 2·H(X)/(H(X)+H(X)) = 1.0.
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let su = mi_slices(&a, &a, 16, "normalized").unwrap();
        assert!(
            (su - 1.0).abs() < 1e-9,
            "SU(X,X) must equal 1.0, got {su}"
        );
    }

    #[test]
    fn mi_mattes_self_exceeds_zero() {
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let mi = mi_slices(&a, &a, 16, "mattes").unwrap();
        assert!(mi > 0.0, "Mattes MI(A,A) must be positive, got {mi}");
    }
}
