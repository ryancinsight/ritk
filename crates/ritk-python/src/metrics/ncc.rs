//! NCC (Normalized Cross-Correlation) slice-level implementation.
//!
//! # Formula
//! NCC = Σ(aᵢ−ā)(bᵢ−b̄) / (N·σ_a·σ_b + ε)

use anyhow::{bail, Result};

/// Pearson r = cov(a,b) / (std_a · std_b + ε).
pub(super) fn ncc_slices(a: &[f32], b: &[f32]) -> Result<f64> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        bail!("cannot compute NCC of empty images");
    }
    let n_f = n as f64;
    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n_f;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n_f;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai as f64 - mean_a;
        let db = bi as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let std_a = (var_a / n_f).sqrt();
    let std_b = (var_b / n_f).sqrt();
    const EPS: f64 = 1e-10;
    Ok(cov / (n_f * (std_a * std_b + EPS)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncc_identical_images_returns_one() {
        let v: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let result = ncc_slices(&v, &v).unwrap();
        assert!((result - 1.0).abs() < 1e-10, "NCC of identical must be 1.0, got {result}");
    }

    #[test]
    fn ncc_anti_correlated_returns_negative_one() {
        let a: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=8).rev().map(|x| x as f32).collect();
        let result = ncc_slices(&a, &b).unwrap();
        assert!((result + 1.0).abs() < 1e-10, "NCC of anti-correlated must be −1, got {result}");
    }

    #[test]
    fn ncc_empty_errors() {
        assert!(ncc_slices(&[], &[]).is_err());
    }
}
