//! Multivariate Variation of Information.
//!
//! Delegates to `ritk_core::statistics::information::multivariate_variation_of_information`.
//! VI_n(X₁,...,Xₙ) = (2 / n(n−1)) · Σ_{i<j} VI(Xᵢ,Xⱼ)   (average pairwise VI)

use anyhow::Result;
use ritk_core::statistics::information::multivariate_variation_of_information as core_mvi;

/// Average pairwise VI over n image channels.
///
/// Requires `channels.len() ≥ 2`; all slices must have equal length.
pub(super) fn multivariate_vi_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_mvi(channels, num_bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvi_identical_slices_is_zero() {
        let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
        let mvi =
            multivariate_vi_slices(&[x.as_slice(), x.as_slice(), x.as_slice()], 8).unwrap();
        assert!(mvi.abs() < 1e-9, "MVI(X,X,X)={mvi:.10} must be 0");
    }
}
