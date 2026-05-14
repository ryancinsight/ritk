//! Variation of Information metric between clusterings or image channels.
//!
//! # Definitions
//!
//! VI(X,Y) = H(X|Y) + H(Y|X) = H(X) + H(Y) − 2·I(X,Y)   (Meilă 2003)
//!
//! VI_n(X₁,...,Xₙ) = average pairwise VI over all C(n,2) pairs   (multivariate extension)
//!
//! VI is a proper metric (non-negativity, symmetry, triangle inequality).
//! VI = 0 iff X and Y are identical; VI = H(X)+H(Y) for independent X,Y.
//!
//! # References
//!
//! Meilă, M. (2003). Comparing clusterings by the variation of information.
//! *COLT*, pp. 173–187.

use anyhow::{bail, Result};

use super::entropy::marginal_entropy;
use super::mutual_information::mutual_information;

/// VI(X,Y) = H(X) + H(Y) − 2·I(X,Y).
///
/// Returns `max(VI, 0.0)` — negative values are numerical artefacts.
///
/// # Errors
/// Returns an error when lengths differ, inputs are empty, or `num_bins < 2`.
pub fn variation_of_information(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    let h_a = marginal_entropy(a, num_bins)?;
    let h_b = marginal_entropy(b, num_bins)?;
    let mi = mutual_information(a, b, num_bins)?;
    Ok((h_a + h_b - 2.0 * mi).max(0.0))
}

/// Average pairwise Variation of Information over n channels.
///
/// VI_n(X₁,...,Xₙ) = (2 / n(n−1)) · Σ_{i<j} VI(Xᵢ,Xⱼ)
///
/// This extends the bivariate metric to n channels by averaging all C(n,2)
/// pairwise VIs. The result lies in [0, max_pair_VI].
///
/// # Errors
///
/// Returns an error when `channels` is empty, fewer than 2 channels are provided,
/// channel lengths differ, any channel is empty, or `num_bins < 2`.
pub fn multivariate_variation_of_information(
    channels: &[&[f32]],
    num_bins: usize,
) -> Result<f64> {
    let n = channels.len();
    if n < 2 {
        bail!("at least 2 channels required, got {}", n);
    }
    let len = channels[0].len();
    if len == 0 {
        bail!("channels must not be empty");
    }
    for (i, ch) in channels.iter().enumerate() {
        if ch.len() != len {
            bail!("channel {} length {} != channel 0 length {}", i, ch.len(), len);
        }
    }
    let n_pairs = n * (n - 1) / 2;
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += variation_of_information(channels[i], channels[j], num_bins)?;
        }
    }
    Ok(sum / n_pairs as f64)
}
