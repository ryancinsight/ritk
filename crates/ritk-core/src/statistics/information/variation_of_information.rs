//! Variation of Information metric between two images.
//!
//! # Definition (Meilă 2003)
//!
//! VI(X,Y) = H(X|Y) + H(Y|X) = H(X) + H(Y) − 2·I(X,Y)
//!
//! VI is a proper metric (non-negativity, symmetry, triangle inequality).
//! VI = 0 iff X and Y are identical; VI = H(X)+H(Y) for independent X,Y.
//!
//! # Reference
//!
//! Meilă, M. (2003). Comparing clusterings by the variation of information.
//! *COLT*, pp. 173–187.

use anyhow::Result;

use super::mutual_information::mutual_information;
use super::entropy::marginal_entropy;

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
