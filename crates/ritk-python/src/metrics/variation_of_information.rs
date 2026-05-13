//! Variation of Information (VI) between two images.
//!
//! Delegates to `ritk_core::statistics::information::variation_of_information`.
//! See that module for the mathematical definition (Meilă 2003).

use anyhow::Result;
use ritk_core::statistics::information::variation_of_information as core_vi;

/// VI(X,Y) = H(X) + H(Y) − 2·I(X,Y).
///
/// Delegates to `ritk_core::statistics::information::variation_of_information`.
pub(super) fn variation_of_information_slices(
    a: &[f32],
    b: &[f32],
    num_bins: usize,
) -> Result<f64> {
    core_vi(a, b, num_bins)
}
