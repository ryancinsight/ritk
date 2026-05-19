//! Ensemble segmentation: consensus labeling from multiple raters.
//!
//! # Algorithms
//! - [`staple`]: STAPLE EM algorithm (Warfield et al. 2004). Estimates probabilistic
//!   ground truth from K binary segmentation masks with per-rater sensitivity/specificity.

mod staple;

pub use staple::{staple, StapleResult};

#[cfg(test)]
mod tests_staple;
