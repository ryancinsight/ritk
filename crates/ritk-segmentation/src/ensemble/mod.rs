//! Ensemble segmentation: consensus labeling from multiple raters.
//!
//! # Algorithms
//! - [`staple`]: STAPLE EM algorithm (Warfield et al. 2004). Estimates probabilistic
//!   ground truth from K binary segmentation masks with per-rater sensitivity/specificity.
//! - [`multi_label_staple`]: multi-label generalization emitting a hard consensus
//!   label map from K integer label maps.

mod multi_label_staple;
mod staple;

pub use multi_label_staple::{multi_label_staple, MultiLabelStapleResult};
pub use staple::{staple, StapleConvergence, StapleResult};

#[cfg(test)]
mod tests_staple;
