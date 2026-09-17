//! Robust rigid fitting from bidirectional point correspondences.
//!
//! The estimator implements the rigid subset of the symmetric block-matching
//! update described by Modat et al. (2014), sections 2.1–2.3. Forward matches
//! are expressed fixed→moving and reverse matches moving→fixed. Each direction
//! is fitted independently with 50%-trimmed least squares. The reverse fit is
//! inverted, then the two fixed→moving transforms are averaged as
//! `exp((log(F) + log(B⁻¹)) / 2)` in transformation space.

mod correspondence;
mod estimate;
mod lie;
mod limits;
mod transform;
mod trimmed;

pub use correspondence::{
    FixedToMovingCorrespondence, MovingToFixedCorrespondence, SymmetricRigidFit,
};
pub use estimate::fit_symmetric_trimmed_rigid;

#[cfg(test)]
use super::error::RegistrationError;
#[cfg(test)]
use crate::types::AffineTransform;
#[cfg(test)]
use correspondence::RigidCorrespondence;
#[cfg(test)]
use correspondence::{
    discard_conflicting_endpoint_pairs, forward_correspondences, reverse_correspondences,
};
#[cfg(test)]
use transform::squared_residual;

#[cfg(test)]
#[path = "../robust_rigid_tests.rs"]
mod tests;
