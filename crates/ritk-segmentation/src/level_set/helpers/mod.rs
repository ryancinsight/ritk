//! Single source of truth (SSOT) for level-set numerical primitives.

pub mod evolve;
pub mod indexing;
pub mod math;
pub mod ops;

pub(crate) use evolve::{
    binary_mask, checked_dims, evolve_to_convergence, MaxAbsRate, RootMeanSquare,
};
pub(crate) use indexing::idx_clamped;
pub(crate) use math::{
    edge_stopping_fields, regularised_dirac, regularised_heaviside, smooth_or_borrow,
};
pub(crate) use ops::{compute_curvature_into, evolve_slices_with_metric, upwind_advection_into};

#[cfg(test)]
#[path = "../tests_helpers.rs"]
mod tests;
