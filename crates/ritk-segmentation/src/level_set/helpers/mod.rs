//! Single source of truth (SSOT) for level-set numerical primitives.

pub mod indexing;
pub mod math;
pub mod ops;

pub(crate) use indexing::idx_clamped;
pub(crate) use math::{
    compute_edge_stopping, gaussian_smooth, regularised_dirac, regularised_heaviside,
    smooth_or_borrow,
};
pub(crate) use ops::{
    compute_curvature_into, compute_field_gradient, compute_field_gradient_into,
    compute_gradient_magnitude, evolve_slices_with_metric, upwind_advection_into,
};

#[cfg(test)]
#[path = "../tests_helpers.rs"]
mod tests;
