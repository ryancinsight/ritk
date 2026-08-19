//! Dispatch manifest for B-spline displacement evaluation.
//!
//! The public evaluation surface is partitioned by operation family while
//! this module remains the stable internal path used by registration and its
//! tests. Dense support tables and cache-based sparse evaluation therefore
//! share one re-export surface without duplicating an algorithm.
//!
//! Grid selection, dense support construction, and sparse cache evaluation
//! live in [`grid`], [`dense`], and [`sparse`], respectively.

#[path = "dense.rs"]
mod dense;
#[path = "grid.rs"]
mod grid;
#[path = "sparse.rs"]
mod sparse;

pub use dense::{
    evaluate_bspline_displacement_dense_into, evaluate_bspline_displacement_dense_with,
    DenseSupport,
};
pub use grid::{init_control_grid, should_use_dense_path, DENSE_LATTICE_CUTOFF};
pub use sparse::{
    evaluate_bspline_displacement, evaluate_bspline_displacement_fast,
    evaluate_bspline_displacement_fast_into,
};
