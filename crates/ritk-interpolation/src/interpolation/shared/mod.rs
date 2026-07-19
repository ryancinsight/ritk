//! Shared utilities for interpolation implementations.
//!
//! Provides zero-cost helpers consumed by both linear and nearest-neighbor
//! interpolation paths, eliminating duplicated clone-and-compare patterns.

pub mod in_bounds;

pub use in_bounds::{compute_oob_mask, OutOfBoundsMode};
