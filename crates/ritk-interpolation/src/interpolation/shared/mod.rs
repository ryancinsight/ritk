//! Shared utilities for interpolation implementations.
//!
//! Provides zero-cost helpers consumed by the linear, nearest-neighbor, sinc
//! and B-spline interpolation paths, eliminating duplicated clone-and-compare
//! patterns and per-kernel index math.

pub mod in_bounds;
pub(crate) mod indexing;

pub use in_bounds::{compute_oob_mask, OutOfBoundsMode};
pub(crate) use indexing::{clamp_index, compute_strides};
