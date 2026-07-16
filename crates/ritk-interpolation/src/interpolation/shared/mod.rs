//! Shared utilities for interpolation implementations.

pub mod oob_mask;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutOfBoundsMode {
    Clamp,
    ZeroPad,
}

pub use oob_mask::compute_oob_mask;
