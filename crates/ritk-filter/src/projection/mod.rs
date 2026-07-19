//! Intensity projection filters for 3-D images.
//!
//! # Shape convention
//! All operations follow the RITK convention: `shape = [Z, Y, X]`
//! (axis 0 = Z, axis 1 = Y, axis 2 = X). The projected axis is collapsed
//! to size 1 in the output while all other axes are unchanged.
//!
//! # Precision
//! Max and min accumulation uses native `f32`. Mean, sum, and std-dev
//! accumulation uses `f64` to prevent catastrophic cancellation across
//! large slabs. Median follows [`f32::total_cmp`], giving NaNs, infinities,
//! and signed zero a deterministic order without panicking.
//!
//! # Memory
//! Projection inputs use the image's canonical copy-on-write host view.
//! Contiguous Coeus storage is borrowed without allocation; a non-contiguous
//! view is materialized once in logical row-major order.
//!
//! # Parallelization
//! Each filter parallelises over the output pixels (the non-collapsed axes)
//! using Moirai indexed collection. The inner reduction over the collapsed axis
//! is a sequential `fold` per output pixel.

mod filters;
mod ops;

pub use filters::{
    BinaryProjectionFilter, BinaryThresholdProjectionFilter, MaxIntensityProjectionFilter,
    MeanIntensityProjectionFilter, MedianIntensityProjectionFilter, MinIntensityProjectionFilter,
    StdDevIntensityProjectionFilter, SumIntensityProjectionFilter,
};

/// Axis along which the projection is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionAxis {
    /// Project along axis 0 (Z): output shape `[1, Y, X]`.
    Z = 0,
    /// Project along axis 1 (Y): output shape `[Z, 1, X]`.
    Y = 1,
    /// Project along axis 2 (X): output shape `[Z, Y, 1]`.
    X = 2,
}

#[cfg(test)]
#[path = "../tests_projection.rs"]
mod tests;
