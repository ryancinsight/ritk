//! Coeus-typed image I/O contract (ADR 0002, cutover step 2).
//!
//! The parallel family to the Burn-typed [`super::ImageReader`]/
//! [`super::ImageWriter`]: the same reader/writer role interfaces, over
//! [`ritk_image::coeus::Image`]. Per-format crates implement these behind the
//! `coeus` feature as they gain Coeus paths; Burn traits and implementations
//! are untouched. Consumers (`ritk-cli`/`ritk-python`) switch to this contract
//! in the ADR 0002 cutover, after which the Burn contract is removed —
//! parallel-then-cutover, never a shim.

use coeus_core::{ComputeBackend, Scalar};
use ritk_image::coeus::Image;
use std::path::Path;

/// Read a Coeus-backed image from a path (Coeus counterpart of
/// [`super::ImageReader`]).
pub trait CoeusImageReader<T: Scalar, B: ComputeBackend, const D: usize> {
    /// Read an image natively from a path, preserving spatial metadata.
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<T, B, D>>;
}

/// Write a Coeus-backed image to a path (Coeus counterpart of
/// [`super::ImageWriter`]).
pub trait CoeusImageWriter<T: Scalar, B: ComputeBackend, const D: usize> {
    /// Write an image to disk without lossy approximation of voxels or
    /// spatial metadata beyond the format's own representation limits.
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<T, B, D>) -> std::io::Result<()>;
}

/// Map a format crate's `anyhow` error onto the contract's `std::io::Error`
/// (shared by every per-format implementor; one mapping, not N copies).
pub(crate) fn to_io_err(e: anyhow::Error) -> std::io::Error {
    std::io::Error::other(e.to_string())
}
