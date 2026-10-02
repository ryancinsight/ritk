//! Analyze 7.5 reader and writer for 3-D medical images.
//!
//! # Format
//!
//! Analyze 7.5 (Mayo Clinic, 1989) stores a 3-D volume as two files:
//!
//! * `<name>.hdr` — 348-byte binary header (little-endian).
//! * `<name>.img` — raw voxel values (little-endian).
//!
//! # Sample Types
//!
//! Analyze stores `u8`, `i16`, `i32`, `f32`, or `f64` samples, named by the
//! header's `datatype` code. [`read_analyze`] returns an image of the
//! caller's sample type `T` under a
//! [`Conversion`](ritk_codecs::sample::Conversion) policy and applies the
//! `funused1` scale factor in `T`; [`read_analyze_stored`] returns the stored
//! samples and the scale as a [`Rescale`](ritk_codecs::sample::Rescale)
//! (ADR 0053). [`write_analyze`] stores the image's own sample type and
//! refuses the types Analyze has no code for.
//!
//! # Axis Convention
//!
//! Analyze stores voxels with X varying fastest (column-major for [X, Y, Z]).
//! RITK stores tensors with shape `[nz, ny, nx]` (Z-major ZYX).
//! Both produce the same flat byte sequence, so no in-memory permutation
//! is required.

pub(crate) mod codec;
pub mod reader;
pub mod writer;

pub use reader::{read_analyze, read_analyze_stored, AnalyzeReader};
pub use writer::{write_analyze, AnalyzeWriter};

// Re-export datatype codes for documentation and test helpers.
pub use reader::{DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT, DT_SIGNED_SHORT, DT_UNSIGNED_CHAR};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_samples;
