//! Analyze 7.5 reader and writer for 3-D medical images.
//!
//! # Format
//!
//! Analyze 7.5 (Mayo Clinic, 1989) stores a 3-D volume as two files:
//!
//! * `<name>.hdr` — 348-byte binary header (little-endian).
//! * `<name>.img` — raw voxel values (little-endian).
//!
//! # Axis Convention
//!
//! Analyze stores voxels with X varying fastest (column-major for [X, Y, Z]).
//! RITK stores tensors with shape `[nz, ny, nx]` (Z-major ZYX).
//! Both produce the same flat byte sequence, so no in-memory permutation
//! is required.

pub mod reader;
pub mod writer;

pub use reader::{read_analyze, AnalyzeReader};
pub use writer::{write_analyze, AnalyzeWriter};

// Re-export datatype codes for documentation and test helpers.
pub use reader::{DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT, DT_SIGNED_SHORT, DT_UNSIGNED_CHAR};

#[cfg(test)]
mod tests;
