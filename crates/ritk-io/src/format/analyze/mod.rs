//! Analyze 7.5 medical image format support.
//!
//! The Analyze 7.5 format (Mayo Clinic, 1989) consists of a `.hdr` header file
//! (348 bytes, little-endian) paired with a `.img` raw data file.
//!
//! # Example
//!
//! ```rust,ignore
//! use ritk_io::format::analyze::{read_analyze, write_analyze};
//!
//! // Read an Analyze image pair (.hdr + .img)
//! let image = read_analyze::<NdArray<f32>, _>("brain.hdr", &device)?;
//!
//! // Write an image as Analyze format
//! write_analyze("output.hdr", &image)?;
//! ```

pub mod reader;
pub mod writer;

pub use reader::read_analyze;
pub use writer::write_analyze;
