//! Analyze 7.5 medical image format support.
//!
//! This module provides read/write access to Analyze 7.5 image pairs
//! (`.hdr` + `.img`) through the `read_analyze` and `write_analyze` APIs.
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

pub use reader::{read_analyze, AnalyzeReader};
pub use writer::{write_analyze, AnalyzeWriter};
