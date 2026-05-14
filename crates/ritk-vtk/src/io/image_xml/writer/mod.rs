//! VTK XML ImageData (.vti) writer (ASCII inline and binary-appended formats).
//!
//! # Format Reference
//! VTK XML Format Specification v0.1, Kitware Inc.

pub mod ascii;
pub mod binary;

pub use ascii::{write_vti_image_data, write_vti_str};
pub use binary::{write_vti_binary_appended_bytes, write_vti_binary_appended_to_file};

#[cfg(test)]
mod tests;
