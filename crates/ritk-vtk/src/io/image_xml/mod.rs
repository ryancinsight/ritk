//! VTK XML ImageData (.vti) format support.
//!
//! Provides ASCII-inline and binary-appended VTI reader and writer.
//! Format reference: VTK XML Format Specification v0.1, Kitware Inc.

pub mod reader;
pub mod writer;

pub use reader::{read_vti_binary_appended, read_vti_binary_appended_bytes, read_vti_image_data};
pub use writer::{
    write_vti_binary_appended_bytes, write_vti_binary_appended_to_file, write_vti_image_data,
    write_vti_str,
};
