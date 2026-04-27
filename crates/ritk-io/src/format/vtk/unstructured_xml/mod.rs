//! VTK XML UnstructuredGrid (.vtu) format support.
//!
//! Provides ASCII-inline VTU reader and writer.
//! Format reference: VTK XML Format Specification, Kitware Inc.

pub mod reader;
pub mod writer;

pub use reader::read_vtu_unstructured_grid;
pub use writer::{write_vtu_str, write_vtu_unstructured_grid};
