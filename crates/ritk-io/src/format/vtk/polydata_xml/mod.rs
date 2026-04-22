//! VTK XML PolyData (.vtp) format support.
//!
//! Provides ASCII-inline VTP reader and writer.
//! Format reference: VTK XML Format Specification v0.1, Kitware Inc.

pub mod reader;
pub mod writer;

pub use reader::read_vtp_polydata;
pub use writer::write_vtp_polydata;
