//! STL mesh I/O for VtkPolyData.
//!
//! Supports both ASCII and binary STL (little-endian) on read.
//! Writes binary (compact, deterministic) and ASCII (human-readable) STL.
//!
//! STL has no shared-vertex topology. Each triangle carries three dedicated
//! point entries. `VtkPolyData::polygons` stores triangles only; cell_data
//! `"Normals"` stores per-facet normals.

pub mod reader;
pub mod writer;

pub use reader::read_stl_mesh;
pub use writer::{write_stl_ascii, write_stl_binary};

#[cfg(test)]
#[path = "tests_stl.rs"]
mod tests;
