//! OBJ (Wavefront) mesh I/O for VtkPolyData.
//!
//! Supports ASCII OBJ: vertices (`v`), vertex normals (`vn`), and polygon
//! faces (`f`) with the `v`, `v/t`, `v/t/n`, and `v//n` face-vertex formats.
//! All other directives are skipped silently.

pub mod reader;
pub mod writer;

pub use reader::read_obj_mesh;
pub use writer::write_obj_mesh;

#[cfg(test)]
#[path = "tests_obj.rs"]
mod tests;
