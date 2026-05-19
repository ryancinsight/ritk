//! PLY mesh I/O for VtkPolyData.
//!
//! Supported formats: `ascii 1.0` and `binary_little_endian 1.0`.
//! Big-endian PLY returns `Err` on read.
//!
//! Vertex properties recognised: `x`, `y`, `z` (required),
//! `nx`, `ny`, `nz` (optional → `point_data["Normals"]`).
//! Face property: `property list <count_type> <index_type> vertex_indices`.
//! Property types: `char`, `uchar`, `short`, `ushort`, `int`, `uint`,
//! `float`, `double` (and their sized aliases `int8`…`float64`).

pub mod reader;
pub mod writer;

pub use reader::read_ply_mesh;
pub use writer::{write_ply_ascii, write_ply_binary_le};

#[cfg(test)]
#[path = "tests_ply.rs"]
mod tests;
