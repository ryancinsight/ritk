//! VTK XML UnstructuredGrid (.vtu) reader (ASCII inline format).
//!
//! # Format Reference
//! VTK XML Format Specification v0.1, Kitware Inc.
//!
//! # Parsing Contract
//! - Finds the first `<Piece>` tag and reads `NumberOfPoints` / `NumberOfCells`.
//! - `<Points>` section: single DataArray of `n_points * 3` f32 coordinates.
//! - `<Cells>` section contains three named DataArrays:
//!   - `"connectivity"` : flat point-index list (length = sum of all cell sizes).
//!   - `"offsets"`      : cumulative cell-size sums; `offsets[i] = Σ size[0..=i]`.
//!   - `"types"`        : per-cell VTK integer type codes.
//! - Cell `i` spans `connectivity[offsets[i-1]..offsets[i]]` (`offsets[-1] = 0`).
//! - Unknown type codes are mapped to `VtkCellType::Vertex` with a `tracing::warn`.
//! - `<PointData>` and `<CellData>` are optional; absent sections yield empty maps.

mod xml_helpers;
mod parse;

pub use parse::read_vtu_unstructured_grid;
#[cfg(test)]
pub(crate) use parse::parse_vtu;

#[cfg(test)]
mod tests;
