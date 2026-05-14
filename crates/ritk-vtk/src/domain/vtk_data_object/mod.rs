//! VTK data-object type hierarchy.
//!
//! # Mathematical Specification
//!
//! ## VtkPolyData
//! A polygonal mesh M = (V, C, A_P, A_C) where:
//! - V ⊂ ℝ³: ordered point set, |V| = n_points
//! - C: connectivity table partitioned into {vertices, lines, polygons, strips}
//!   Each cell c_i is a sequence of point indices from [0, n_points).
//! - A_P: named per-point attribute arrays, each of length n_points × ncomp.
//! - A_C: named per-cell attribute arrays, each of length n_cells × ncomp.
//!
//! ## Invariants
//! - All point indices in C are in [0, n_points).
//! - Attribute array lengths equal n_points × ncomp (point data) or
//!   n_cells × ncomp (cell data), where n_cells = |vertices|+|lines|+|polygons|+|strips|.
//!
//! # Reference
//! VTK File Formats (legacy), sections 4.1-4.6, Kitware Inc.

mod attribute;
mod poly_data;
mod data_object;
mod structured_grid;
mod cell_type;
mod unstructured_grid;
mod image_data;

pub use attribute::AttributeArray;
pub use poly_data::VtkPolyData;
pub use data_object::VtkDataObject;
pub use structured_grid::VtkStructuredGrid;
pub use cell_type::VtkCellType;
pub use unstructured_grid::VtkUnstructuredGrid;
pub use image_data::VtkImageData;

#[cfg(test)]
mod tests;
