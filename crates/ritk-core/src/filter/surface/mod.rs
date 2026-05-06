//! 3D surface extraction filters.
//!
//! This module provides the marching cubes isosurface extractor and the
//! `Mesh` geometry type produced by surface-extraction operations.
//!
//! # ITK / VTK parity
//! - [`MarchingCubesFilter`]: parity with `itk::BinaryMask3DMeshSource` and
//!   `vtkMarchingCubes`.
//!
//! # Usage
//! ```rust,ignore
//! use ritk_core::filter::surface::{MarchingCubesFilter, Mesh};
//!
//! let mesh: Mesh = MarchingCubesFilter::new()
//!     .with_isovalue(0.5)
//!     .with_spacing([1.0, 1.0, 1.0])
//!     .extract(&label_data, [nz, ny, nx]);
//! ```

pub mod marching_cubes;
pub mod mesh;

pub use marching_cubes::MarchingCubesFilter;
pub use mesh::{Mesh, MeshBuilder};
