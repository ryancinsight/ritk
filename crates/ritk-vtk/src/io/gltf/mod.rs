//! glTF 2.0 mesh writer for VtkPolyData.
//!
//! Produces a single `.gltf` JSON file (no sidecar `.bin`).
//! Geometry data is embedded as a base64 data URI.
//!
//! Polygons are fan-triangulated before export.
//! Only `VtkPolyData::points` and `VtkPolyData::polygons` are exported.

pub mod writer;

pub use writer::write_gltf;
