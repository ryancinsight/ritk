//! Canonical mesh type for surface-extraction filters.
//!
//! # SSOT
//! `Mesh` is a type alias for [`gaia::IndexedMesh<f64>`], the watertight,
//! spatially-deduplicated, indexed triangle surface mesh from the `gaia` crate.
//! Vertex welding is performed automatically at insert time via `VertexPool`
//! (spatial-hash deduplication with 1e-4 mm tolerance).
//!
//! # Invariants (enforced by gaia)
//! - Every inserted vertex is unique to within the weld tolerance.
//! - `face_count()` counts fully-inserted triangles.
//! - `is_watertight()` reports manifold + closed-surface status.

/// Canonical watertight indexed surface mesh backed by [`gaia::IndexedMesh<f64>`].
///
/// Coordinates follow the medical-imaging convention: `[x, y, z]` in physical
/// millimetres, where x = column axis, y = row axis, z = slice axis.
pub type Mesh = gaia::IndexedMesh<f64>;

/// Re-export `MeshBuilder` for consumers that need low-level triangle construction.
pub use gaia::MeshBuilder;
