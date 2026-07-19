//! Voxel neighbourhood connectivity for morphological operations.

use serde::{Deserialize, Serialize};

/// Voxel neighbourhood connectivity for morphological operations.
///
/// Determines which neighbours are considered when evaluating voxel adjacency:
///
/// - `Face6`: face-adjacent neighbours only (6-connected in 3D, 4-connected in 2D).
///   This is the ITK default for contour filters.
/// - `Vertex26`: face + edge + vertex neighbours (26-connected in 3D, 8-connected in 2D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Connectivity {
    /// Face-adjacent neighbours only (6-connected in 3D, 4-connected in 2D).
    #[default]
    Face6,
    /// Face + edge + vertex neighbours (26-connected in 3D, 8-connected in 2D).
    Vertex26,
}

impl Connectivity {
    /// Returns `true` if this is [`Connectivity::Vertex26`].
    ///
    /// Backward-compatible accessor for the former `fully_connected: bool` field.
    pub fn fully_connected(&self) -> bool {
        matches!(self, Self::Vertex26)
    }

    /// Whether the neighbour at offset `(dz, dy, dx)` (each in `-1..=1`) lies
    /// inside this connectivity's structuring element.
    ///
    /// Face connectivity (`Face6`) admits only axis-aligned steps (Manhattan
    /// distance ≤ 1); vertex connectivity (`Vertex26`) admits the whole 3×3×3
    /// neighbourhood.
    #[inline]
    pub fn includes(self, dz: i32, dy: i32, dx: i32) -> bool {
        match self {
            Self::Face6 => dz.abs() + dy.abs() + dx.abs() <= 1,
            Self::Vertex26 => true,
        }
    }
}
