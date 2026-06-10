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
}
