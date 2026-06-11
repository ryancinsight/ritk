//! Validated volume dimensions newtype.

use serde::{Deserialize, Serialize};

/// Newtype wrapping `[usize; 3]` to type-distinguish volume (image) spatial
/// dimensions from other `[usize; 3]` arrays (control-grid dims, strides, etc.).
///
/// Axis order: `[nz, ny, nx]` (Z-fastest outermost, X-fastest innermost).
///
/// # Invariants
/// All three dimensions are non-zero for a valid image (not enforced by the
/// newtype; use `total_voxels()` to detect degenerate shapes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct VolumeDims(pub [usize; 3]);

impl VolumeDims {
    /// Construct from explicit `[nz, ny, nx]` dimensions.
    #[inline]
    pub fn new(dims: [usize; 3]) -> Self {
        Self(dims)
    }

    /// Return the inner `[usize; 3]`.
    #[inline]
    pub fn as_array(self) -> [usize; 3] {
        self.0
    }

    /// Total voxel count: `nz * ny * nx`.
    #[inline]
    pub fn total_voxels(self) -> usize {
        self.0.iter().product()
    }
}

impl From<[usize; 3]> for VolumeDims {
    fn from(v: [usize; 3]) -> Self {
        Self(v)
    }
}

impl From<VolumeDims> for [usize; 3] {
    fn from(v: VolumeDims) -> Self {
        v.0
    }
}

impl std::fmt::Display for VolumeDims {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}, {}]", self.0[0], self.0[1], self.0[2])
    }
}
