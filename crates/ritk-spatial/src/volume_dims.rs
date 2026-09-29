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

    /// Total voxel count, or `None` when `nz * ny * nx` overflows `usize`.
    ///
    /// The checked counterpart of [`Self::total_voxels`], for header fields that
    /// arrive from a file and may name an impossible shape. Callers keep their
    /// own error type and message so the diagnostic names the format and field.
    #[inline]
    pub fn checked_total_voxels(self) -> Option<usize> {
        self.0.into_iter().try_fold(1usize, usize::checked_mul)
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

#[cfg(test)]
mod tests {
    use super::VolumeDims;

    #[test]
    fn checked_total_voxels_multiplies_dimensions() {
        assert_eq!(VolumeDims::new([2, 3, 4]).checked_total_voxels(), Some(24));
    }

    #[test]
    fn checked_total_voxels_rejects_overflow() {
        assert_eq!(
            VolumeDims::new([usize::MAX, 2, 1]).checked_total_voxels(),
            None
        );
    }
}
