//! Validated volume dimensions newtype.

/// Newtype wrapping `[usize; 3]` to distinguish volume (image) spatial
/// dimensions from control-grid dimensions at the type level.
///
/// Axis order: `[nz, ny, nx]` (Z-fastest outermost, X-fastest innermost).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VolumeDims(pub [usize; 3]);

impl VolumeDims {
    /// Create from explicit `[nz, ny, nx]` dimensions.
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
