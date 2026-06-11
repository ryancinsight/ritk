//! Newtype for B-spline FFD control-grid dimensions.

/// Strongly-typed control-grid dimensions `[nz, ny, nx]` for a B-spline FFD.
///
/// Prevents accidental mixing of voxel-image dimensions (`VolumeDims`) and
/// control-grid dimensions that differ by ~3 voxels on each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlGridDims(pub [usize; 3]);

impl ControlGridDims {
    /// Construct from a `[nz, ny, nx]` array.
    #[inline]
    pub fn new(dims: [usize; 3]) -> Self {
        Self(dims)
    }

    /// Return the underlying array.
    #[inline]
    pub fn as_array(self) -> [usize; 3] {
        self.0
    }

    /// Total number of control-grid nodes (nz × ny × nx).
    #[inline]
    pub fn num_nodes(self) -> usize {
        self.0[0] * self.0[1] * self.0[2]
    }
}

impl From<[usize; 3]> for ControlGridDims {
    fn from(v: [usize; 3]) -> Self {
        Self(v)
    }
}

impl From<ControlGridDims> for [usize; 3] {
    fn from(v: ControlGridDims) -> Self {
        v.0
    }
}
