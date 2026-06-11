//! `AffineTransform` — row-major homogeneous 4×4 transformation matrix newtype.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Row-major homogeneous 4×4 affine transformation matrix stored as `[f64; 16]`.
///
/// Layout: `[m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33]`
/// where `m_ij` is row `i`, column `j`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct AffineTransform(pub [f64; 16]);

impl AffineTransform {
    /// 4×4 identity matrix.
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    /// Construct from a raw `[f64; 16]` array.
    pub fn new(matrix: [f64; 16]) -> Self {
        Self(matrix)
    }

    /// Access the underlying `[f64; 16]` array by reference.
    pub fn as_array(&self) -> &[f64; 16] {
        &self.0
    }

    /// Access the underlying `[f64; 16]` array by mutable reference.
    pub fn as_array_mut(&mut self) -> &mut [f64; 16] {
        &mut self.0
    }
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::ops::Index<(usize, usize)> for AffineTransform {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(
            row < 4 && col < 4,
            "AffineTransform index out of bounds: ({row}, {col})"
        );
        &self.0[row * 4 + col]
    }
}

impl AsRef<[f64; 16]> for AffineTransform {
    fn as_ref(&self) -> &[f64; 16] {
        &self.0
    }
}

impl AsMut<[f64; 16]> for AffineTransform {
    fn as_mut(&mut self) -> &mut [f64; 16] {
        &mut self.0
    }
}
