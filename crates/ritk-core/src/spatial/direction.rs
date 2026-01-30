//! Direction type for representing image orientation.
//!
//! Direction matrices represent orientation of image axes in physical space.

use nalgebra::SMatrix;
use super::Vector;

/// Direction matrix representing image orientation.
///
/// The direction matrix is a D×D matrix where each column represents
/// direction of the corresponding image axis in physical space.
/// Column i represents the direction of the i-th image axis.
///
/// This is a thin wrapper around nalgebra's SMatrix to provide
/// domain-specific functionality while maintaining all nalgebra operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direction<const D: usize>(pub SMatrix<f64, D, D>);

impl<const D: usize> Direction<D> {
    /// Create an identity direction matrix (no rotation).
    pub fn identity() -> Self {
        Self(SMatrix::identity())
    }

    /// Create a zero matrix.
    pub fn zeros() -> Self {
        Self(SMatrix::zeros())
    }

    /// Check if direction matrix is orthogonal (rotation matrix).
    pub fn is_orthogonal(&self) -> bool {
        let product = self.0 * self.0.transpose();
        let identity = Self::identity();
        (0..D).all(|i| {
            (0..D).all(|j| {
                (product[(i, j)] - identity.0[(i, j)]).abs() < 1e-6
            })
        })
    }

    /// Check if direction matrix is a proper rotation (det = 1).
    pub fn is_proper_rotation(&self) -> bool {
        self.is_orthogonal() && (self.determinant() - 1.0).abs() < 1e-6
    }

    /// Compute the determinant of the direction matrix.
    pub fn determinant(&self) -> f64 {
        // For 2x2 matrix
        if D == 2 {
            self.0[(0, 0)] * self.0[(1, 1)] - self.0[(0, 1)] * self.0[(1, 0)]
        }
        // For 3x3 matrix
        else if D == 3 {
            let a = self.0[(0, 0)];
            let b = self.0[(0, 1)];
            let c = self.0[(0, 2)];
            let d = self.0[(1, 0)];
            let e = self.0[(1, 1)];
            let f = self.0[(1, 2)];
            let g = self.0[(2, 0)];
            let h = self.0[(2, 1)];
            let i = self.0[(2, 2)];
            
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        }
        // For other dimensions, return 0.0 (not supported)
        else {
            0.0
        }
    }

    /// Try to compute the inverse of the direction matrix.
    pub fn try_inverse(&self) -> Option<Self> {
        self.0.try_inverse().map(Self)
    }

    /// Get the axis directions as vectors.
    pub fn axis_directions(&self) -> Vec<Vector<D>> {
        (0..D).map(|i| {
            let mut v = Vector::zeros();
            for j in 0..D {
                v[j] = self.0[(j, i)];
            }
            v
        }).collect()
    }

    /// Get the inner nalgebra matrix.
    pub fn inner(&self) -> &SMatrix<f64, D, D> {
        &self.0
    }

    /// Get mutable reference to inner nalgebra matrix.
    pub fn inner_mut(&mut self) -> &mut SMatrix<f64, D, D> {
        &mut self.0
    }
}

impl<const D: usize> std::ops::Index<(usize, usize)> for Direction<D> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> std::ops::IndexMut<(usize, usize)> for Direction<D> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize> std::ops::Mul for Direction<D> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}

impl<const D: usize> std::ops::Mul<Vector<D>> for Direction<D> {
    type Output = Vector<D>;

    fn mul(self, vector: Vector<D>) -> Self::Output {
        Vector(self.0 * vector.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases for testing
    type Direction3 = Direction<3>;
    type Vector3 = Vector<3>;

    #[test]
    fn test_direction_identity() {
        let d = Direction3::identity();
        assert_eq!(d[(0, 0)], 1.0);
        assert_eq!(d[(1, 1)], 1.0);
        assert_eq!(d[(2, 2)], 1.0);
    }

    #[test]
    fn test_direction_is_orthogonal() {
        let identity = Direction3::identity();
        assert!(identity.is_orthogonal());

        // Create a simple rotation matrix (90 degrees around Z)
        let mut rot = Direction3::zeros();
        rot[(0, 0)] = 0.0;
        rot[(0, 1)] = -1.0;
        rot[(0, 2)] = 0.0;
        rot[(1, 0)] = 1.0;
        rot[(1, 1)] = 0.0;
        rot[(1, 2)] = 0.0;
        rot[(2, 0)] = 0.0;
        rot[(2, 1)] = 0.0;
        rot[(2, 2)] = 1.0;
        assert!(rot.is_orthogonal());
    }

    #[test]
    fn test_direction_is_proper_rotation() {
        let identity = Direction3::identity();
        assert!(identity.is_proper_rotation());

        // Create a reflection matrix (det = -1)
        let mut reflection = Direction3::identity();
        reflection[(0, 0)] = -1.0;
        assert!(!reflection.is_proper_rotation());
    }

    #[test]
    fn test_direction_axis_directions() {
        let identity = Direction3::identity();
        let axes = identity.axis_directions();
        assert_eq!(axes.len(), 3);
        assert_eq!(axes[0], Vector3::new([1.0, 0.0, 0.0]));
        assert_eq!(axes[1], Vector3::new([0.0, 1.0, 0.0]));
        assert_eq!(axes[2], Vector3::new([0.0, 0.0, 1.0]));
    }
}
