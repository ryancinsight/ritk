//! Vector type for representing spatial displacements and directions.
//!
//! Vectors represent displacements, directions, and other vector quantities.

use nalgebra::SVector;

/// A vector in D-dimensional space.
///
/// Vectors represent displacements, directions, and other vector quantities.
/// Used for spacing, offsets, and spatial transformations.
///
/// This is a thin wrapper around nalgebra's SVector to provide
/// domain-specific functionality while maintaining all nalgebra operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const D: usize>(pub SVector<f64, D>);

impl<const D: usize> Vector<D> {
    /// Create a new vector from components.
    pub fn new(components: [f64; D]) -> Self {
        Self(SVector::from(components))
    }

    /// Create a zero vector.
    pub fn zeros() -> Self {
        Self(SVector::zeros())
    }

    /// Create a new vector from a slice of components.
    pub fn from_slice(components: &[f64]) -> Self {
        assert!(components.len() == D, "Component slice length must match dimension");
        let mut vector = Self::zeros();
        for i in 0..D {
            vector.0[i] = components[i];
        }
        vector
    }

    /// Convert vector to a vector of components.
    pub fn to_vec(&self) -> Vec<f64> {
        (0..D).map(|i| self.0[i]).collect()
    }

    /// Create a unit vector along the x-axis.
    pub fn x_axis() -> Self {
        let mut v = Self::zeros();
        v.0[0] = 1.0;
        v
    }

    /// Create a unit vector along the y-axis.
    pub fn y_axis() -> Self {
        let mut v = Self::zeros();
        v.0[1] = 1.0;
        v
    }

    /// Create a unit vector along the z-axis.
    pub fn z_axis() -> Self {
        let mut v = Self::zeros();
        v.0[2] = 1.0;
        v
    }

    /// Get the inner nalgebra vector.
    pub fn inner(&self) -> &SVector<f64, D> {
        &self.0
    }

    /// Get mutable reference to inner nalgebra vector.
    pub fn inner_mut(&mut self) -> &mut SVector<f64, D> {
        &mut self.0
    }
}

impl<const D: usize> std::ops::Index<usize> for Vector<D> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> std::ops::IndexMut<usize> for Vector<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize> std::ops::Add for Vector<D> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}

impl<const D: usize> std::ops::Sub for Vector<D> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}

impl<const D: usize> std::ops::Mul<f64> for Vector<D> {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        Self(self.0 * scalar)
    }
}

impl<const D: usize> std::ops::Div<f64> for Vector<D> {
    type Output = Self;

    fn div(self, scalar: f64) -> Self::Output {
        Self(self.0 / scalar)
    }
}

impl<const D: usize> std::ops::Neg for Vector<D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases for testing
    type Vector3 = Vector<3>;

    #[test]
    fn test_vector_creation() {
        let v = Vector3::new([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_vector_from_slice() {
        let components = vec![1.0, 2.0, 3.0];
        let v = Vector3::from_slice(&components);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_vector_to_vec() {
        let v = Vector3::new([1.0, 2.0, 3.0]);
        let components = v.to_vec();
        assert_eq!(components, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_axes() {
        let x = Vector3::x_axis();
        assert_eq!(x, Vector3::new([1.0, 0.0, 0.0]));

        let y = Vector3::y_axis();
        assert_eq!(y, Vector3::new([0.0, 1.0, 0.0]));

        let z = Vector3::z_axis();
        assert_eq!(z, Vector3::new([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_vector_arithmetic() {
        let v1 = Vector3::new([1.0, 2.0, 3.0]);
        let v2 = Vector3::new([4.0, 5.0, 6.0]);

        let sum = v1 + v2;
        assert_eq!(sum, Vector3::new([5.0, 7.0, 9.0]));

        let diff = v2 - v1;
        assert_eq!(diff, Vector3::new([3.0, 3.0, 3.0]));

        let scaled = v1 * 2.0;
        assert_eq!(scaled, Vector3::new([2.0, 4.0, 6.0]));

        let divided = v2 / 2.0;
        assert_eq!(divided, Vector3::new([2.0, 2.5, 3.0]));

        let negated = -v1;
        assert_eq!(negated, Vector3::new([-1.0, -2.0, -3.0]));
    }
}
