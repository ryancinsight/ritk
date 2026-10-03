//! Vector type for representing spatial displacements and directions.
//!
//! Vectors represent displacements, directions, and other vector quantities.

use leto::FixedVector;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A vector in D-dimensional space.
///
/// Vectors represent displacements, directions, and other vector quantities.
/// Used for spacing, offsets, and spatial transformations.
///
/// This is a stack-backed wrapper around Leto's fixed vector primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const D: usize>(pub FixedVector<f64, D>);

impl<const D: usize> Serialize for Vector<D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(D))?;
        for value in self.as_slice() {
            seq.serialize_element(value)?;
        }
        seq.end()
    }
}

impl<'de, const D: usize> Deserialize<'de> for Vector<D> {
    fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let components = Vec::<f64>::deserialize(deserializer)?;
        if components.len() != D {
            return Err(serde::de::Error::invalid_length(
                components.len(),
                &"component count matching the vector dimension",
            ));
        }
        Ok(Self::from_slice(&components))
    }
}

impl<const D: usize> Vector<D> {
    /// Conventional threshold for callers that classify short vectors as degenerate.
    pub const DEGENERATE_LENGTH: f64 = 1e-9;

    /// Create a new vector from components.
    pub fn new(components: [f64; D]) -> Self {
        Self(FixedVector::new(components))
    }

    /// Create a zero vector.
    pub fn zeros() -> Self {
        Self(FixedVector::zeros())
    }

    /// Create a new vector from a slice of components.
    pub fn from_slice(components: &[f64]) -> Self {
        assert!(
            components.len() == D,
            "Component slice length must match dimension"
        );
        let mut vector = Self::zeros();
        for (i, &c) in components.iter().enumerate().take(D) {
            vector.0[i] = c;
        }
        vector
    }

    /// Convert vector to a fixed-size array of components (zero-allocation).
    pub fn to_array(&self) -> [f64; D] {
        self.0.into_array()
    }

    /// Borrow the vector components as a slice without allocation.
    pub fn as_slice(&self) -> &[f64] {
        self.0.as_array()
    }

    /// Compute the Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.as_slice()
            .iter()
            .fold(0.0_f64, |magnitude, component| magnitude.hypot(*component))
    }

    /// Compute the dot product with another vector.
    pub fn dot(&self, other: &Self) -> f64 {
        self.0.dot(&other.0)
    }

    /// Return the unit-length direction of this vector.
    ///
    /// Returns `None` when the vector is zero-length or non-finite, because no
    /// direction cosine is defined for that input.
    pub fn normalized(&self) -> Option<Self> {
        let norm = self.norm();
        if norm.is_finite() && norm > 0.0 {
            Some(*self / norm)
        } else {
            None
        }
    }

    /// Divide by `length` to recover a unit direction, or return `fallback`
    /// when `length` is zero, negative, or non-finite.
    ///
    /// Image readers store each axis as a direction column scaled by its
    /// spacing; any positive spacing remains meaningful regardless of scale.
    #[must_use]
    pub fn unit_direction_or(self, length: f64, fallback: Self) -> Self {
        if length.is_finite() && length > 0.0 {
            self / length
        } else {
            fallback
        }
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

    /// Get the inner fixed vector.
    pub fn inner(&self) -> &FixedVector<f64, D> {
        &self.0
    }

    /// Get mutable reference to inner fixed vector.
    pub fn inner_mut(&mut self) -> &mut FixedVector<f64, D> {
        &mut self.0
    }
}

impl Vector<3> {
    /// Compute the 3D cross product.
    pub fn cross(&self, other: &Self) -> Self {
        Self::new([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
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
    fn unit_direction_or_divides_by_the_length() {
        let scaled = Vector3::new([0.0, 3.0, 4.0]);
        assert_eq!(
            scaled.unit_direction_or(5.0, Vector3::x_axis()).to_array(),
            [0.0, 0.6, 0.8]
        );
    }

    #[test]
    fn unit_direction_or_returns_the_fallback_for_nonpositive_or_nonfinite_length() {
        let scaled = Vector3::new([0.0, 3.0, 4.0]);
        for length in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                scaled.unit_direction_or(length, Vector3::z_axis()),
                Vector3::z_axis()
            );
        }

        let just_above = Vector3::DEGENERATE_LENGTH * 2.0;
        assert_eq!(
            scaled
                .unit_direction_or(just_above, Vector3::z_axis())
                .to_array(),
            (scaled / just_above).to_array()
        );
    }

    #[test]
    fn unit_direction_or_preserves_small_positive_axis_orientation() {
        let spacing = Vector3::DEGENERATE_LENGTH / 10.0;
        let scaled = Vector3::new([-spacing, 0.0, 0.0]);
        assert_eq!(
            scaled
                .unit_direction_or(spacing, Vector3::z_axis())
                .to_array(),
            [-1.0, 0.0, 0.0]
        );
    }

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
    fn test_vector_to_array() {
        let v = Vector3::new([1.0, 2.0, 3.0]);
        let components = v.to_array();
        assert_eq!(components, [1.0, 2.0, 3.0]);
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

    #[test]
    fn test_vector_dot_norm_and_normalized() {
        let v = Vector3::new([3.0, 4.0, 0.0]);
        assert_eq!(v.dot(&Vector3::new([1.0, 2.0, 3.0])), 11.0);
        assert_eq!(v.norm(), 5.0);
        assert_eq!(v.normalized(), Some(Vector3::new([0.6, 0.8, 0.0])));
        assert_eq!(Vector3::zeros().normalized(), None);
    }

    #[test]
    fn vector_norm_preserves_finite_values_across_extreme_scales() {
        assert_eq!(Vector3::new([1.0e-200, 0.0, 0.0]).norm(), 1.0e-200);
        assert_eq!(Vector3::new([1.0e200, 0.0, 0.0]).norm(), 1.0e200);
        assert_eq!(
            Vector3::new([1.0e200, 1.0e200, 0.0]).norm(),
            1.0e200_f64.hypot(1.0e200)
        );
    }

    #[test]
    fn test_vector_cross_product() {
        assert_eq!(
            Vector3::x_axis().cross(&Vector3::y_axis()),
            Vector3::z_axis()
        );
        assert_eq!(
            Vector3::y_axis().cross(&Vector3::x_axis()),
            -Vector3::z_axis()
        );
        assert_eq!(
            Vector3::new([2.0, 3.0, 4.0]).cross(&Vector3::new([5.0, 6.0, 7.0])),
            Vector3::new([-3.0, 6.0, -3.0])
        );
    }
}
