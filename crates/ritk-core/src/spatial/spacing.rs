//! Spacing type for representing physical distances between pixels/voxels.
//!
//! Spacing represents the physical distance between adjacent pixels/voxels
//! along each axis of an image.

use super::Vector;

/// Spacing between adjacent pixels/voxels along each axis.
///
/// Spacing is a vector where each component represents the physical distance
/// between adjacent pixels/voxels along that axis.
///
/// This is a type alias to Vector for semantic clarity.
pub type Spacing<const D: usize> = Vector<D>;

impl<const D: usize> Spacing<D> {
    /// Create uniform spacing (same value for all dimensions).
    pub fn uniform(value: f64) -> Self {
        let mut spacing = Vector::zeros();
        for i in 0..D {
            spacing[i] = value;
        }
        spacing
    }

    /// Check if spacing is uniform (all components equal).
    pub fn is_uniform(&self) -> bool {
        if D == 0 {
            return true;
        }
        let first = self[0];
        (1..D).all(|i| (self[i] - first).abs() < 1e-9)
    }

    /// Get the minimum spacing value.
    pub fn min_spacing(&self) -> f64 {
        (0..D).map(|i| self[i]).fold(f64::INFINITY, |a, b| a.min(b))
    }

    /// Get the maximum spacing value.
    pub fn max_spacing(&self) -> f64 {
        (0..D).map(|i| self[i]).fold(f64::NEG_INFINITY, |a, b| a.max(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases for testing
    type Spacing3 = Spacing<3>;

    #[test]
    fn test_spacing_creation() {
        let s = Spacing3::new([1.0, 2.0, 3.0]);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 2.0);
        assert_eq!(s[2], 3.0);
    }

    #[test]
    fn test_spacing_uniform() {
        let s = Spacing3::uniform(1.0);
        assert_eq!(s, Spacing3::new([1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_spacing_is_uniform() {
        let uniform = Spacing3::uniform(1.0);
        assert!(uniform.is_uniform());

        let non_uniform = Spacing3::new([1.0, 2.0, 3.0]);
        assert!(!non_uniform.is_uniform());
    }

    #[test]
    fn test_spacing_min_max() {
        let s = Spacing3::new([1.0, 2.0, 3.0]);
        assert_eq!(s.min_spacing(), 1.0);
        assert_eq!(s.max_spacing(), 3.0);
    }
}
