//! Image metadata types.
//!
//! This module provides types for representing image metadata
//! such as origin, spacing, and direction.

use crate::spatial::{Point, Spacing, Direction};

/// Image metadata containing physical space information.
///
/// Metadata describes how image indices map to physical coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageMetadata<const D: usize> {
    /// Physical coordinate of the first pixel (index 0, 0, ...).
    origin: Point<D>,
    /// Physical distance between pixels along each axis.
    spacing: Spacing<D>,
    /// Orientation of the image axes.
    direction: Direction<D>,
}

impl<const D: usize> ImageMetadata<D> {
    /// Create new image metadata.
    pub fn new(origin: Point<D>, spacing: Spacing<D>, direction: Direction<D>) -> Self {
        Self {
            origin,
            spacing,
            direction,
        }
    }

    /// Get the origin.
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Get the spacing.
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Get the direction.
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }

    /// Set the origin.
    pub fn set_origin(&mut self, origin: Point<D>) {
        self.origin = origin;
    }

    /// Set the spacing.
    pub fn set_spacing(&mut self, spacing: Spacing<D>) {
        self.spacing = spacing;
    }

    /// Set the direction.
    pub fn set_direction(&mut self, direction: Direction<D>) {
        self.direction = direction;
    }

    /// Create default metadata (identity transform, unit spacing, zero origin).
    pub fn default_for_shape(_shape: [usize; D]) -> Self {
        Self {
            origin: Point::origin(),
            spacing: Spacing::uniform(1.0),
            direction: Direction::identity(),
        }
    }
}

impl<const D: usize> Default for ImageMetadata<D> {
    fn default() -> Self {
        Self {
            origin: Point::origin(),
            spacing: Spacing::uniform(1.0),
            direction: Direction::identity(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases for testing
    type Point3 = Point<3>;
    type Spacing3 = Spacing<3>;
    type Direction3 = Direction<3>;

    #[test]
    fn test_metadata_creation() {
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();
        let metadata = ImageMetadata::new(origin, spacing, direction);
        assert_eq!(metadata.origin(), &origin);
        assert_eq!(metadata.spacing(), &spacing);
        assert_eq!(metadata.direction(), &direction);
    }

    #[test]
    fn test_metadata_default() {
        let metadata = ImageMetadata::<3>::default();
        assert_eq!(metadata.origin(), &Point3::origin());
        assert_eq!(metadata.spacing(), &Spacing3::uniform(1.0));
        assert_eq!(metadata.direction(), &Direction3::identity());
    }

    #[test]
    fn test_metadata_setters() {
        let mut metadata = ImageMetadata::<3>::default();
        let new_origin = Point3::new([1.0, 2.0, 3.0]);
        metadata.set_origin(new_origin);
        assert_eq!(metadata.origin(), &new_origin);
    }
}
