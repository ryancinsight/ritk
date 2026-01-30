//! Spatial types for representing points, vectors, spacing, and direction matrices.
//!
//! This module provides the fundamental spatial types used throughout ritk.
//! All types are based on nalgebra for efficient linear algebra operations.

pub mod point;
pub mod vector;
pub mod spacing;
pub mod direction;

pub use point::Point;
pub use vector::Vector;
pub use spacing::Spacing;
pub use direction::Direction;

// Common type aliases for 2D and 3D
pub type Point2 = Point<2>;
pub type Point3 = Point<3>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Spacing2 = Spacing<2>;
pub type Spacing3 = Spacing<3>;
pub type Direction2 = Direction<2>;
pub type Direction3 = Direction<3>;
