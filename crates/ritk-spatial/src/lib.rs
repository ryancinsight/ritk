//! Spatial types: Point, Direction, Spacing, Vector, VoxelIndex, VolumeDims.
//!
//! Leaf crate with no ritk-internal dependencies — only nalgebra, burn, and serde.

pub mod direction;
pub mod point;
pub mod spacing;
pub mod vector;
pub mod volume_dims;
pub mod voxel_index;

pub use direction::Direction;
pub use point::Point;
pub use spacing::{InvalidSpacing, Spacing};
pub use vector::Vector;
pub use volume_dims::VolumeDims;
pub use voxel_index::VoxelIndex;

// Common type aliases for 2D and 3D
pub type Point2 = Point<2>;
pub type Point3 = Point<3>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Spacing2 = Spacing<2>;
pub type Spacing3 = Spacing<3>;
pub type Direction2 = Direction<2>;
pub type Direction3 = Direction<3>;
