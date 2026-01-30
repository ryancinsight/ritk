use nalgebra::{Point as NaPoint, SMatrix, SVector};

pub type Point<const D: usize> = NaPoint<f64, D>;
pub type Vector<const D: usize> = SVector<f64, D>;
pub type Spacing<const D: usize> = SVector<f64, D>;
pub type Direction<const D: usize> = SMatrix<f64, D, D>;

// Common aliases
pub type Point2 = Point<2>;
pub type Point3 = Point<3>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Spacing3 = Spacing<3>;
pub type Direction3 = Direction<3>;
