//! Point type for representing spatial coordinates.
//!
//! Points represent positions in physical space.

use nalgebra::Point as NaPoint;
use super::Vector;
use serde::{Serialize, Deserialize};
use burn::module::{Module, ModuleDisplay, ModuleDisplayDefault, AutodiffModule, Content};
use burn::record::{Record, PrecisionSettings};
use burn::tensor::backend::{Backend, AutodiffBackend};

/// A point in D-dimensional space.
///
/// Points represent positions in physical coordinate systems.
/// Used for image origins, physical coordinates, and spatial transformations.
///
/// This is a thin wrapper around nalgebra's Point to provide
/// domain-specific functionality while maintaining all nalgebra operations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point<const D: usize>(pub NaPoint<f64, D>);

impl<B: Backend, const D: usize> Record<B> for Point<D> {
    type Item<S: PrecisionSettings> = Point<D>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl<B: Backend, const D: usize> Module<B> for Point<D> {
    type Record = Self;

    fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // No tensors to visit
    }

    fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        record
    }

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        devices
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }
}

impl<B: AutodiffBackend, const D: usize> AutodiffModule<B> for Point<D> {
    type InnerModule = Point<D>;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl<const D: usize> ModuleDisplayDefault for Point<D> {
    fn content(&self, content: Content) -> Option<Content> {
        Some(content.set_top_level_type(&format!("Point{}D", D)))
    }
}

impl<const D: usize> ModuleDisplay for Point<D> {}


impl<const D: usize> Point<D> {
    /// Create a new point from coordinates.
    pub fn new(coords: [f64; D]) -> Self {
        Self(NaPoint::from(coords))
    }

    /// Create a point at the origin (all coordinates zero).
    pub fn origin() -> Self {
        Self(NaPoint::origin())
    }

    /// Create a new point from a slice of coordinates.
    pub fn from_slice(coords: &[f64]) -> Self {
        assert!(coords.len() == D, "Coordinate slice length must match dimension");
        let mut point = Self::origin();
        for i in 0..D {
            point.0.coords[i] = coords[i];
        }
        point
    }

    /// Convert point to a vector of coordinates.
    pub fn to_vec(&self) -> Vec<f64> {
        (0..D).map(|i| self.0.coords[i]).collect()
    }

    /// Get the inner nalgebra point.
    pub fn inner(&self) -> &NaPoint<f64, D> {
        &self.0
    }

    /// Get mutable reference to inner nalgebra point.
    pub fn inner_mut(&mut self) -> &mut NaPoint<f64, D> {
        &mut self.0
    }
}

impl<const D: usize> std::ops::Index<usize> for Point<D> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0.coords[index]
    }
}

impl<const D: usize> std::ops::IndexMut<usize> for Point<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0.coords[index]
    }
}

impl<const D: usize> std::ops::Sub for Point<D> {
    type Output = Vector<D>;

    fn sub(self, other: Self) -> Self::Output {
        Vector(self.0.coords - other.0.coords)
    }
}

impl<const D: usize> std::ops::Add<Vector<D>> for Point<D> {
    type Output = Self;

    fn add(self, vector: Vector<D>) -> Self::Output {
        Self(self.0 + vector.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases for testing
    type Point3 = Point<3>;
    type Vector3 = Vector<3>;

    #[test]
    fn test_point_creation() {
        let p = Point3::new([1.0, 2.0, 3.0]);
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 2.0);
        assert_eq!(p[2], 3.0);
    }

    #[test]
    fn test_point_origin() {
        let p = Point3::origin();
        assert_eq!(p[0], 0.0);
        assert_eq!(p[1], 0.0);
        assert_eq!(p[2], 0.0);
    }

    #[test]
    fn test_point_from_slice() {
        let coords = vec![1.0, 2.0, 3.0];
        let p = Point3::from_slice(&coords);
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 2.0);
        assert_eq!(p[2], 3.0);
    }

    #[test]
    fn test_point_to_vec() {
        let p = Point3::new([1.0, 2.0, 3.0]);
        let coords = p.to_vec();
        assert_eq!(coords, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_point_subtraction() {
        let p1 = Point3::new([5.0, 5.0, 5.0]);
        let p2 = Point3::new([2.0, 3.0, 4.0]);
        let diff = p1 - p2;
        assert_eq!(diff, Vector3::new([3.0, 2.0, 1.0]));
    }

    #[test]
    fn test_point_vector_addition() {
        let p = Point3::new([1.0, 2.0, 3.0]);
        let v = Vector3::new([4.0, 5.0, 6.0]);
        let result = p + v;
        assert_eq!(result, Point3::new([5.0, 7.0, 9.0]));
    }
}
