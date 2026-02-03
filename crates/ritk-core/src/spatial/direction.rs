//! Direction type for representing image orientation.
//!
//! Direction matrices represent orientation of image axes in physical space.

use nalgebra::SMatrix;
use super::Vector;
use serde::{Serialize, Deserialize};
use burn::module::{Module, ModuleDisplay, ModuleDisplayDefault, AutodiffModule, Content};
use burn::record::{Record, PrecisionSettings};
use burn::tensor::backend::{Backend, AutodiffBackend};

/// Direction matrix representing image orientation.
///
/// The direction matrix is a D×D matrix where each column represents
/// direction of the corresponding image axis in physical space.
/// Column i represents the direction of the i-th image axis.
///
/// This is a thin wrapper around nalgebra's SMatrix to provide
/// domain-specific functionality while maintaining all nalgebra operations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Direction<const D: usize>(pub SMatrix<f64, D, D>);

impl<B: Backend, const D: usize> Record<B> for Direction<D> {
    type Item<S: PrecisionSettings> = Direction<D>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl<B: Backend, const D: usize> Module<B> for Direction<D> {
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

impl<B: AutodiffBackend, const D: usize> AutodiffModule<B> for Direction<D> {
    type InnerModule = Direction<D>;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl<const D: usize> ModuleDisplayDefault for Direction<D> {
    fn content(&self, content: Content) -> Option<Content> {
        Some(content.set_top_level_type(&format!("Direction{}D", D)))
    }
}

impl<const D: usize> ModuleDisplay for Direction<D> {}


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
    ///
    /// Computes the determinant using cofactor expansion for D=2,3 and
    /// Gaussian elimination (LU decomposition) for D>3.
    pub fn determinant(&self) -> f64 {
        match D {
            2 => {
                self.0[(0, 0)] * self.0[(1, 1)] - self.0[(0, 1)] * self.0[(1, 0)]
            }
            3 => {
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
            _ => {
                // General implementation using Gaussian elimination
                let mut m = self.0;
                let mut det = 1.0;
                
                for i in 0..D {
                    // Find pivot
                    let mut pivot_idx = i;
                    let mut pivot_val = m[(i, i)].abs();
                    
                    for k in (i + 1)..D {
                        let val = m[(k, i)].abs();
                        if val > pivot_val {
                            pivot_val = val;
                            pivot_idx = k;
                        }
                    }
                    
                    if pivot_val < 1e-10 {
                        return 0.0;
                    }
                    
                    if pivot_idx != i {
                        m.swap_rows(i, pivot_idx);
                        det = -det;
                    }
                    
                    det *= m[(i, i)];
                    
                    for j in (i + 1)..D {
                        let factor = m[(j, i)] / m[(i, i)];
                        for k in i..D {
                            m[(j, k)] -= factor * m[(i, k)];
                        }
                    }
                }
                
                det
            }
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
