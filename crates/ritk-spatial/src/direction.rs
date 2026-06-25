//! Direction type for representing image orientation.
//!
//! Direction matrices represent orientation of image axes in physical space.

use crate::vector::Vector;
use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::record::{PrecisionSettings, Record};
use burn::tensor::backend::{AutodiffBackend, Backend};
use leto::FixedMatrix;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Tolerance for checking whether a matrix row/column is orthogonal.
/// Derived from f64 machine epsilon (~2.2e-16) with a 10-order practical margin.
const ORTHOGONALITY_TOLERANCE: f64 = 1e-6;

/// Threshold below which a pivot value is treated as singular in direction normalization.
const PIVOT_SINGULARITY_THRESHOLD: f64 = 1e-10;

/// Direction matrix representing image orientation.
///
/// The direction matrix is a D×D matrix where each column represents
/// direction of the corresponding image axis in physical space.
/// Column i represents the direction of the i-th image axis.
///
/// This is a stack-backed wrapper around Leto's fixed matrix primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direction<const D: usize>(pub FixedMatrix<f64, D, D>);

impl<const D: usize> Serialize for Direction<D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut rows = serializer.serialize_seq(Some(D))?;
        for row in 0..D {
            let mut values = Vec::with_capacity(D);
            for column in 0..D {
                values.push(self[(row, column)]);
            }
            rows.serialize_element(&values)?;
        }
        rows.end()
    }
}

impl<'de, const D: usize> Deserialize<'de> for Direction<D> {
    fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let rows = Vec::<Vec<f64>>::deserialize(deserializer)?;
        if rows.len() != D {
            return Err(serde::de::Error::invalid_length(
                rows.len(),
                &"row count matching the direction dimension",
            ));
        }

        let mut matrix = FixedMatrix::zeros();
        for (row_index, row) in rows.iter().enumerate() {
            if row.len() != D {
                return Err(serde::de::Error::invalid_length(
                    row.len(),
                    &"column count matching the direction dimension",
                ));
            }
            for (column_index, value) in row.iter().copied().enumerate() {
                matrix[(row_index, column_index)] = value;
            }
        }

        Ok(Self(matrix))
    }
}

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
        *self
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
        Self(FixedMatrix::identity())
    }

    /// Create a zero matrix.
    pub fn zeros() -> Self {
        Self(FixedMatrix::zeros())
    }

    /// Create a direction matrix from row-major rows.
    pub const fn from_rows(rows: [[f64; D]; D]) -> Self {
        Self(FixedMatrix::from_rows(rows))
    }

    /// Create a direction matrix from axis direction columns.
    pub fn from_columns(columns: [Vector<D>; D]) -> Self {
        Self(FixedMatrix::from_columns(columns.map(|column| column.0)))
    }

    /// Iterate over entries in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.0.iter()
    }

    /// Check if direction matrix is orthogonal (rotation matrix).
    pub fn is_orthogonal(&self) -> bool {
        let product = self.0 * self.0.transpose();
        let identity = Self::identity();
        (0..D).all(|i| {
            (0..D).all(|j| (product[(i, j)] - identity.0[(i, j)]).abs() < ORTHOGONALITY_TOLERANCE)
        })
    }

    /// Check if direction matrix is a proper rotation (det = 1).
    pub fn is_proper_rotation(&self) -> bool {
        self.is_orthogonal() && (self.determinant() - 1.0).abs() < ORTHOGONALITY_TOLERANCE
    }

    /// Compute the determinant of the direction matrix.
    ///
    /// Computes the determinant using cofactor expansion for D=2,3 and
    /// Gaussian elimination (LU decomposition) for D>3.
    pub fn determinant(&self) -> f64 {
        match D {
            2 => self.0[(0, 0)] * self.0[(1, 1)] - self.0[(0, 1)] * self.0[(1, 0)],
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
                let mut m = self.0.into_rows();
                let mut det = 1.0;

                for i in 0..D {
                    // Find pivot
                    let mut pivot_idx = i;
                    let mut pivot_val = m[i][i].abs();

                    for (k, row) in m.iter().enumerate().take(D).skip(i + 1) {
                        let val = row[i].abs();
                        if val > pivot_val {
                            pivot_val = val;
                            pivot_idx = k;
                        }
                    }

                    if pivot_val < PIVOT_SINGULARITY_THRESHOLD {
                        return 0.0;
                    }

                    if pivot_idx != i {
                        m.swap(i, pivot_idx);
                        det = -det;
                    }

                    det *= m[i][i];

                    let pivot_row = m[i];
                    let pivot = m[i][i];
                    for row in m.iter_mut().take(D).skip(i + 1) {
                        let factor = row[i] / pivot;
                        for (k, value) in row.iter_mut().enumerate().take(D).skip(i) {
                            *value -= factor * pivot_row[k];
                        }
                    }
                }

                det
            }
        }
    }

    /// Try to compute the inverse of the direction matrix.
    pub fn try_inverse(&self) -> Option<Self> {
        let mut matrix = self.0.into_rows();
        let mut inverse = FixedMatrix::<f64, D, D>::identity().into_rows();

        for column in 0..D {
            let mut pivot_row = column;
            let mut pivot_value = matrix[column][column].abs();
            for (row, values) in matrix.iter().enumerate().take(D).skip(column + 1) {
                let value = values[column].abs();
                if value > pivot_value {
                    pivot_value = value;
                    pivot_row = row;
                }
            }

            if pivot_value < PIVOT_SINGULARITY_THRESHOLD {
                return None;
            }

            if pivot_row != column {
                matrix.swap(column, pivot_row);
                inverse.swap(column, pivot_row);
            }

            let pivot = matrix[column][column];
            for col in 0..D {
                matrix[column][col] /= pivot;
                inverse[column][col] /= pivot;
            }

            for row in 0..D {
                if row == column {
                    continue;
                }
                let factor = matrix[row][column];
                for col in 0..D {
                    matrix[row][col] -= factor * matrix[column][col];
                    inverse[row][col] -= factor * inverse[column][col];
                }
            }
        }

        Some(Self(FixedMatrix::from_rows(inverse)))
    }

    /// Get the axis directions as a fixed-size array (zero-allocation).
    pub fn axis_directions_array(&self) -> [Vector<D>; D] {
        std::array::from_fn(|i| {
            let mut v = Vector::zeros();
            for j in 0..D {
                v[j] = self.0[(j, i)];
            }
            v
        })
    }

    /// Get the inner fixed matrix.
    pub fn inner(&self) -> &FixedMatrix<f64, D, D> {
        &self.0
    }

    /// Get mutable reference to inner fixed matrix.
    pub fn inner_mut(&mut self) -> &mut FixedMatrix<f64, D, D> {
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

impl Direction<3> {
    /// Create a 3D direction matrix from row-major entries.
    pub const fn from_row_major(entries: [f64; 9]) -> Self {
        Self(FixedMatrix::from_row_major(entries))
    }

    /// Create a 3D direction matrix from column-major entries.
    pub const fn from_column_major(entries: [f64; 9]) -> Self {
        Self(FixedMatrix::from_column_major(entries))
    }

    /// Return the 3D direction matrix in row-major order.
    pub fn to_row_major(self) -> [f64; 9] {
        self.0.into_row_major()
    }

    /// Return the 3D direction matrix in column-major order.
    pub fn to_column_major(self) -> [f64; 9] {
        self.0.into_column_major()
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
    fn test_direction_axis_directions_array() {
        let identity = Direction3::identity();
        let axes = identity.axis_directions_array();
        assert_eq!(axes[0], Vector3::new([1.0, 0.0, 0.0]));
        assert_eq!(axes[1], Vector3::new([0.0, 1.0, 0.0]));
        assert_eq!(axes[2], Vector3::new([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_direction_inverse_multiplies_to_identity() {
        let direction = Direction3::from_rows([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]);

        let inverse = direction
            .try_inverse()
            .expect("invariant: matrix is nonsingular");
        let product = direction * inverse;

        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((product[(row, col)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_direction_storage_order_conversions() {
        let row_major = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let column_major = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        let direction = Direction3::from_row_major(row_major);

        assert_eq!(direction.to_row_major(), row_major);
        assert_eq!(direction.to_column_major(), column_major);
        assert_eq!(Direction3::from_column_major(column_major), direction);
    }
}
