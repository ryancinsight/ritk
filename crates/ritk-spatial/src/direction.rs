//! Direction type for representing image orientation.
//!
//! Direction matrices represent orientation of image axes in physical space.

use crate::vector::Vector;
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
    ///
    /// Returns [`None`] when the matrix is numerically singular, when any
    /// entry of the input is non-finite, and when the pivoted elimination
    /// produces a non-finite entry — a pivot, a normalized row, or an updated
    /// row of the working matrix or of the inverse. That last case is how
    /// overflow surfaces: `MAX + MAX` becomes an infinity and any later
    /// `inf - inf` a NaN, so an entry leaving the finite range is rejected the
    /// moment it does, rather than returned as a plausible-looking inverse.
    /// The pivot threshold and the elimination order are unchanged.
    pub fn try_inverse(&self) -> Option<Self> {
        let mut matrix = self.0.into_rows();
        if !matrix.iter().flatten().all(|value| value.is_finite()) {
            return None;
        }
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

            if !pivot_value.is_finite() || pivot_value < PIVOT_SINGULARITY_THRESHOLD {
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
                if !matrix[column][col].is_finite() || !inverse[column][col].is_finite() {
                    return None;
                }
            }

            for row in 0..D {
                if row == column {
                    continue;
                }
                let factor = matrix[row][column];
                for col in 0..D {
                    matrix[row][col] -= factor * matrix[column][col];
                    inverse[row][col] -= factor * inverse[column][col];
                    if !matrix[row][col].is_finite() || !inverse[row][col].is_finite() {
                        return None;
                    }
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
        Self(<FixedMatrix<f64, 3, 3>>::from_row_major(entries))
    }

    /// Create a 3D direction matrix from column-major entries.
    pub const fn from_column_major(entries: [f64; 9]) -> Self {
        Self(<FixedMatrix<f64, 3, 3>>::from_column_major(entries))
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

    /// Every non-finite value at every entry of ranks one through three is
    /// refused: 3 ranks of 1, 4 and 9 entries against NaN and both infinities
    /// is the 42 cases the item names.
    #[test]
    fn try_inverse_rejects_non_finite_input_at_every_entry() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut rank_one = [[1.0_f64; 1]; 1];
            rank_one[0][0] = bad;
            assert!(
                Direction::<1>::from_rows(rank_one).try_inverse().is_none(),
                "rank 1 must refuse {bad}"
            );

            for index in 0..4 {
                let mut rank_two = [[0.0_f64; 2]; 2];
                rank_two[0][0] = 1.0;
                rank_two[1][1] = 1.0;
                rank_two[index / 2][index % 2] = bad;
                assert!(
                    Direction::<2>::from_rows(rank_two).try_inverse().is_none(),
                    "rank 2 entry {index} must refuse {bad}"
                );
            }

            for index in 0..9 {
                let mut rank_three = [[0.0_f64; 3]; 3];
                rank_three[0][0] = 1.0;
                rank_three[1][1] = 1.0;
                rank_three[2][2] = 1.0;
                rank_three[index / 3][index % 3] = bad;
                assert!(
                    Direction::<3>::from_rows(rank_three)
                        .try_inverse()
                        .is_none(),
                    "rank 3 entry {index} must refuse {bad}"
                );
            }
        }
    }

    /// Overflow while normalizing the pivot row leaves the finite range, so the
    /// inverse is refused instead of being returned with an infinity in it.
    #[test]
    fn try_inverse_rejects_overflow_during_normalization() {
        // Column 0's pivot is the 1e-9 it sits on, and normalizing the row
        // divides 1e300 by it: 1e309 overflows to an infinity.
        let direction = Direction::<2>::from_rows([[1e-9, 1e300], [0.0, 1.0]]);
        assert!(direction.try_inverse().is_none());
    }

    /// A pivoted, non-orthonormal matrix with an exactly representable inverse
    /// is still accepted: the guard rejects non-finite arithmetic, not shape.
    #[test]
    fn try_inverse_accepts_an_exact_pivoted_shear() {
        let direction = Direction::<2>::from_rows([[0.0, 1.0], [1.0, 0.0]]);
        let inverse = direction
            .try_inverse()
            .expect("invariant: the swap fixture is its own inverse");
        assert_eq!(inverse, direction);
    }

    /// The widest exactly representable diagonal whose every pivot clears the
    /// unchanged singularity threshold still inverts, and its inverse is the
    /// exact reciprocal triple with no overflow.
    #[test]
    fn try_inverse_accepts_the_extreme_dyadic_diagonal() {
        let direction = Direction::<3>::from_rows([
            [2f64.powi(1023), 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2f64.powi(-9)],
        ]);
        let inverse = direction
            .try_inverse()
            .expect("invariant: dyadic diagonal clears the pivot threshold");
        assert_eq!(inverse[(0, 0)].to_bits(), 2f64.powi(-1023).to_bits());
        assert_eq!(inverse[(1, 1)].to_bits(), 1.0_f64.to_bits());
        assert_eq!(inverse[(2, 2)].to_bits(), 2f64.powi(9).to_bits());
    }

    /// A diagonal entry below the unchanged pivot threshold is refused as
    /// singular — including the smallest subnormal-scale dyadic — so the
    /// finiteness guard never has to launder an overflow into a "valid" answer.
    #[test]
    fn try_inverse_refuses_a_diagonal_below_the_pivot_threshold() {
        let direction = Direction::<3>::from_rows([
            [2f64.powi(1023), 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2f64.powi(-1023)],
        ]);
        assert!(direction.try_inverse().is_none());
    }
}
