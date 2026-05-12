//! NRRD file-space conversion for RITK's internal ZYX image axes.
//!
//! RITK stores tensors and spatial metadata in `[depth,row,col] = [z,y,x]`
//! order. NRRD `sizes` and `space directions` fields list file axes as
//! `[x,y,z]`. Therefore the NRRD file vectors and internal metadata columns
//! differ only by column order:
//!
//! ```text
//! A_internal[:, depth] = A_nrrd[:, z]
//! A_internal[:, row]   = A_nrrd[:, y]
//! A_internal[:, col]   = A_nrrd[:, x]
//! ```

use nalgebra::{SMatrix, Vector3};
use ritk_core::spatial::{Direction, Spacing};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InternalSpatialMetadata {
    pub(crate) spacing: Spacing<3>,
    pub(crate) direction: Direction<3>,
}

/// Convert NRRD `[x,y,z]` space-direction vectors into RITK internal
/// `[depth,row,col]` spacing and direction columns.
pub(crate) fn metadata_from_file_space_directions(
    file_vectors: [[f64; 3]; 3],
) -> InternalSpatialMetadata {
    let scaled_columns = [
        vector_from_array(file_vectors[2]),
        vector_from_array(file_vectors[1]),
        vector_from_array(file_vectors[0]),
    ];

    metadata_from_internal_scaled_columns(scaled_columns)
}

/// Convert NRRD `[x,y,z]` scalar spacings into RITK internal
/// `[depth,row,col]` spacing with the canonical axis-aligned direction
/// columns `[z,y,x]`.
pub(crate) fn metadata_from_file_spacings(file_spacing: [f64; 3]) -> InternalSpatialMetadata {
    metadata_from_file_space_directions([
        [file_spacing[0], 0.0, 0.0],
        [0.0, file_spacing[1], 0.0],
        [0.0, 0.0, file_spacing[2]],
    ])
}

/// Build NRRD `[x,y,z]` space-direction vectors from RITK internal
/// `[depth,row,col]` metadata.
pub(crate) fn file_space_directions_from_internal(
    spacing: [f64; 3],
    direction_row_major: [f64; 9],
) -> [[f64; 3]; 3] {
    let internal_columns = [
        scaled_direction_column(direction_row_major, spacing, 0),
        scaled_direction_column(direction_row_major, spacing, 1),
        scaled_direction_column(direction_row_major, spacing, 2),
    ];

    [
        internal_columns[2],
        internal_columns[1],
        internal_columns[0],
    ]
}

fn metadata_from_internal_scaled_columns(
    scaled_columns: [Vector3<f64>; 3],
) -> InternalSpatialMetadata {
    let spacing = Spacing::new([
        scaled_columns[0].norm(),
        scaled_columns[1].norm(),
        scaled_columns[2].norm(),
    ]);

    let direction_columns = [
        normalized_or_axis(
            scaled_columns[0],
            spacing[0],
            Vector3::z_axis().into_inner(),
        ),
        normalized_or_axis(
            scaled_columns[1],
            spacing[1],
            Vector3::y_axis().into_inner(),
        ),
        normalized_or_axis(
            scaled_columns[2],
            spacing[2],
            Vector3::x_axis().into_inner(),
        ),
    ];

    InternalSpatialMetadata {
        spacing,
        direction: Direction(SMatrix::<f64, 3, 3>::from_columns(&direction_columns)),
    }
}

fn vector_from_array(value: [f64; 3]) -> Vector3<f64> {
    Vector3::new(value[0], value[1], value[2])
}

fn normalized_or_axis(vector: Vector3<f64>, norm: f64, fallback: Vector3<f64>) -> Vector3<f64> {
    if norm > 1e-9 {
        vector / norm
    } else {
        fallback
    }
}

fn scaled_direction_column(
    direction_row_major: [f64; 9],
    spacing: [f64; 3],
    column: usize,
) -> [f64; 3] {
    [
        direction_row_major[column] * spacing[column],
        direction_row_major[3 + column] * spacing[column],
        direction_row_major[6 + column] * spacing[column],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn assert_close(got: f64, expected: f64) {
        assert!(
            (got - expected).abs() < EPS,
            "got {got:.12}, expected {expected:.12}"
        );
    }

    #[test]
    fn file_space_directions_are_reordered_into_internal_axes() {
        let metadata = metadata_from_file_space_directions([
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);

        assert_close(metadata.spacing[0], 2.0);
        assert_close(metadata.spacing[1], 3.0);
        assert_close(metadata.spacing[2], 4.0);

        let expected =
            SMatrix::<f64, 3, 3>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(metadata.direction.0, expected);
    }

    #[test]
    fn internal_metadata_columns_are_reordered_into_file_axes() {
        let directions = file_space_directions_from_internal(
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        );

        assert_eq!(directions[0], [4.0, 0.0, 0.0]);
        assert_eq!(directions[1], [0.0, 3.0, 0.0]);
        assert_eq!(directions[2], [0.0, 0.0, 2.0]);
    }
}
