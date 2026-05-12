//! MetaImage file-space conversion for RITK's internal ZYX image axes.
//!
//! RITK tensors and spatial metadata use `[depth,row,col] = [z,y,x]`.
//! MetaImage `DimSize`, `ElementSpacing`, and `TransformMatrix` fields are
//! listed in file-axis order `[x,y,z]`. The conversion is therefore a column
//! reorder between file axes and internal tensor axes:
//!
//! ```text
//! A_internal[:, depth] = A_metaimage[:, z]
//! A_internal[:, row]   = A_metaimage[:, y]
//! A_internal[:, col]   = A_metaimage[:, x]
//! ```

use nalgebra::{SMatrix, Vector3};
use ritk_core::spatial::{Direction, Spacing};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InternalSpatialMetadata {
    pub(crate) spacing: Spacing<3>,
    pub(crate) direction: Direction<3>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MetaImageSpatialFields {
    pub(crate) element_spacing: [f64; 3],
    pub(crate) transform_matrix_row_major: [f64; 9],
}

/// Convert MetaImage `[x,y,z]` spacing and direction fields into RITK
/// `[depth,row,col]` spacing and direction columns.
pub(crate) fn metadata_from_file_transform(
    file_spacing: [f64; 3],
    transform_matrix_row_major: [f64; 9],
) -> InternalSpatialMetadata {
    let file_columns = [
        scaled_file_direction_column(transform_matrix_row_major, file_spacing, 0),
        scaled_file_direction_column(transform_matrix_row_major, file_spacing, 1),
        scaled_file_direction_column(transform_matrix_row_major, file_spacing, 2),
    ];

    metadata_from_internal_scaled_columns([file_columns[2], file_columns[1], file_columns[0]])
}

/// Convert RITK `[depth,row,col]` spacing and direction columns into MetaImage
/// `[x,y,z]` header fields.
pub(crate) fn file_spatial_fields_from_internal(
    spacing: [f64; 3],
    direction_row_major: [f64; 9],
) -> MetaImageSpatialFields {
    let internal_columns = [
        direction_column(direction_row_major, 0),
        direction_column(direction_row_major, 1),
        direction_column(direction_row_major, 2),
    ];
    let file_columns = [
        internal_columns[2],
        internal_columns[1],
        internal_columns[0],
    ];

    MetaImageSpatialFields {
        element_spacing: [spacing[2], spacing[1], spacing[0]],
        transform_matrix_row_major: [
            file_columns[0][0],
            file_columns[1][0],
            file_columns[2][0],
            file_columns[0][1],
            file_columns[1][1],
            file_columns[2][1],
            file_columns[0][2],
            file_columns[1][2],
            file_columns[2][2],
        ],
    }
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

fn scaled_file_direction_column(
    transform_matrix_row_major: [f64; 9],
    file_spacing: [f64; 3],
    column: usize,
) -> Vector3<f64> {
    Vector3::new(
        transform_matrix_row_major[column] * file_spacing[column],
        transform_matrix_row_major[3 + column] * file_spacing[column],
        transform_matrix_row_major[6 + column] * file_spacing[column],
    )
}

fn direction_column(direction_row_major: [f64; 9], column: usize) -> [f64; 3] {
    [
        direction_row_major[column],
        direction_row_major[3 + column],
        direction_row_major[6 + column],
    ]
}

fn normalized_or_axis(vector: Vector3<f64>, norm: f64, fallback: Vector3<f64>) -> Vector3<f64> {
    if norm > 1e-9 {
        vector / norm
    } else {
        fallback
    }
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
    fn file_transform_columns_are_reordered_into_internal_axes() {
        let metadata = metadata_from_file_transform(
            [4.0, 3.0, 2.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );

        assert_close(metadata.spacing[0], 2.0);
        assert_close(metadata.spacing[1], 3.0);
        assert_close(metadata.spacing[2], 4.0);

        let expected =
            SMatrix::<f64, 3, 3>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(metadata.direction.0, expected);
    }

    #[test]
    fn internal_metadata_columns_are_reordered_into_file_fields() {
        let fields = file_spatial_fields_from_internal(
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        );

        assert_eq!(fields.element_spacing, [4.0, 3.0, 2.0]);
        assert_eq!(
            fields.transform_matrix_row_major,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }
}
