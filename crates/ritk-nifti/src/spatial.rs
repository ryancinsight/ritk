//! NIfTI file-space affine conversion for RITK's internal ZYX image axes.
//!
//! RITK stores tensors as `[depth, row, col] = [z, y, x]`; NIfTI stores voxel
//! axes as `[x, y, z]`. RITK physical coordinates use LPS, while NIfTI affines
//! are interpreted as RAS. Therefore the file affine and internal affine differ
//! by both row signs and column order:
//!
//! ```text
//! A_lps_internal[:, depth] = ras_to_lps(A_ras_nifti[:, z])
//! A_lps_internal[:, row]   = ras_to_lps(A_ras_nifti[:, y])
//! A_lps_internal[:, col]   = ras_to_lps(A_ras_nifti[:, x])
//! ```

use nalgebra::{SMatrix, Vector3};
use ritk_spatial::{Direction, Point, Spacing};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InternalSpatialMetadata {
    pub(crate) origin: Point<3>,
    pub(crate) spacing: Spacing<3>,
    pub(crate) direction: Direction<3>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NiftiSformRows {
    pub(crate) x: [f32; 4],
    pub(crate) y: [f32; 4],
    pub(crate) z: [f32; 4],
}

/// Convert a NIfTI RAS affine into RITK LPS metadata for internal `[z,y,x]`
/// tensor axes.
pub(crate) fn metadata_from_nifti_ras_affine(affine: [[f32; 4]; 4]) -> InternalSpatialMetadata {
    let lps_file = ras_affine_to_lps_file_axes(affine);
    let origin = Point::new([lps_file[0][3], lps_file[1][3], lps_file[2][3]]);

    let scaled_columns = [
        // Internal depth axis corresponds to NIfTI z.
        Vector3::new(lps_file[0][2], lps_file[1][2], lps_file[2][2]),
        // Internal row axis corresponds to NIfTI y.
        Vector3::new(lps_file[0][1], lps_file[1][1], lps_file[2][1]),
        // Internal column axis corresponds to NIfTI x.
        Vector3::new(lps_file[0][0], lps_file[1][0], lps_file[2][0]),
    ];

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
        origin,
        spacing,
        direction: Direction(SMatrix::<f64, 3, 3>::from_columns(&direction_columns)),
    }
}

/// Build NIfTI RAS sform rows from RITK LPS metadata whose axes are ordered
/// `[depth,row,col]`.
pub(crate) fn sform_from_internal_lps_metadata(
    origin: [f64; 3],
    spacing: [f64; 3],
    direction_row_major: [f64; 9],
) -> NiftiSformRows {
    let lps_internal_columns = [
        scaled_direction_column(direction_row_major, spacing, 0),
        scaled_direction_column(direction_row_major, spacing, 1),
        scaled_direction_column(direction_row_major, spacing, 2),
    ];

    let file_columns = [
        // NIfTI x axis is RITK col.
        lps_internal_columns[2],
        // NIfTI y axis is RITK row.
        lps_internal_columns[1],
        // NIfTI z axis is RITK depth.
        lps_internal_columns[0],
    ];

    NiftiSformRows {
        x: [
            -(file_columns[0][0] as f32),
            -(file_columns[1][0] as f32),
            -(file_columns[2][0] as f32),
            -(origin[0] as f32),
        ],
        y: [
            -(file_columns[0][1] as f32),
            -(file_columns[1][1] as f32),
            -(file_columns[2][1] as f32),
            -(origin[1] as f32),
        ],
        z: [
            file_columns[0][2] as f32,
            file_columns[1][2] as f32,
            file_columns[2][2] as f32,
            origin[2] as f32,
        ],
    }
}

fn ras_affine_to_lps_file_axes(affine: [[f32; 4]; 4]) -> [[f64; 4]; 3] {
    [
        [
            -(affine[0][0] as f64),
            -(affine[0][1] as f64),
            -(affine[0][2] as f64),
            -(affine[0][3] as f64),
        ],
        [
            -(affine[1][0] as f64),
            -(affine[1][1] as f64),
            -(affine[1][2] as f64),
            -(affine[1][3] as f64),
        ],
        [
            affine[2][0] as f64,
            affine[2][1] as f64,
            affine[2][2] as f64,
            affine[2][3] as f64,
        ],
    ]
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
    fn ras_affine_columns_are_reordered_into_internal_axes() {
        let affine = [
            [-4.0, 0.0, 0.0, -10.0],
            [0.0, -3.0, 0.0, -20.0],
            [0.0, 0.0, 2.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let metadata = metadata_from_nifti_ras_affine(affine);

        assert_close(metadata.origin[0], 10.0);
        assert_close(metadata.origin[1], 20.0);
        assert_close(metadata.origin[2], 30.0);
        assert_close(metadata.spacing[0], 2.0);
        assert_close(metadata.spacing[1], 3.0);
        assert_close(metadata.spacing[2], 4.0);

        let expected =
            SMatrix::<f64, 3, 3>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(metadata.direction.0, expected);
    }

    #[test]
    fn internal_metadata_columns_are_reordered_into_nifti_sform() {
        let rows = sform_from_internal_lps_metadata(
            [10.0, 20.0, 30.0],
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        );

        assert_eq!(rows.x, [-4.0, -0.0, -0.0, -10.0]);
        assert_eq!(rows.y, [-0.0, -3.0, -0.0, -20.0]);
        assert_eq!(rows.z, [0.0, 0.0, 2.0, 30.0]);
    }
}
