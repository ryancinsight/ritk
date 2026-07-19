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

use anyhow::{anyhow, bail, Result};
use ritk_spatial::{Direction, Point, Spacing, Vector};

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
pub(crate) fn metadata_from_nifti_ras_affine(
    affine: [[f32; 4]; 4],
) -> Result<InternalSpatialMetadata> {
    ensure_finite_affine(affine)?;

    let lps_file = ras_affine_to_lps_file_axes(affine);
    let origin = Point::new([lps_file[0][3], lps_file[1][3], lps_file[2][3]]);

    let scaled_columns = [
        // Internal depth axis corresponds to NIfTI z.
        Vector::new([lps_file[0][2], lps_file[1][2], lps_file[2][2]]),
        // Internal row axis corresponds to NIfTI y.
        Vector::new([lps_file[0][1], lps_file[1][1], lps_file[2][1]]),
        // Internal column axis corresponds to NIfTI x.
        Vector::new([lps_file[0][0], lps_file[1][0], lps_file[2][0]]),
    ];

    let spacing = Spacing::try_new([
        scaled_columns[0].norm(),
        scaled_columns[1].norm(),
        scaled_columns[2].norm(),
    ])
    .map_err(|e| anyhow!("invalid NIfTI affine spacing: {e}"))?;

    let direction_columns = [
        normalized_column(scaled_columns[0], spacing[0], "depth")?,
        normalized_column(scaled_columns[1], spacing[1], "row")?,
        normalized_column(scaled_columns[2], spacing[2], "column")?,
    ];

    Ok(InternalSpatialMetadata {
        origin,
        spacing,
        direction: Direction::from_columns(direction_columns),
    })
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

fn ensure_finite_affine(affine: [[f32; 4]; 4]) -> Result<()> {
    for (row_idx, row) in affine.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                bail!("NIfTI affine entry [{row_idx},{col_idx}] must be finite, got {value}");
            }
        }
    }

    Ok(())
}

fn normalized_column(vector: Vector<3>, norm: f64, axis: &str) -> Result<Vector<3>> {
    if norm.is_finite() && norm > 0.0 {
        Ok(vector / norm)
    } else {
        bail!("NIfTI affine {axis} column must have positive finite norm, got {norm}")
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

        let metadata = metadata_from_nifti_ras_affine(affine)
            .expect("positive finite affine must produce spatial metadata");

        assert_close(metadata.origin[0], 10.0);
        assert_close(metadata.origin[1], 20.0);
        assert_close(metadata.origin[2], 30.0);
        assert_close(metadata.spacing[0], 2.0);
        assert_close(metadata.spacing[1], 3.0);
        assert_close(metadata.spacing[2], 4.0);

        let expected = Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(metadata.direction, expected);
    }

    #[test]
    fn zero_affine_column_is_rejected() {
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let err = metadata_from_nifti_ras_affine(affine)
            .expect_err("zero NIfTI z column must be rejected");

        assert!(
            err.to_string().contains("invalid NIfTI affine spacing"),
            "error must name invalid spacing: {err}"
        );
    }

    #[test]
    fn non_finite_affine_entry_is_rejected() {
        let affine = [
            [1.0, 0.0, 0.0, f32::NAN],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let err = metadata_from_nifti_ras_affine(affine)
            .expect_err("non-finite NIfTI affine entry must be rejected");

        assert!(
            err.to_string().contains("must be finite"),
            "error must name finite affine invariant: {err}"
        );
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
