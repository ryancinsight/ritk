//! `SpatialTransform` variants and 4×4 homogeneous-matrix helpers.

use ndarray::Array3;

use super::error::SpatialError;

/// Classical spatial transform variants.
#[derive(Debug, Clone)]
pub enum SpatialTransform {
    /// Rigid body transform: 3×3 rotation + 3×1 translation.
    RigidBody {
        /// Row-major 9-element rotation matrix.
        rotation: [f64; 9],
        /// Translation vector [tx, ty, tz].
        translation: [f64; 3],
    },
    /// Affine transform: 3×4 matrix (12 DOF).
    Affine {
        /// Row-major 12-element affine matrix [r00, r01, r02, t0, ...].
        matrix: [f64; 12],
    },
    /// Non-rigid transform via deformation field.
    NonRigid {
        /// Deformation field tensor.
        deformation_field: Array3<f64>,
    },
}

/// Build 4×4 homogeneous transformation matrix from rotation and translation.
pub(crate) fn build_homogeneous_matrix(rotation: &[f64; 9], translation: &[f64; 3]) -> [f64; 16] {
    [
        rotation[0],
        rotation[1],
        rotation[2],
        translation[0],
        rotation[3],
        rotation[4],
        rotation[5],
        translation[1],
        rotation[6],
        rotation[7],
        rotation[8],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// Extract `SpatialTransform` from a 4×4 homogeneous matrix.
///
/// Determines rigid vs affine by checking whether all three column norms
/// are within 1% of unity.
pub(crate) fn extract_spatial_transform(
    matrix: &[f64; 16],
) -> Result<SpatialTransform, SpatialError> {
    let scale_x = (matrix[0].powi(2) + matrix[1].powi(2) + matrix[2].powi(2)).sqrt();
    let scale_y = (matrix[4].powi(2) + matrix[5].powi(2) + matrix[6].powi(2)).sqrt();
    let scale_z = (matrix[8].powi(2) + matrix[9].powi(2) + matrix[10].powi(2)).sqrt();

    let is_rigid = (scale_x - 1.0).abs() < 0.01
        && (scale_y - 1.0).abs() < 0.01
        && (scale_z - 1.0).abs() < 0.01;

    if is_rigid {
        let rotation = [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ];
        let translation = [matrix[3], matrix[7], matrix[11]];
        Ok(SpatialTransform::RigidBody {
            rotation,
            translation,
        })
    } else {
        let affine_matrix = [
            matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5], matrix[6], matrix[7],
            matrix[8], matrix[9], matrix[10], matrix[11],
        ];
        Ok(SpatialTransform::Affine {
            matrix: affine_matrix,
        })
    }
}

/// Transform a single point using a 4×4 homogeneous matrix.
pub(crate) fn transform_point(point: &[f64; 3], transform: &[f64; 16]) -> [f64; 3] {
    let x =
        transform[0] * point[0] + transform[1] * point[1] + transform[2] * point[2] + transform[3];
    let y =
        transform[4] * point[0] + transform[5] * point[1] + transform[6] * point[2] + transform[7];
    let z = transform[8] * point[0]
        + transform[9] * point[1]
        + transform[10] * point[2]
        + transform[11];
    [x, y, z]
}
