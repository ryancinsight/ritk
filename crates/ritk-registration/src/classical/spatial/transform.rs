//! `SpatialTransform` variants and 4×4 homogeneous-matrix helpers.

use ndarray::Array3;

use super::error::SpatialError;
use crate::types::AffineTransform;

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
pub(crate) fn build_homogeneous_matrix(
    rotation: &[f64; 9],
    translation: &[f64; 3],
) -> AffineTransform {
    AffineTransform([
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
    ])
}

/// Extract `SpatialTransform` from a 4×4 homogeneous matrix.
///
/// Determines rigid vs affine by checking whether all three column norms
/// are within 1% of unity.
pub(crate) fn extract_spatial_transform(
    matrix: &AffineTransform,
) -> Result<SpatialTransform, SpatialError> {
    let m = &matrix.0;
    let scale_x = (m[0].powi(2) + m[1].powi(2) + m[2].powi(2)).sqrt();
    let scale_y = (m[4].powi(2) + m[5].powi(2) + m[6].powi(2)).sqrt();
    let scale_z = (m[8].powi(2) + m[9].powi(2) + m[10].powi(2)).sqrt();

    let is_rigid = (scale_x - 1.0).abs() < 0.01
        && (scale_y - 1.0).abs() < 0.01
        && (scale_z - 1.0).abs() < 0.01;

    if is_rigid {
        let rotation = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
        let translation = [m[3], m[7], m[11]];
        Ok(SpatialTransform::RigidBody {
            rotation,
            translation,
        })
    } else {
        let affine_matrix = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11],
        ];
        Ok(SpatialTransform::Affine {
            matrix: affine_matrix,
        })
    }
}

/// Transform a single point using a 4×4 homogeneous matrix.
pub(crate) fn transform_point(point: &[f64; 3], transform: &AffineTransform) -> [f64; 3] {
    let t = &transform.0;
    let x = t[0] * point[0] + t[1] * point[1] + t[2] * point[2] + t[3];
    let y = t[4] * point[0] + t[5] * point[1] + t[6] * point[2] + t[7];
    let z = t[8] * point[0] + t[9] * point[1] + t[10] * point[2] + t[11];
    [x, y, z]
}
