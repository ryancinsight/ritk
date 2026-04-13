//! Classical spatial transformation implementations.
//!
//! This module provides non-ML spatial transforms for registration, including
//! the Kabsch SVD algorithm for rigid-body landmark registration and affine
//! transformation operations.

use nalgebra::{Matrix3, Vector3};
use ndarray::{Array2, Array3};
use thiserror::Error;

/// Errors for classical spatial transform operations.
#[derive(Error, Debug)]
pub enum SpatialError {
    #[error("Invalid point set: {0}")]
    InvalidPointSet(String),
    #[error("SVD did not converge: {0}")]
    SvdConvergence(String),
    #[error("Invalid transform matrix: {0}")]
    InvalidTransform(String),
}

/// Classical spatial transform variants.
#[derive(Debug, Clone)]
pub enum SpatialTransform {
    /// Rigid body transform: 3x3 rotation + 3x1 translation
    RigidBody {
        /// Row-major 9-element rotation matrix
        rotation: [f64; 9],
        /// Translation vector [tx, ty, tz]
        translation: [f64; 3],
    },
    /// Affine transform: 3x4 matrix (12 DOF)
    Affine {
        /// Row-major 12-element affine matrix [r00, r01, r02, t0, ...]
        matrix: [f64; 12],
    },
    /// Non-rigid transform via deformation field
    NonRigid {
        /// Deformation field tensor
        deformation_field: Array3<f64>,
    },
}

/// Compute the centroid of a point set.
pub(crate) fn compute_centroid(points: &Array2<f64>) -> Vector3<f64> {
    let n = points.nrows() as f64;
    let mut sum = Vector3::zeros();
    for i in 0..points.nrows() {
        let row = points.row(i);
        sum += Vector3::new(row[0], row[1], row[2]);
    }
    sum / n
}

/// Center points by subtracting their centroid.
pub(crate) fn center_points(points: &Array2<f64>, centroid: &Vector3<f64>) -> Array2<f64> {
    let mut centered = points.clone();
    for mut row in centered.rows_mut() {
        row[0] -= centroid[0];
        row[1] -= centroid[1];
        row[2] -= centroid[2];
    }
    centered
}

/// Compute optimal rigid-body rotation using Kabsch SVD algorithm.
#[rustfmt::skip]
pub(crate) fn kabsch_algorithm(
    fixed_centered: &Array2<f64>,
    moving_centered: &Array2<f64>,
) -> Result<[f64; 9], SpatialError> {
    if fixed_centered.nrows() != moving_centered.nrows() {
        return Err(SpatialError::InvalidPointSet(
            "Point sets must have same number of rows".to_string(),
        ));
    }
    if fixed_centered.ncols() != 3 || moving_centered.ncols() != 3 {
        return Err(SpatialError::InvalidPointSet(
            "Points must have 3 columns".to_string(),
        ));
    }

    let mut h = Matrix3::zeros();
    for i in 0..fixed_centered.nrows() {
        let pf = fixed_centered.row(i);
        let pm = moving_centered.row(i);
        h += Matrix3::new(
            pm[0] * pf[0], pm[0] * pf[1], pm[0] * pf[2],
            pm[1] * pf[0], pm[1] * pf[1], pm[1] * pf[2],
            pm[2] * pf[0], pm[2] * pf[1], pm[2] * pf[2],
        );
    }

    let svd = h.svd(true, true);
    let u = svd.u.ok_or_else(|| SpatialError::SvdConvergence("U not found".to_string()))?;
    let v_t = svd.v_t.ok_or_else(|| SpatialError::SvdConvergence("V not found".to_string()))?;

    let mut r = v_t.transpose() * u.transpose();

    if r.determinant() < 0.0 {
        let mut v_corrected = v_t.transpose();
        v_corrected.set_column(2, &(-v_t.transpose().column(2)));
        r = v_corrected * u.transpose();
    }

    let rotation = [
        r[(0, 0)], r[(0, 1)], r[(0, 2)],
        r[(1, 0)], r[(1, 1)], r[(1, 2)],
        r[(2, 0)], r[(2, 1)], r[(2, 2)],
    ];

    Ok(rotation)
}

/// Build 4x4 homogeneous transformation matrix from rotation and translation.
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

/// Extract spatial transform from 4x4 homogeneous matrix.
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

/// Transform a single point using a 4x4 homogeneous matrix.
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

const EULER_STEP: f64 = 0.01;
const TRANSLATION_STEP: f64 = 1.0;
const SCALE_STEP: f64 = 0.02;

/// Generate perturbations for rigid-body (6-DOF) optimization.
pub(crate) fn generate_transform_perturbations() -> [[f64; 6]; 12] {
    [
        [EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, EULER_STEP, 0.0, 0.0, 0.0, 0.0],
        [0.0, -EULER_STEP, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, EULER_STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, -EULER_STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, TRANSLATION_STEP, 0.0, 0.0],
        [0.0, 0.0, 0.0, -TRANSLATION_STEP, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, TRANSLATION_STEP, 0.0],
        [0.0, 0.0, 0.0, 0.0, -TRANSLATION_STEP, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, TRANSLATION_STEP],
        [0.0, 0.0, 0.0, 0.0, 0.0, -TRANSLATION_STEP],
    ]
}

/// Apply rigid-body perturbation to a 4x4 transformation matrix.
pub(crate) fn apply_transform_perturbation(
    current: &[f64; 16],
    perturbation: &[f64; 6],
) -> [f64; 16] {
    let [dtheta_x, dtheta_y, dtheta_z, dtx, dty, dtz] = *perturbation;

    let rx = Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        dtheta_x.cos(),
        -dtheta_x.sin(),
        0.0,
        dtheta_x.sin(),
        dtheta_x.cos(),
    );
    let ry = Matrix3::new(
        dtheta_y.cos(),
        0.0,
        dtheta_y.sin(),
        0.0,
        1.0,
        0.0,
        -dtheta_y.sin(),
        0.0,
        dtheta_y.cos(),
    );
    let rz = Matrix3::new(
        dtheta_z.cos(),
        -dtheta_z.sin(),
        0.0,
        dtheta_z.sin(),
        dtheta_z.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let r_perturb = rx * ry * rz;

    let current_r = Matrix3::new(
        current[0],
        current[1],
        current[2],
        current[4],
        current[5],
        current[6],
        current[8],
        current[9],
        current[10],
    );
    let current_t = Vector3::new(current[3], current[7], current[11]);

    let new_r = current_r * r_perturb;
    let new_t = current_t + Vector3::new(dtx, dty, dtz);

    build_homogeneous_matrix(
        &[
            new_r[(0, 0)],
            new_r[(0, 1)],
            new_r[(0, 2)],
            new_r[(1, 0)],
            new_r[(1, 1)],
            new_r[(1, 2)],
            new_r[(2, 0)],
            new_r[(2, 1)],
            new_r[(2, 2)],
        ],
        &[new_t[0], new_t[1], new_t[2]],
    )
}

/// Generate perturbations for affine (9-DOF) optimization.
pub(crate) fn generate_affine_perturbations() -> [[f64; 9]; 18] {
    [
        [EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -EULER_STEP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, TRANSLATION_STEP, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -TRANSLATION_STEP, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, TRANSLATION_STEP, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -TRANSLATION_STEP, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, TRANSLATION_STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -TRANSLATION_STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SCALE_STEP, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -SCALE_STEP, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SCALE_STEP, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -SCALE_STEP, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SCALE_STEP],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -SCALE_STEP],
    ]
}

/// Apply affine perturbation to a 4x4 transformation matrix.
pub(crate) fn apply_affine_perturbation(current: &[f64; 16], perturbation: &[f64; 9]) -> [f64; 16] {
    let [dtheta_x, dtheta_y, dtheta_z, dtx, dty, dtz, dsx, dsy, dsz] = *perturbation;

    let r = Matrix3::new(
        current[0],
        current[1],
        current[2],
        current[4],
        current[5],
        current[6],
        current[8],
        current[9],
        current[10],
    );
    let t = Vector3::new(current[3], current[7], current[11]);

    let s = Matrix3::new(
        1.0 + dsx,
        0.0,
        0.0,
        0.0,
        1.0 + dsy,
        0.0,
        0.0,
        0.0,
        1.0 + dsz,
    );

    let rx = Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        dtheta_x.cos(),
        -dtheta_x.sin(),
        0.0,
        dtheta_x.sin(),
        dtheta_x.cos(),
    );
    let ry = Matrix3::new(
        dtheta_y.cos(),
        0.0,
        dtheta_y.sin(),
        0.0,
        1.0,
        0.0,
        -dtheta_y.sin(),
        0.0,
        dtheta_y.cos(),
    );
    let rz = Matrix3::new(
        dtheta_z.cos(),
        -dtheta_z.sin(),
        0.0,
        dtheta_z.sin(),
        dtheta_z.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let new_r = r * s * rx * ry * rz;
    let new_t = t + Vector3::new(dtx, dty, dtz);

    [
        new_r[(0, 0)],
        new_r[(0, 1)],
        new_r[(0, 2)],
        new_t[0],
        new_r[(1, 0)],
        new_r[(1, 1)],
        new_r[(1, 2)],
        new_t[1],
        new_r[(2, 0)],
        new_r[(2, 1)],
        new_r[(2, 2)],
        new_t[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// Compute Fiducial Registration Error (FRE) between fixed and moving points.
pub(crate) fn compute_fre(
    fixed: &Array2<f64>,
    moving: &Array2<f64>,
    rotation: &[f64; 9],
    translation: &[f64; 3],
) -> f64 {
    let r = Matrix3::new(
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
        rotation[4],
        rotation[5],
        rotation[6],
        rotation[7],
        rotation[8],
    );
    let t: Vector3<f64> = Vector3::new(translation[0], translation[1], translation[2]);

    let mut sum_squared_error = 0.0;
    for i in 0..fixed.nrows() {
        let p_fixed: Vector3<f64> = Vector3::new(fixed.row(i)[0], fixed.row(i)[1], fixed.row(i)[2]);
        let p_moving: Vector3<f64> =
            Vector3::new(moving.row(i)[0], moving.row(i)[1], moving.row(i)[2]);
        let p_transformed: Vector3<f64> = r * p_moving;
        let p_transformed = p_transformed + t;
        let error: Vector3<f64> = p_fixed - p_transformed;
        sum_squared_error += error.dot(&error);
    }

    (sum_squared_error / fixed.nrows() as f64).sqrt()
}

/// Apply a 4x4 homogeneous transformation to a 3D volume.
pub fn apply_transform(volume: &Array3<f64>, transform: &[f64; 16]) -> Array3<f64> {
    let (depth, height, width) = volume.dim();
    let mut result = Array3::zeros((depth, height, width));

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let src = transform_point(&[x as f64, y as f64, z as f64], transform);
                let src_x = src[0].round() as isize;
                let src_y = src[1].round() as isize;
                let src_z = src[2].round() as isize;

                if src_x >= 0
                    && src_x < width as isize
                    && src_y >= 0
                    && src_y < height as isize
                    && src_z >= 0
                    && src_z < depth as isize
                {
                    result[[z, y, x]] = volume[[src_z as usize, src_y as usize, src_x as usize]];
                }
            }
        }
    }

    result
}

/// Apply transform to volume and return the transformed volume.
#[allow(dead_code)]
pub(crate) fn apply_transform_to_volume(
    volume: &Array3<f64>,
    transform: &[f64; 16],
) -> Array3<f64> {
    apply_transform(volume, transform)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kabsch_identity() {
        let fixed =
            Array2::from_shape_vec((3, 3), vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
        let rotation = kabsch_algorithm(&fixed, &fixed).unwrap();

        let expected = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        for (r, e) in rotation.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_build_homogeneous_matrix() {
        let rotation = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let translation = [1., 2., 3.];
        let matrix = build_homogeneous_matrix(&rotation, &translation);

        assert_eq!(matrix[3], 1.);
        assert_eq!(matrix[7], 2.);
        assert_eq!(matrix[11], 3.);
        assert_eq!(matrix[15], 1.);
    }

    #[test]
    fn test_transform_point() {
        let point = [1., 0., 0.];
        let transform = [
            1., 0., 0., 10., 0., 1., 0., 20., 0., 0., 1., 30., 0., 0., 0., 1.,
        ];
        let result = transform_point(&point, &transform);

        assert!((result[0] - 11.).abs() < 1e-10);
        assert!((result[1] - 20.).abs() < 1e-10);
        assert!((result[2] - 30.).abs() < 1e-10);
    }
}
