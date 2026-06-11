//! Rigid-body (6-DOF) perturbation generation and application.
//!
//! Perturbation vector layout: [dθ_x, dθ_y, dθ_z, dt_x, dt_y, dt_z].

use nalgebra::{Matrix3, Vector3};

use super::transform::build_homogeneous_matrix;
use super::{EULER_STEP, TRANSLATION_STEP};
use crate::types::AffineTransform;

/// Generate the 12 canonical ±1-step 6-DOF rigid perturbations.
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

/// Apply a 6-DOF rigid perturbation to a 4×4 homogeneous matrix.
///
/// Rotation composition: R_new = R_current · R_x(dθ_x) · R_y(dθ_y) · R_z(dθ_z).
/// Translation composition: t_new = t_current + [dt_x, dt_y, dt_z].
pub(crate) fn apply_transform_perturbation(
    current: &AffineTransform,
    perturbation: &[f64; 6],
) -> AffineTransform {
    let [dtheta_x, dtheta_y, dtheta_z, dtx, dty, dtz] = *perturbation;
    let c = &current.0;

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

    let current_r = Matrix3::new(c[0], c[1], c[2], c[4], c[5], c[6], c[8], c[9], c[10]);
    let current_t = Vector3::new(c[3], c[7], c[11]);

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
