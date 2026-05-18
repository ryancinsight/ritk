//! Affine (9-DOF) perturbation generation and application.
//!
//! Perturbation vector layout: [dθ_x, dθ_y, dθ_z, dt_x, dt_y, dt_z, ds_x, ds_y, ds_z].

use nalgebra::{Matrix3, Vector3};

use super::{EULER_STEP, SCALE_STEP, TRANSLATION_STEP};

/// Generate the 18 canonical ±1-step 9-DOF affine perturbations.
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

/// Apply a 9-DOF affine perturbation to a 4×4 homogeneous matrix.
///
/// Composition: R_new = R_current · S(ds) · R_x(dθ_x) · R_y(dθ_y) · R_z(dθ_z)
/// where S = diag(1+ds_x, 1+ds_y, 1+ds_z).
/// Translation: t_new = t_current + [dt_x, dt_y, dt_z].
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
