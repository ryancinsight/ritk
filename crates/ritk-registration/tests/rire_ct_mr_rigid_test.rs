//! RIRE CT/MR T1 rigid-transform pure math tests.
//!
//! These tests exercise the rigid-transform math helpers (apply, inverse,
//! orthogonality) and the 8-corner fiducial verification from the RIRE
//! standard file `ct_T1.standard`. They require no external data and run in
//! milliseconds.
//!
//! # RIRE provenance
//!
//! Images and the standard transformation were provided as part of the project
//! *Retrospective Image Registration Evaluation* (RIRE), National Institutes
//! of Health, Project Number 8R01EB002124-03, Principal Investigator
//! J. Michael Fitzpatrick, Vanderbilt University, Nashville TN.
//! Data site: <https://rire.insight-journal.org/>
//! License: Creative Commons Attribution 3.0 United States.

mod common;

use common::{
    apply_rigid, mat3_det, mat3_mul, mat3_transpose, rigid_inverse, GT_ROT, GT_TRANS, RIRE_CORNERS,
};
use std::f64::consts::PI;

// ── Group 1 — Pure math tests ────────────────────────────────────────────────

/// # Specification
///
/// The ground-truth rotation must be a proper orthogonal matrix (rotation group
/// SO(3)): `R · R^T = I` and `det(R) = +1`.
///
/// Verification:
/// - Each diagonal element of `R · R^T` must equal `1.0` to within `1e-9`.
/// - Each off-diagonal element of `R · R^T` must equal `0.0` to within `1e-9`.
/// - `det(R)` must equal `+1.0` to within `1e-9`.
///
/// A failure here would indicate a transcription error in `GT_ROT` and would
/// invalidate all downstream tests that rely on the ground-truth transform.
#[test]
fn test_rire_gt_rotation_matrix_is_proper_orthogonal() {
    let rrt = mat3_mul(&GT_ROT, &mat3_transpose(&GT_ROT));
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = rrt[i * 3 + j];
            assert!(
                (actual - expected).abs() < 1e-9,
                "R·R^T[{},{}] = {:.12e}, expected {:.1} (tol 1e-9)",
                i,
                j,
                actual,
                expected
            );
        }
    }
    let det = mat3_det(&GT_ROT);
    assert!(
        (det - 1.0).abs() < 1e-9,
        "det(GT_ROT) = {:.12e}, expected +1.0 (tol 1e-9)",
        det
    );
}

/// # Specification
///
/// The GT transform must reproduce all 8 RIRE `ct_T1.standard` corners with
/// residual < 0.001 mm (fiducial-based gold standard).
///
/// Each source point `(src_x, src_y, src_z)` transformed by `T(p) = R·p + t`
/// must land within 0.001 mm of the corresponding destination point
/// `(dst_x, dst_y, dst_z)` from `ct_T1.standard`.
///
/// The ground-truth parameters in `GT_ROT` / `GT_TRANS` are documented to
/// reproduce all 8 corners with a maximum residual of 0.000176 mm, well below
/// the RIRE acceptance threshold of 0.01 mm.
#[test]
fn test_rire_gt_eight_corner_verification() {
    for (i, corner) in RIRE_CORNERS.iter().enumerate() {
        let [src_x, src_y, src_z, dst_x, dst_y, dst_z] = *corner;
        let result = apply_rigid(&GT_ROT, &GT_TRANS, &[src_x, src_y, src_z]);
        let dx = result[0] - dst_x;
        let dy = result[1] - dst_y;
        let dz = result[2] - dst_z;
        let residual = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            residual < 0.001,
            "Corner {}: residual = {:.9} mm (>= 0.001 mm)\n\
             src=({:.4}, {:.4}, {:.4})\n\
             got=({:.6}, {:.6}, {:.6})\n\
             exp=({:.4}, {:.4}, {:.4})",
            i + 1,
            residual,
            src_x,
            src_y,
            src_z,
            result[0],
            result[1],
            result[2],
            dst_x,
            dst_y,
            dst_z
        );
    }
}

/// # Specification
///
/// The inverse rigid transform `T^{-1}(T(p)) = p` must hold to within
/// 1e-6 mm at all probe points.
///
/// Validates the `rigid_inverse()` formula:
/// - `R^{-1} = R^T`
/// - `t^{-1} = −R^T · t`
///
/// The 12 non-trivial probe points are distributed throughout the CT physical
/// volume (x, y ∈ [0, 334] mm; z ∈ [0, 112] mm).
///
/// ## Tolerance rationale
///
/// `GT_ROT` is tabulated to 9 decimal places, so `R^T` is not the bit-exact
/// inverse of `R`. The measured orthogonality residual `‖R·Rᵀ − I‖ ≈ 1e-9`
/// combined with `|p| ≈ 50–334 mm` yields a roundtrip error of order
/// `|p| × 1e-9 ≈ 1e-7 mm`. The tolerance 1e-6 mm is far below any
/// physically meaningful threshold (sub-nanometer) while comfortably
/// accommodating finite-precision tabulation.
#[test]
fn test_rire_gt_inverse_roundtrip_exact() {
    let probe_points: [[f64; 3]; 12] = [
        [50.0, 50.0, 20.0],
        [100.0, 200.0, 40.0],
        [333.0, 0.0, 0.0],
        [0.0, 333.0, 112.0],
        [166.0, 166.0, 56.0],
        [10.0, 300.0, 80.0],
        [250.0, 100.0, 10.0],
        [330.0, 330.0, 100.0],
        [0.0, 0.0, 112.0],
        [333.9870, 0.0, 0.0],
        [0.0, 333.9870, 0.0],
        [333.9870, 333.9870, 112.0],
    ];
    let (r_inv, t_inv) = rigid_inverse(&GT_ROT, &GT_TRANS);
    for (i, p) in probe_points.iter().enumerate() {
        let p_fwd = apply_rigid(&GT_ROT, &GT_TRANS, p);
        let p_back = apply_rigid(&r_inv, &t_inv, &p_fwd);
        let dx = p_back[0] - p[0];
        let dy = p_back[1] - p[1];
        let dz = p_back[2] - p[2];
        let deviation = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            deviation < 1e-6,
            "Probe {}: T^{{-1}}(T(p)) roundtrip deviation = {:.3e} mm (>= 1e-6)\n\
             p = ({:.6}, {:.6}, {:.6})\n\
             T(p) = ({:.6}, {:.6}, {:.6})\n\
             T^{{-1}} = ({:.6}, {:.6}, {:.6})",
            i,
            deviation,
            p[0],
            p[1],
            p[2],
            p_fwd[0],
            p_fwd[1],
            p_fwd[2],
            p_back[0],
            p_back[1],
            p_back[2]
        );
    }
}

/// # Specification
///
/// Any rigid perturbation composed with its inverse must be the identity to
/// machine precision. This validates the `rigid_inverse()` function for
/// transforms other than the ground truth.
///
/// The perturbation used here:
/// - Rotation: 0.05 rad about the z-axis.
/// - Translation: [3.5, -7.2, 2.1] mm.
///
/// For 10 probe points uniformly distributed in the CT physical volume, the
/// roundtrip `P^{-1}(P(p))` must recover `p` to within 1e-9 mm.
#[test]
fn test_rire_perturbation_and_inverse_math_roundtrip() {
    // 0.05 rad rotation about z-axis (row-major):
    //  [ cos(a), -sin(a), 0 ]
    //  [ sin(a),  cos(a), 0 ]
    //  [      0,       0, 1 ]
    let a: f64 = 0.05;
    let rot_z: [f64; 9] = [a.cos(), -a.sin(), 0.0, a.sin(), a.cos(), 0.0, 0.0, 0.0, 1.0];
    let trans: [f64; 3] = [3.5, -7.2, 2.1];
    let (r_inv, t_inv) = rigid_inverse(&rot_z, &trans);

    // Verify the perturbation rotation is itself proper orthogonal.
    let det = mat3_det(&rot_z);
    assert!(
        (det - 1.0).abs() < 1e-12,
        "Perturbation rotation det = {:.12e}, expected +1.0",
        det
    );

    let probe_points: [[f64; 3]; 10] = [
        [0.0, 0.0, 0.0],
        [100.0, 50.0, 30.0],
        [200.0, 150.0, 60.0],
        [300.0, 250.0, 90.0],
        [50.0, 300.0, 10.0],
        [250.0, 10.0, 100.0],
        [150.0, 150.0, 50.0],
        [333.9870, 333.9870, 112.0],
        [10.0, 10.0, 5.0],
        [320.0, 280.0, 95.0],
    ];
    for (i, p) in probe_points.iter().enumerate() {
        let p_fwd = apply_rigid(&rot_z, &trans, p);
        let p_back = apply_rigid(&r_inv, &t_inv, &p_fwd);
        let dx = p_back[0] - p[0];
        let dy = p_back[1] - p[1];
        let dz = p_back[2] - p[2];
        let deviation = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            deviation < 1e-9,
            "Perturbation probe {}: roundtrip deviation = {:.3e} mm (>= 1e-9)\n\
             p = ({:.6}, {:.6}, {:.6})\n\
             P(p) = ({:.6}, {:.6}, {:.6})\n\
             back = ({:.6}, {:.6}, {:.6})",
            i,
            deviation,
            p[0],
            p[1],
            p[2],
            p_fwd[0],
            p_fwd[1],
            p_fwd[2],
            p_back[0],
            p_back[1],
            p_back[2]
        );
    }
    // Suppress unused import warning for PI (used in documentation context).
    let _ = PI;
}
