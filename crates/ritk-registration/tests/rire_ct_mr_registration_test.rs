//! RIRE CT/MR T1 registration integration tests — smoke test & index.
//!
//! This file serves as the entry point and index for the RIRE CT/MR T1
//! registration test suite. The full tests have been partitioned into
//! dedicated files by test group:
//!
//! - **`rire_ct_mr_rigid_test.rs`** — Pure math tests (rotation orthogonality,
//!   8-corner verification, inverse roundtrip, perturbation roundtrip).
//!   These require no external data and run in milliseconds.
//!
//! - **`rire_ct_mr_diffeomorphic_test.rs`** — Image integration tests
//!   (MHA metadata loading, GT alignment NCC, inverse shift recovery).
//!   These require RIRE test data and are tagged `#[ignore]`.
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

use common::{apply_rigid, mat3_det, mat3_mul, mat3_transpose, GT_ROT, GT_TRANS};

/// Smoke test: verify that the ground-truth rotation matrix is proper
/// orthogonal (`R · R^T ≈ I`, `det(R) ≈ +1`). This is a lightweight check
/// that the shared constants in `common::` are correctly imported.
#[test]
fn test_rire_gt_rotation_smoke() {
    let rrt = mat3_mul(&GT_ROT, &mat3_transpose(&GT_ROT));
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (rrt[i * 3 + j] - expected).abs() < 1e-6,
                "R·R^T[{},{}] = {:.6e}, expected {:.1}",
                i,
                j,
                rrt[i * 3 + j],
                expected
            );
        }
    }
    let det = mat3_det(&GT_ROT);
    assert!(
        (det - 1.0).abs() < 1e-6,
        "det(GT_ROT) = {:.6e}, expected +1.0",
        det
    );
}

/// Smoke test: verify that `apply_rigid` produces a finite output for the
/// ground-truth transform applied to a known point.
#[test]
fn test_rire_apply_rigid_smoke() {
    let p = [100.0, 200.0, 50.0];
    let result = apply_rigid(&GT_ROT, &GT_TRANS, &p);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "apply_rigid produced non-finite result: {:?}",
        result
    );
}
