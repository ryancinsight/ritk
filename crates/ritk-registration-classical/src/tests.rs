use crate::engine::ImageRegistration;
use crate::spatial::{apply_affine_perturbation, kabsch_algorithm, SpatialTransform};
use ndarray::{Array1, Array2};

#[test]
fn test_rigid_registration_landmarks() {
    let registration = ImageRegistration::default();

    // Create simple landmark sets (translated by [1, 2, 3])
    let fixed_landmarks =
        Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

    let moving_landmarks =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 3.0, 3.0]).unwrap();

    let result = registration
        .rigid_registration_landmarks(&fixed_landmarks, &moving_landmarks)
        .unwrap();

    // Check that we got a rigid body transform
    match result.spatial_transform {
        Some(SpatialTransform::RigidBody { translation, .. }) => {
            // Translation should be approximately [-1, -2, -3] to align centroids
            assert!((translation[0] + 1.0).abs() < 0.1);
            assert!((translation[1] + 2.0).abs() < 0.1);
            assert!((translation[2] + 3.0).abs() < 0.1);
        }
        _ => panic!("Expected RigidBody transform"),
    }

    // FRE should be small for perfect alignment
    assert!(result.quality_metrics.fre.unwrap() < 0.1);

    // Confidence should be high
    assert!(result.confidence > 0.9);
}

#[test]
fn test_temporal_synchronization() {
    let registration = ImageRegistration::default();

    // Create reference and target signals with known phase offset
    let n_samples = 1000;
    let sampling_rate = 1000.0; // 1 kHz
    let ref_signal = Array1::from_vec(
        (0..n_samples)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin())
            .collect(),
    );
    let target_signal = Array1::from_vec(
        (0..n_samples)
            .map(|i| {
                (2.0 * std::f64::consts::PI * i as f64 / 100.0 + std::f64::consts::PI / 4.0).sin()
            })
            .collect(),
    );

    let sync = registration
        .temporal_synchronization(&ref_signal, &target_signal, sampling_rate)
        .unwrap();

    // Phase offset should be reasonable (cross-correlation result)
    assert!(sync.phase_offset.abs() < 2.0 * std::f64::consts::PI);

    // Quality metrics should be computed
    assert!(sync.quality_metrics.rms_timing_error >= 0.0);
    assert!(sync.quality_metrics.phase_lock_stability >= 0.0);
    assert!(sync.quality_metrics.phase_lock_stability <= 1.0);
    assert!(sync.quality_metrics.sync_success_rate >= 0.0);
    assert!(sync.quality_metrics.sync_success_rate <= 1.0);
}

#[test]
fn test_registration_quality_metrics() {
    let registration = ImageRegistration::default();

    let fixed = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let moving = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    let result = registration
        .rigid_registration_landmarks(&fixed, &moving)
        .unwrap();

    // For identical point sets, FRE should be very small
    assert!(result.quality_metrics.fre.unwrap() < 1e-10);

    // Confidence should be very high
    assert!(result.confidence > 0.99);
}

#[test]
fn test_kabsch_identity_returns_identity_rotation() {
    // Two identical point clouds → optimal rotation must be identity
    let pts = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    let r = kabsch_algorithm(&pts, &pts).unwrap();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0_f64];
    for (ri, ei) in r.iter().zip(identity.iter()) {
        assert!((ri - ei).abs() < 1e-10, "Expected identity, got {r:?}");
    }
}

#[test]
fn test_kabsch_recovers_90_degree_rotation() {
    // Rotating around Z by 90°: (x,y,z) → (-y, x, z)
    //   R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    // fixed = original points, moving = R * fixed
    // kabsch_algorithm(fixed, moving) should return R

    let fixed =
        Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    // Apply Rz(90°) to get moving: (1,0,0)→(0,1,0), (0,1,0)→(-1,0,0), (0,0,1)→(0,0,1)
    let moving =
        Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();

    let r = kabsch_algorithm(&fixed, &moving).unwrap();

    // r should rotate moving → fixed: R_result * moving[i] ≈ fixed[i]
    // R_result should be Rz(-90°) = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    let tol = 1e-10;
    // Check by applying r to each moving point and comparing with fixed
    for i in 0..3 {
        let mp = [moving[[i, 0]], moving[[i, 1]], moving[[i, 2]]];
        let transformed = [
            r[0] * mp[0] + r[1] * mp[1] + r[2] * mp[2],
            r[3] * mp[0] + r[4] * mp[1] + r[5] * mp[2],
            r[6] * mp[0] + r[7] * mp[1] + r[8] * mp[2],
        ];
        let fp = [fixed[[i, 0]], fixed[[i, 1]], fixed[[i, 2]]];
        assert!((transformed[0] - fp[0]).abs() < tol, "row {i} x mismatch");
        assert!((transformed[1] - fp[1]).abs() < tol, "row {i} y mismatch");
        assert!((transformed[2] - fp[2]).abs() < tol, "row {i} z mismatch");
    }
}

#[test]
fn test_kabsch_rotation_is_orthogonal() {
    // Any output of kabsch_algorithm must be a valid rotation matrix: R Rᵀ = I, det R = +1
    let fixed = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let moving = Array2::from_shape_vec(
        (4, 3),
        vec![0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 1.0, 1.0],
    )
    .unwrap();

    let r = kabsch_algorithm(&fixed, &moving).unwrap();

    // Check R Rᵀ = I  (orthogonality)
    for row in 0..3 {
        for col in 0..3 {
            let dot: f64 = (0..3).map(|k| r[row * 3 + k] * r[col * 3 + k]).sum();
            let expected = if row == col { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-10, "R Rᵀ [{row},{col}] = {dot}");
        }
    }

    // Check det R = +1
    let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
        + r[2] * (r[3] * r[7] - r[4] * r[6]);
    assert!((det - 1.0).abs() < 1e-10, "det R = {det}, expected +1");
}

#[test]
fn test_affine_perturbation_identity_scale() {
    // Zero scale perturbation should not change scale (column norms unchanged)
    let identity_transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0_f64,
    ];
    let zero_perturbation = [0.0_f64; 9];
    let result = apply_affine_perturbation(&identity_transform, &zero_perturbation);

    for (r, e) in result.iter().zip(identity_transform.iter()) {
        assert!((r - e).abs() < 1e-12, "Expected identity, got {result:?}");
    }
}

#[test]
fn test_affine_perturbation_scale_applied() {
    // A +10% x-scale perturbation on identity should yield sx = 1.1
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0_f64,
    ];
    let perturbation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0_f64];
    let result = apply_affine_perturbation(&identity, &perturbation);

    // Column 0 norm (x-scale) should be 1.1
    let scale_x = (result[0].powi(2) + result[4].powi(2) + result[8].powi(2)).sqrt();
    assert!((scale_x - 1.1).abs() < 1e-10, "x-scale = {scale_x}");

    // Columns 1 and 2 should be unchanged (sy = sz = 1.0)
    let scale_y = (result[1].powi(2) + result[5].powi(2) + result[9].powi(2)).sqrt();
    let scale_z = (result[2].powi(2) + result[6].powi(2) + result[10].powi(2)).sqrt();
    assert!((scale_y - 1.0).abs() < 1e-10, "y-scale = {scale_y}");
    assert!((scale_z - 1.0).abs() < 1e-10, "z-scale = {scale_z}");
}
