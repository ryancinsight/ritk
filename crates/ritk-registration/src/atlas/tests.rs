//! Tests for atlas registration.

use super::*;
use crate::diffeomorphic::multires_syn::InverseConsistency;

/// Minimal SyN config for fast tests on small images.
fn test_syn_config() -> MultiResSyNConfig {
    MultiResSyNConfig {
        num_levels: 1,
        iterations_per_level: vec![5],
        sigma_smooth: 1.0,
        convergence_threshold: 1e-6,
        convergence_window: 3,
        n_squarings: 2,
        cc_window_radius: 1,
        gradient_step: 0.25,
        enforce_inverse_consistency: InverseConsistency::Relaxed,
    }
}

fn test_atlas_config() -> AtlasConfig {
    AtlasConfig {
        max_iterations: 3,
        convergence_threshold: 1e-3,
        syn_config: test_syn_config(),
    }
}

// â”€â”€ Positive tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Two identical constant subjects.
///
/// Analytical expectation: Tâ° = c. Registration of constant c to
/// constant c yields zero velocity fields (CC gradient is zero when
/// image gradient is zero). Mean of warped subjects = c. Mean
/// velocity = 0, so sharpening is identity. RMS change = 0 â†’
/// convergence in one iteration. Template = c everywhere.
#[test]
fn two_identical_constant_subjects_template_equals_constant() {
    let dims = [4, 4, 4];
    let n = 64;
    let val = 3.0f32;
    let s1 = vec![val; n];
    let s2 = vec![val; n];
    let subjects: Vec<&[f32]> = vec![&s1, &s2];
    let reg = AtlasRegistration::new(test_atlas_config());
    let result = reg.build_atlas(&subjects, dims, [1.0, 1.0, 1.0]).unwrap();
    // Template must be val at every voxel.
    for (i, &v) in result.template.iter().enumerate() {
        assert!(
            (v - val).abs() < 1e-4,
            "template[{}] = {} deviates from expected {}",
            i,
            v,
            val
        );
    }
    // Convergence in 1 outer iteration (RMS = 0 < threshold).
    assert_eq!(result.num_iterations, 1);
    assert!(result.convergence_history[0] < 1e-3);
    // One result per subject.
    assert_eq!(result.subject_results.len(), 2);
    // CC for constant images is 0 (no local variance).
    for sr in &result.subject_results {
        assert!(sr.final_cc.abs() < 1e-6, "final_cc = {}", sr.final_cc);
    }
}

/// Three uniform images with distinct values.
///
/// Analytical expectation: Tâ° = mean(2, 4, 6) = 4. Registration of a
/// uniform image to another uniform image produces zero displacement
/// (gradient is zero). Mean of warped subjects = mean of originals = 4.
/// Sharpening is identity (zero velocity). RMS change = 0 â†’ converge
/// in 1 iteration. Template â‰ˆ 4 everywhere.
#[test]
fn three_uniform_images_template_is_mean() {
    let dims = [4, 4, 4];
    let n = 64;
    let s1 = vec![2.0f32; n];
    let s2 = vec![4.0f32; n];
    let s3 = vec![6.0f32; n];
    let subjects: Vec<&[f32]> = vec![&s1, &s2, &s3];
    let expected = 4.0f32;
    let reg = AtlasRegistration::new(test_atlas_config());
    let result = reg.build_atlas(&subjects, dims, [1.0, 1.0, 1.0]).unwrap();
    for (i, &v) in result.template.iter().enumerate() {
        assert!(
            (v - expected).abs() < 0.1,
            "template[{}] = {} deviates from expected {}",
            i,
            v,
            expected
        );
    }
    assert_eq!(result.subject_results.len(), 3);
    assert!(result.convergence_history[0] < 0.1);
}

// â”€â”€ Boundary tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Single subject: template equals that subject.
///
/// Tâ° = Iâ‚. Registration of Iâ‚ to Iâ‚ â†’ identity. Template
/// unchanged. RMS = 0 â†’ converge in 1 iteration.
#[test]
fn single_subject_template_equals_subject() {
    let dims = [4, 4, 4];
    let n = 64;
    let val = 5.0f32;
    let s = vec![val; n];
    let subjects: Vec<&[f32]> = vec![&s];
    let reg = AtlasRegistration::new(test_atlas_config());
    let result = reg.build_atlas(&subjects, dims, [1.0, 1.0, 1.0]).unwrap();
    for (i, &v) in result.template.iter().enumerate() {
        assert!(
            (v - val).abs() < 1e-4,
            "template[{}] = {} deviates from expected {}",
            i,
            v,
            val
        );
    }
    assert_eq!(result.num_iterations, 1);
    assert_eq!(result.subject_results.len(), 1);
}

// â”€â”€ Negative tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Empty subjects slice returns `InvalidConfiguration`.
#[test]
fn empty_subjects_returns_error() {
    let subjects: Vec<&[f32]> = vec![];
    let reg = AtlasRegistration::new(test_atlas_config());
    let err = reg
        .build_atlas(&subjects, [4, 4, 4], [1.0, 1.0, 1.0])
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::InvalidConfiguration(_)),
        "expected InvalidConfiguration, got {:?}",
        err
    );
}

/// Subject with wrong length returns `DimensionMismatch`.
#[test]
fn dimension_mismatch_returns_error() {
    let s1 = vec![1.0f32; 64]; // 4*4*4
    let s2 = vec![1.0f32; 27]; // wrong
    let subjects: Vec<&[f32]> = vec![&s1, &s2];
    let reg = AtlasRegistration::new(test_atlas_config());
    let err = reg
        .build_atlas(&subjects, [4, 4, 4], [1.0, 1.0, 1.0])
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// `max_iterations = 0` returns `InvalidConfiguration`.
#[test]
fn zero_max_iterations_returns_error() {
    let s = vec![1.0f32; 64];
    let subjects: Vec<&[f32]> = vec![&s];
    let mut cfg = test_atlas_config();
    cfg.max_iterations = 0;
    let reg = AtlasRegistration::new(cfg);
    let err = reg
        .build_atlas(&subjects, [4, 4, 4], [1.0, 1.0, 1.0])
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::InvalidConfiguration(_)),
        "expected InvalidConfiguration, got {:?}",
        err
    );
}

/// Zero-volume dims returns `InvalidConfiguration`.
#[test]
fn zero_dims_returns_error() {
    let s = vec![1.0f32; 0];
    let subjects: Vec<&[f32]> = vec![&s];
    let reg = AtlasRegistration::new(test_atlas_config());
    let err = reg
        .build_atlas(&subjects, [0, 4, 4], [1.0, 1.0, 1.0])
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::InvalidConfiguration(_)),
        "expected InvalidConfiguration, got {:?}",
        err
    );
}
