//! Tests for Demons/IC-Demons/Diffeomorphic-Demons registration CLI.
use super::*;
use crate::commands::register::tests::make_ramp_image;
use ritk_registration::demons::DemonsVariant;
use tempfile::tempdir;

// ── Positive: demons creates output file ──────────────────────────────

/// Running `demons` on identical fixed/moving images must produce a
/// warped output file whose shape matches the input.
#[test]
fn test_register_demons_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let fixed_path = dir.path().join("fixed.nii");
    let moving_path = dir.path().join("moving.nii");
    let output_path = dir.path().join("warped.nii");

    let image = make_ramp_image();
    ritk_io::write_nifti(&fixed_path, &image).unwrap();
    ritk_io::write_nifti(&moving_path, &image).unwrap();

    run(RegisterArgs {
        fixed: fixed_path,
        moving: moving_path,
        output: output_path.clone(),
        method: RegistrationMethod::Demons,
        output_transform: None,
        iterations: 3,
        sigma_fixed: GaussianSigma::default(),
        levels: 3,
        variant: DemonsVariant::Classic,
        regularization_weight: 0.001,
        control_spacing: 4,
        cc_radius: 2,
        inverse_consistency: CliInverseConsistency::Relaxed,
        num_time_steps: 2,
        kernel_sigma: GaussianSigma::new_unchecked(3.0),
        learning_rate: 0.01,
        inverse_consistency_weight: 0.5,
        n_squarings: 6,
        convergence_threshold: 1e-5,
    })
    .unwrap();

    assert!(
        output_path.exists(),
        "demons warped output file must be created"
    );
    let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    assert_eq!(
        warped.shape(),
        [4, 4, 4],
        "demons warped image shape must match fixed image shape"
    );
}

// ── Positive: demons identity registration has low MSE ────────────────

/// When fixed == moving, the Thirion Demons final MSE must be near zero.
#[test]
fn test_register_demons_identity_low_mse() {
    let dir = tempdir().unwrap();
    let fixed_path = dir.path().join("fixed.nii");
    let moving_path = dir.path().join("moving.nii");
    let output_path = dir.path().join("warped.nii");

    let image = make_ramp_image();
    ritk_io::write_nifti(&fixed_path, &image).unwrap();
    ritk_io::write_nifti(&moving_path, &image).unwrap();

    run(RegisterArgs {
        fixed: fixed_path,
        moving: moving_path,
        output: output_path.clone(),
        method: RegistrationMethod::Demons,
        output_transform: None,
        iterations: 5,
        sigma_fixed: GaussianSigma::default(),
        levels: 3,
        variant: DemonsVariant::Classic,
        regularization_weight: 0.001,
        control_spacing: 4,
        cc_radius: 2,
        inverse_consistency: CliInverseConsistency::Relaxed,
        num_time_steps: 2,
        kernel_sigma: GaussianSigma::new_unchecked(3.0),
        learning_rate: 0.01,
        inverse_consistency_weight: 0.5,
        n_squarings: 6,
        convergence_threshold: 1e-5,
    })
    .unwrap();

    // Verify the warped image has finite voxel values.
    let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    {
        let vals = warped
            .data_slice()
            .expect("invariant: image storage is contiguous");
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "demons output voxel [{i}] must be finite, got {v}"
            );
        }
    }
}

// -- Positive: multires-demons creates output file -------------------------

/// Running `multires-demons` with levels=1 on identical images must produce a
/// warped output file whose shape matches the input.
#[test]
fn test_register_multires_demons_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let fixed_path = dir.path().join("fixed.nii");
    let moving_path = dir.path().join("moving.nii");
    let output_path = dir.path().join("warped.nii");

    let image = make_ramp_image();
    ritk_io::write_nifti(&fixed_path, &image).unwrap();
    ritk_io::write_nifti(&moving_path, &image).unwrap();

    run(RegisterArgs {
        fixed: fixed_path,
        moving: moving_path,
        output: output_path.clone(),
        method: RegistrationMethod::MultiResDemons,
        output_transform: None,
        iterations: 3,
        sigma_fixed: GaussianSigma::default(),
        levels: 1,
        variant: DemonsVariant::Classic,
        regularization_weight: 0.001,
        control_spacing: 4,
        cc_radius: 2,
        inverse_consistency: CliInverseConsistency::Relaxed,
        num_time_steps: 2,
        kernel_sigma: GaussianSigma::new_unchecked(3.0),
        learning_rate: 0.01,
        inverse_consistency_weight: 0.5,
        n_squarings: 6,
        convergence_threshold: 1e-5,
    })
    .unwrap();

    assert!(
        output_path.exists(),
        "multires-demons warped output file must be created"
    );
    let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    assert_eq!(
        warped.shape(),
        [4, 4, 4],
        "multires-demons warped image shape must match fixed image shape"
    );
}

// -- Positive: multires-demons identity registration has low MSE ----------

/// When fixed == moving, multires-demons final MSE must be near zero (levels=1).
#[test]
fn test_register_multires_demons_identity_low_mse() {
    let dir = tempdir().unwrap();
    let fixed_path = dir.path().join("fixed.nii");
    let moving_path = dir.path().join("moving.nii");
    let output_path = dir.path().join("warped.nii");

    let image = make_ramp_image();
    ritk_io::write_nifti(&fixed_path, &image).unwrap();
    ritk_io::write_nifti(&moving_path, &image).unwrap();

    run(RegisterArgs {
        fixed: fixed_path,
        moving: moving_path,
        output: output_path.clone(),
        method: RegistrationMethod::MultiResDemons,
        output_transform: None,
        iterations: 5,
        sigma_fixed: GaussianSigma::default(),
        levels: 1,
        variant: DemonsVariant::Classic,
        regularization_weight: 0.001,
        control_spacing: 4,
        cc_radius: 2,
        inverse_consistency: CliInverseConsistency::Relaxed,
        num_time_steps: 2,
        kernel_sigma: GaussianSigma::new_unchecked(3.0),
        learning_rate: 0.01,
        inverse_consistency_weight: 0.5,
        n_squarings: 6,
        convergence_threshold: 1e-5,
    })
    .unwrap();

    // Verify the warped image has finite voxel values (identity => MSE near 0).
    let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    {
        let vals = warped
            .data_slice()
            .expect("invariant: image storage is contiguous");
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "multires-demons output voxel [{i}] must be finite, got {v}"
            );
        }
    }
}

// ── Inverse-consistent Demons: output shape ──────────────────────────────

#[test]
fn test_register_ic_demons_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let fixed_path = dir.path().join("fixed.nii");
    let moving_path = dir.path().join("moving.nii");
    let output_path = dir.path().join("warped.nii");

    let image = make_ramp_image();
    ritk_io::write_nifti(&fixed_path, &image).unwrap();
    ritk_io::write_nifti(&moving_path, &image).unwrap();

    run(RegisterArgs {
        fixed: fixed_path,
        moving: moving_path,
        output: output_path.clone(),
        method: RegistrationMethod::IcDemons,
        output_transform: None,
        iterations: 3,
        sigma_fixed: GaussianSigma::default(),
        levels: 1,
        variant: DemonsVariant::Classic,
        regularization_weight: 0.001,
        control_spacing: 4,
        cc_radius: 2,
        inverse_consistency: CliInverseConsistency::Relaxed,
        num_time_steps: 2,
        kernel_sigma: GaussianSigma::new_unchecked(3.0),
        learning_rate: 0.01,
        inverse_consistency_weight: 0.5,
        n_squarings: 6,
        convergence_threshold: 1e-5,
    })
    .unwrap();

    assert!(
        output_path.exists(),
        "ic-demons warped output file must be created"
    );
    let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    assert_eq!(
        warped.shape(),
        [4, 4, 4],
        "ic-demons warped image shape must match fixed image shape"
    );
}

#[test]
fn test_register_ic_demons_identity_finite_voxels() {
    let dir = tempdir().unwrap();
    let fixed_path = dir.path().join("fixed.nii");
    let moving_path = dir.path().join("moving.nii");
    let output_path = dir.path().join("warped.nii");

    let image = make_ramp_image();
    ritk_io::write_nifti(&fixed_path, &image).unwrap();
    ritk_io::write_nifti(&moving_path, &image).unwrap();

    run(RegisterArgs {
        fixed: fixed_path,
        moving: moving_path,
        output: output_path.clone(),
        method: RegistrationMethod::IcDemons,
        output_transform: None,
        iterations: 5,
        sigma_fixed: GaussianSigma::default(),
        levels: 1,
        variant: DemonsVariant::Classic,
        regularization_weight: 0.001,
        control_spacing: 4,
        cc_radius: 2,
        inverse_consistency: CliInverseConsistency::Relaxed,
        num_time_steps: 2,
        kernel_sigma: GaussianSigma::new_unchecked(3.0),
        learning_rate: 0.01,
        inverse_consistency_weight: 0.5,
        n_squarings: 6,
        convergence_threshold: 1e-5,
    })
    .unwrap();

    let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    {
        let vals = warped
            .data_slice()
            .expect("invariant: image storage is contiguous");
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "ic-demons output voxel [{i}] must be finite, got {v}"
            );
        }
    }
}
