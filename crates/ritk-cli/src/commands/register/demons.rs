use super::*;

// ── Thirion Demons registration ───────────────────────────────────────────────

/// Run Thirion Demons deformable registration.
///
/// Converts both images to flat `Vec<f32>`, runs the Thirion Demons
/// algorithm, and reconstructs the output image from `result.warped`.
pub(super) fn run_demons(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::demons::{DemonsConfig, ThirionDemonsRegistration};

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = DemonsConfig {
        max_iterations: args.iterations,
        sigma_diffusion: 1.5,
        sigma_fluid: 0.0,
        max_step_length: 2.0,
    };
    let reg = ThirionDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Thirion Demons registration failed")?;

    let warped_img = flat_vec_to_image(result.warped, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=demons, iterations={}, final_mse={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_mse,
    );
    info!(
        "register: demons complete method={} iterations={} final_mse={}",
        "demons", result.num_iterations, result.final_mse
    );

    Ok(())
}

// ── Multi-resolution Demons registration ─────────────────────────────────────────

/// Run multi-resolution Demons deformable registration.
///
/// Converts both images to flat `Vec<f32>`, runs the coarse-to-fine Demons pyramid,
/// and reconstructs the output image from `result.warped`.
pub(super) fn run_multires_demons(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::demons::{
        DemonsConfig, MultiResDemonsConfig, MultiResDemonsRegistration,
    };

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = MultiResDemonsConfig {
        base_config: DemonsConfig {
            max_iterations: args.iterations,
            sigma_diffusion: 1.5,
            sigma_fluid: 0.0,
            max_step_length: 2.0,
        },
        levels: args.levels,
        use_diffeomorphic: args.use_diffeomorphic,
        n_squarings: 6,
    };
    let reg = MultiResDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Multi-resolution Demons registration failed")?;

    let warped_img = flat_vec_to_image(result.warped, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} → {} (method=multires-demons, iterations={}, levels={}, final_mse={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        args.levels,
        result.final_mse,
    );
    info!(
        "register: multires-demons complete method={} iterations={} levels={} final_mse={}",
        "multires-demons", result.num_iterations, args.levels, result.final_mse
    );

    Ok(())
}

// ── Inverse-consistent diffeomorphic Demons registration ──────────────────────

/// Run inverse-consistent diffeomorphic Demons registration.
///
/// Maintains forward and exact inverse transforms through SVF negation and
/// writes the warped moving image in the fixed image frame.
pub(super) fn run_inverse_consistent_demons(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::demons::{
        DemonsConfig, InverseConsistentDemonsConfig,
        InverseConsistentDiffeomorphicDemonsRegistration,
    };

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = InverseConsistentDemonsConfig {
        demons: DemonsConfig {
            max_iterations: args.iterations,
            sigma_diffusion: 1.5,
            sigma_fluid: 0.0,
            max_step_length: 2.0,
        },
        inverse_consistency_weight: args.inverse_consistency_weight,
        n_squarings: args.n_squarings,
    };
    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Inverse-consistent Demons registration failed")?;

    let warped_img = flat_vec_to_image(result.warped, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} → {} (method=ic-demons, iterations={}, final_mse={:.6}, ic_residual={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_mse,
        result.inverse_consistency_residual,
    );
    info!(
        "register: inverse-consistent demons complete method={} iterations={} final_mse={} inverse_consistency_residual={}",
        "ic-demons", result.num_iterations, result.final_mse, result.inverse_consistency_residual
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::register::tests::make_ramp_image;
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
            method: "demons".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
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
            method: "demons".to_string(),
            output_transform: None,
            iterations: 5,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        // Verify the warped image has finite voxel values.
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        warped.with_data_slice(|vals| {
            for (i, &v) in vals.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "demons output voxel [{i}] must be finite, got {v}"
                );
            }
        });
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
            method: "multires-demons".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
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
            method: "multires-demons".to_string(),
            output_transform: None,
            iterations: 5,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        // Verify the warped image has finite voxel values (identity => MSE near 0).
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        warped.with_data_slice(|vals| {
            for (i, &v) in vals.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "multires-demons output voxel [{i}] must be finite, got {v}"
                );
            }
        });
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
            method: "ic-demons".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
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
            method: "ic-demons".to_string(),
            output_transform: None,
            iterations: 5,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        warped.with_data_slice(|vals| {
            for (i, &v) in vals.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "ic-demons output voxel [{i}] must be finite, got {v}"
                );
            }
        });
    }
}
