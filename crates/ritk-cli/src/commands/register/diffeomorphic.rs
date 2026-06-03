use super::*;

// ── SyN diffeomorphic registration ────────────────────────────────────────────

/// Run greedy SyN diffeomorphic registration.
///
/// Converts both images to flat `Vec<f32>`, runs SyN with local CC metric,
/// and reconstructs the output image from `result.warped_moving` (the moving
/// image warped towards the fixed image's midpoint).
pub(super) fn run_syn(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = SyNConfig {
        max_iterations: args.iterations,
        sigma_smooth: 3.0,
        cc_window_radius: 2,
        gradient_step: 0.25,
        ..Default::default()
    };
    let reg = SyNRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "SyN registration failed")?;

    // SyN produces warped_moving: moving image warped to the midpoint.
    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=syn, iterations={}, final_cc={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_cc,
    );
    info!(
        "register: syn complete method={} iterations={} final_cc={}",
        "syn", result.num_iterations, result.final_cc
    );

    Ok(())
}

// ── BSpline FFD registration ───────────────────────────────────────────────

/// Run B-Spline Free-Form Deformation registration.
///
/// Rueckert et al. (1999): Multi-resolution control-lattice FFD with NCC metric
/// and bending-energy regularization. Control-point spacing halves at each
/// successive level.
pub(super) fn run_bspline_ffd(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration};

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = BSplineFFDConfig {
        initial_control_spacing: [
            args.control_spacing,
            args.control_spacing,
            args.control_spacing,
        ],
        num_levels: args.levels,
        max_iterations_per_level: args.iterations,
        learning_rate: args.learning_rate,
        regularization_weight: args.regularization_weight,
        convergence_threshold: args.convergence_threshold,
    };
    let result = BSplineFFDRegistration::register(
        &fixed_vals,
        &moving_vals,
        fixed_shape,
        [1.0, 1.0, 1.0],
        &config,
    )
    .with_context(|| "BSpline FFD registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=bspline-ffd, levels={}, iterations={}, final_metric={:.6})",
        args.moving.display(),
        args.output.display(),
        args.levels,
        result.num_iterations,
        result.final_metric,
    );
    info!(
        "register: bspline-ffd complete method={} levels={} iterations={} final_metric={}",
        "bspline-ffd", args.levels, result.num_iterations, result.final_metric
    );

    Ok(())
}

// ── Multi-resolution SyN registration ────────────────────────────────────────────

/// Run Multi-Resolution SyN diffeomorphic registration.
///
/// Avants & Gee coarse-to-fine pyramid SyN: Gaussian downsampling with
/// level-doubling velocity fields and optional inverse consistency enforcement.
pub(super) fn run_multires_syn(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::diffeomorphic::multires_syn::{
        MultiResSyNConfig, MultiResSyNRegistration,
    };

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = MultiResSyNConfig {
        num_levels: args.levels,
        iterations_per_level: vec![args.iterations; args.levels],
        sigma_smooth: args.sigma_fixed,
        convergence_threshold: args.convergence_threshold,
        convergence_window: 10,
        n_squarings: 6,
        cc_window_radius: args.cc_radius,
        enforce_inverse_consistency: args.inverse_consistency,
        gradient_step: 0.25,
    };
    let reg = MultiResSyNRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Multi-resolution SyN registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=multires-syn, levels={}, iterations={}, final_cc={:.6})",
        args.moving.display(),
        args.output.display(),
        args.levels,
        result.num_iterations,
        result.final_cc,
    );
    info!(
        "register: multires-syn complete method={} levels={} iterations={} final_cc={}",
        "multires-syn", args.levels, result.num_iterations, result.final_cc
    );

    Ok(())
}

// ── BSpline SyN registration ──────────────────────────────────────────────

/// Run BSpline SyN diffeomorphic registration.
///
/// Symmetric diffeomorphic registration with B-spline-parameterized velocity
/// fields. Provides intrinsic smoothness from B-spline basis and bending-energy
/// regularization.
pub(super) fn run_bspline_syn(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::diffeomorphic::bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration};

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = BSplineSyNConfig {
        max_iterations: args.iterations,
        control_spacing: [
            args.control_spacing,
            args.control_spacing,
            args.control_spacing,
        ],
        sigma_smooth: args.sigma_fixed,
        convergence_threshold: args.convergence_threshold,
        convergence_window: 10,
        n_squarings: 6,
        cc_window_radius: args.cc_radius,
        regularization_weight: args.regularization_weight,
        gradient_step: 0.25,
    };
    let reg = BSplineSyNRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "BSpline SyN registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=bspline-syn, iterations={}, final_cc={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_cc,
    );
    info!(
        "register: bspline-syn complete method={} iterations={} final_cc={}",
        "bspline-syn", result.num_iterations, result.final_cc
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::register::tests::make_ramp_image;
    use tempfile::tempdir;

    /// Build default `RegisterArgs` for diffeomorphic-family tests.
    ///
    /// Writes the ramp image to both `fixed` and `moving` paths inside `dir`.
    fn default_args(dir: &std::path::Path, method: &str, output_name: &str) -> RegisterArgs {
        let fixed_path = dir.join("fixed.nii");
        let moving_path = dir.join("moving.nii");
        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();
        RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: dir.join(output_name),
            method: method.to_string(),
            output_transform: None,
            iterations: 2,
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
        }
    }

    /// Run a diffeomorphic method with default args and return the temp dir + output path.
    ///
    /// The caller must keep the `TempDir` alive until done reading the output file.
    fn run_method(method: &str, output_name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempdir().unwrap();
        let args = default_args(dir.path(), method, output_name);
        let output = args.output.clone();
        run(args).unwrap_or_else(|_| panic!("{method} must succeed"));
        (dir, output)
    }

    // ── Positive: syn creates output file ─────────────────────────────────

    /// Running `syn` on identical fixed/moving images must produce a warped
    /// output file whose shape matches the input.
    #[test]
    fn test_register_syn_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let args = default_args(dir.path(), "syn", "warped.nii");
        run(args).unwrap();
        let output = dir.path().join("warped.nii");
        assert!(output.exists(), "syn warped output file must be created");
        let warped = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            warped.shape(),
            [4, 4, 4],
            "syn warped image shape must match fixed image shape"
        );
    }

    // ── Positive: syn identity registration produces finite voxels ────────

    /// When fixed == moving, the SyN output voxels must all be finite.
    #[test]
    fn test_register_syn_identity_finite_voxels() {
        let (_dir, output) = run_method("syn", "warped.nii");
        let warped = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        warped.with_data_slice(|vals| {
            for (i, &v) in vals.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "syn output voxel [{i}] must be finite, got {v}"
                );
            }
        });
    }

    // ── BSpline FFD ──────────────────────────────────────────────────────────────

    #[test]
    fn test_register_bspline_ffd_creates_output_with_correct_shape() {
        let (_dir, output) = run_method("bspline-ffd", "output.nii");
        assert!(output.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }
    #[test]
    fn test_register_bspline_ffd_identity_finite_voxels() {
        let (_dir, output) = run_method("bspline-ffd", "output.nii");
        let out = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        out.with_data_slice(|vals| {
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "all output voxels must be finite"
            );
        });
    }

    // ── Multi-resolution SyN ─────────────────────────────────────────────────────

    #[test]
    fn test_register_multires_syn_creates_output_with_correct_shape() {
        let (_dir, output) = run_method("multires-syn", "output.nii");
        assert!(output.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }
    #[test]
    fn test_register_multires_syn_identity_finite_voxels() {
        let (_dir, output) = run_method("multires-syn", "output.nii");
        let out = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        out.with_data_slice(|vals| {
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "all output voxels must be finite"
            );
        });
    }

    // ── BSpline SyN ─────────────────────────────────────────────────────────────

    #[test]
    fn test_register_bspline_syn_creates_output_with_correct_shape() {
        let (_dir, output) = run_method("bspline-syn", "output.nii");
        assert!(output.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }
    #[test]
    fn test_register_bspline_syn_identity_finite_voxels() {
        let (_dir, output) = run_method("bspline-syn", "output.nii");
        let out = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        out.with_data_slice(|vals| {
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "all output voxels must be finite"
            );
        });
    }
}
