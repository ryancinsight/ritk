use super::*;

// ── LDDMM registration ────────────────────────────────────────────────────

/// Run LDDMM (Large Deformation Diffeomorphic Metric Mapping) registration.
///
/// Geodesic shooting via EPDiff with a Gaussian RKHS kernel (Miller 2006).
/// The initial velocity field parameterizes the geodesic; the deformation
/// phi_1 at t=1 warps the moving image.
pub(super) fn run_lddmm(args: &RegisterArgs) -> Result<()> {
    use ritk_core::filter::GaussianSigma;
    use ritk_registration::lddmm::{LddmmConfig, LddmmRegistration};

    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let kernel_sigma = GaussianSigma::new(args.kernel_sigma)
        .ok_or_else(|| anyhow::anyhow!("--kernel-sigma must be > 0, got {}", args.kernel_sigma))?;
    let config = LddmmConfig {
        max_iterations: args.iterations,
        num_time_steps: args.num_time_steps,
        kernel_sigma,
        learning_rate: args.learning_rate,
        ..Default::default()
    };
    let reg = LddmmRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "LDDMM registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=lddmm, iterations={}, time_steps={}, final_metric={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        args.num_time_steps,
        result.final_metric,
    );
    info!(
        "register: lddmm complete method={} iterations={} time_steps={} final_metric={}",
        "lddmm", result.num_iterations, args.num_time_steps, result.final_metric
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::register::tests::make_ramp_image;
    use ritk_registration::demons::DemonsVariant;
    use tempfile::tempdir;

    // ── LDDMM: output shape ────────────────────────────────────────────────────────────

    #[test]
    fn test_register_lddmm_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "lddmm".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            variant: DemonsVariant::Classic,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: CliInverseConsistency::Relaxed,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("lddmm must succeed");

        assert!(output_path.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }

    #[test]
    fn test_register_lddmm_identity_finite_voxels() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "lddmm".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            variant: DemonsVariant::Classic,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: CliInverseConsistency::Relaxed,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("lddmm must succeed");

        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        out.with_data_slice(|vals| {
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "all output voxels must be finite"
            );
        });
    }

    // ── Boundary: image_to_array3 round-trip preserves values ────────────

    /// Converting an image to `Array3<f64>` and back must preserve voxel
    /// values within f32 precision.
    #[test]
    fn test_image_to_array3_and_back_preserves_values() {
        let image = make_ramp_image();
        let arr = image_to_array3(&image);

        // Verify shape.
        assert_eq!(arr.dim(), (4, 4, 4), "array shape must match image shape");

        // Verify values: flat index i → value i * 4.0.
        let flat: Vec<f64> = arr.iter().copied().collect();
        for (i, &v) in flat.iter().enumerate() {
            let expected = i as f64 * 4.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "element [{i}]: expected {expected}, got {v}"
            );
        }

        // Convert back and verify sum is preserved.
        let reconstructed = array3_to_image(arr, &image);
        let orig_sum: f32 = image.with_data_slice(|s| s.iter().copied().sum());
        let recon_sum: f32 = reconstructed.with_data_slice(|s| s.iter().copied().sum());
        assert!(
            (orig_sum - recon_sum).abs() < 1e-3,
            "voxel sum must be preserved: orig={orig_sum}, recon={recon_sum}"
        );
    }

    // ── Boundary: image_to_flat_vec round-trip preserves values ───────────

    /// Converting an image to flat vec and back must preserve voxel values.
    #[test]
    fn test_image_to_flat_vec_and_back_preserves_values() {
        let image = make_ramp_image();
        let (data, shape) = image_to_flat_vec(&image);

        assert_eq!(shape, [4, 4, 4], "flat_vec shape must match image shape");
        assert_eq!(data.len(), 64, "flat_vec length must equal total voxels");

        // Verify individual values.
        for (i, &v) in data.iter().enumerate() {
            let expected = i as f32 * 4.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "flat_vec element [{i}]: expected {expected}, got {v}"
            );
        }

        // Round-trip.
        let reconstructed = flat_vec_to_image(data, shape, &image);
        let recon_vals: Vec<f32> = reconstructed.data_slice().into_owned();
        let orig_vals: Vec<f32> = image.data_slice().into_owned();
        for (i, (&o, &r)) in orig_vals.iter().zip(recon_vals.iter()).enumerate() {
            assert!(
                (o - r).abs() < 1e-6,
                "round-trip voxel [{i}]: orig={o}, recon={r}"
            );
        }
    }
}
