use super::*;

// ── MI-based registration (rigid / affine) ────────────────────────────────────

/// Run rigid-mi or affine-mi registration via the classical MI engine.
pub(super) fn run_mi_registration(args: &RegisterArgs) -> Result<()> {
    // ── 1. Read images ─────────────────────────────────────────────────────
    let fixed_img = super::super::read_image(&args.fixed)?;
    let moving_img = super::super::read_image(&args.moving)?;

    // ── 2. Optional pre-registration Gaussian smoothing ───────────────────
    // GaussianFilter skips any dimension whose sigma ≤ 1e-6, so sigma=0.0
    // is a safe no-op.
    let (fixed_img, moving_img) = if args.sigma_fixed > 1e-12 {
        let filter: GaussianFilter<Backend> = GaussianFilter::new(vec![args.sigma_fixed; 3]);
        (filter.apply(&fixed_img), filter.apply(&moving_img))
    } else {
        (fixed_img, moving_img)
    };

    // ── 3. Convert images to ndarray::Array3<f64> ─────────────────────────
    let fixed_arr = image_to_array3(&fixed_img);
    let moving_arr = image_to_array3(&moving_img);

    // ── 4. Build registration engine with user-supplied iteration budget ───
    let config = ClassicalConfig {
        max_iterations: args.iterations,
        tolerance: 1e-6,
        step_multiplier: 1.0,
    };
    let metric = MutualInformationMetric::default();
    let reg = ImageRegistration::with_config(config, metric);

    // Identity 4×4 homogeneous matrix as the initial transform.
    let identity: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    // ── 5. Run registration ────────────────────────────────────────────────
    let result = match args.method.as_str() {
        "rigid-mi" => reg
            .rigid_registration_mutual_info(&moving_arr, &fixed_arr, &identity)
            .with_context(|| "rigid MI registration failed")?,
        "affine-mi" => reg
            .affine_registration_mutual_info(&moving_arr, &fixed_arr, &identity)
            .with_context(|| "affine MI registration failed")?,
        _ => unreachable!("run_mi_registration called with non-MI method"),
    };

    // ── 6. Warp moving image with estimated transform ──────────────────────
    // Re-read the original (un-smoothed) moving image for warping so the
    // output preserves the full-resolution signal.
    let moving_orig = super::super::read_image(&args.moving)?;
    let moving_orig_arr = image_to_array3(&moving_orig);
    let warped_arr = spatial::apply_transform(&moving_orig_arr, &result.transform);

    // ── 7. Convert warped array back to Image and write output ─────────────
    // Spatial metadata comes from the fixed image (the output lives in the
    // fixed image's coordinate frame).
    let warped_img = array3_to_image(warped_arr, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    // ── 8. Optionally write transform JSON ─────────────────────────────────
    if let Some(ref tx_path) = args.output_transform {
        let json = serde_json::to_string_pretty(&result.transform)
            .context("Failed to serialise transform to JSON")?;
        std::fs::write(tx_path, &json)
            .with_context(|| format!("Failed to write transform to {}", tx_path.display()))?;
        info!("register: transform written path={}", tx_path.display());
    }

    // ── 9. Print summary ───────────────────────────────────────────────────
    let q = &result.quality;
    println!(
        "Registered {} \u{2192} {} (method={}, iterations={}, converged={}, MI={:.6}, cost={:.6})",
        args.moving.display(),
        args.output.display(),
        args.method,
        q.iterations,
        q.converged,
        q.mutual_information,
        q.final_cost,
    );
    info!(
        "register: MI registration complete method={} iterations={} converged={} mi={} final_cost={}",
        args.method, q.iterations, q.converged, q.mutual_information, q.final_cost
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::register::tests::make_ramp_image;
    use ritk_registration::demons::DemonsVariant;
    use tempfile::tempdir;

    // ── Positive: rigid-mi creates output file ────────────────────────────

    /// Running `rigid-mi` on identical fixed/moving images must produce a
    /// warped output file whose shape matches the input.
    #[test]
    fn test_register_rigid_mi_creates_output_with_correct_shape() {
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
            method: "rigid-mi".to_string(),
            output_transform: None,
            // Use very few iterations so the test completes quickly.
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            variant: DemonsVariant::Classic,
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

        assert!(output_path.exists(), "warped output file must be created");
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(
            warped.shape(),
            [4, 4, 4],
            "warped image shape must match fixed image shape"
        );
    }

    // ── Positive: affine-mi creates output file ───────────────────────────

    /// Running `affine-mi` must produce a warped output file.
    #[test]
    fn test_register_affine_mi_creates_output() {
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
            method: "affine-mi".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            variant: DemonsVariant::Classic,
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
            "affine warped output file must be created"
        );
    }

    // ── Positive: --output-transform writes valid JSON ────────────────────

    /// When `--output-transform` is supplied the file must exist and parse
    /// as a JSON array of exactly 16 finite float values.
    #[test]
    fn test_register_writes_transform_json_with_16_elements() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");
        let tx_path = dir.path().join("transform.json");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path,
            method: "rigid-mi".to_string(),
            output_transform: Some(tx_path.clone()),
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            variant: DemonsVariant::Classic,
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

        assert!(tx_path.exists(), "transform JSON file must be created");
        let json_str = std::fs::read_to_string(&tx_path).unwrap();
        let values: Vec<f64> =
            serde_json::from_str(&json_str).expect("transform file must be valid JSON");
        assert_eq!(
            values.len(),
            16,
            "transform must contain exactly 16 elements (row-major 4\u{d7}4 matrix)"
        );
        for (i, &v) in values.iter().enumerate() {
            assert!(
                v.is_finite(),
                "transform element [{i}] must be finite, got {v}"
            );
        }
    }

    // ── Positive: identity on identical images preserves voxel sum ────────

    /// When fixed == moving and the registration converges close to identity,
    /// the voxel sum of the warped image must be close to the input sum.
    /// (Exact equality is not required because nearest-neighbour warp may
    /// drop boundary voxels.)
    #[test]
    fn test_register_identity_moving_preserves_voxel_sum_approximately() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        let original_sum: f32 = image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();

        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "rigid-mi".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            variant: DemonsVariant::Classic,
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
        let warped_sum: f32 = warped
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();

        // Allow up to 5 % relative deviation to account for boundary voxels
        // that may fall outside the source volume after warping.
        let rel_err = ((original_sum - warped_sum).abs()) / original_sum.abs().max(1.0);
        assert!(
            rel_err < 0.05,
            "warped voxel sum {warped_sum:.1} must be within 5% of original {original_sum:.1}"
        );
    }
}
