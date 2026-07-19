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
        sigma_diffusion: Some(GaussianSigma::new_unchecked(1.5)),
        sigma_fluid: None,
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
            sigma_diffusion: Some(GaussianSigma::new_unchecked(1.5)),
            sigma_fluid: None,
            max_step_length: 2.0,
        },
        levels: args.levels,
        variant: args.variant,
        n_squarings: 6,
    };
    let reg = MultiResDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Multi-resolution Demons registration failed")?;

    let warped_img = flat_vec_to_image(result.warped, fixed_shape, &fixed_img);
    super::super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=multires-demons, iterations={}, levels={}, final_mse={:.6})",
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
            sigma_diffusion: Some(GaussianSigma::new_unchecked(1.5)),
            sigma_fluid: None,
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
        "Registered {} \u{2192} {} (method=ic-demons, iterations={}, final_mse={:.6}, ic_residual={:.6})",
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
mod tests_demons;
