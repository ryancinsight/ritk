
path = r"crates/ritk-cli/src/commands/filter.rs"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

run_curvature = """
// ── Curvature anisotropic diffusion ─────────────────────────────────────────────────

/// Apply curvature anisotropic diffusion (Alvarez et al. 1992, mean curvature motion).
///
/// Evolves each voxel by the mean curvature of its level set surface:
///   ∂I/∂t = |∇I| · div(∇I / |∇I|)
///
/// Δt ≤ 1/6 is required for stability with unit spacing.
fn run_curvature(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::diffusion::{CurvatureAnisotropicDiffusionFilter, CurvatureConfig};

    let image = read_image(&args.input)?;
    let config = CurvatureConfig {
        num_iterations: args.iterations,
        time_step: args.time_step as f32,
    };
    let filter = CurvatureAnisotropicDiffusionFilter::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied curvature (iters={}, dt={}) to {} → {}",
        args.iterations,
        args.time_step,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        iterations = args.iterations,
        time_step = args.time_step,
        "filter: curvature complete"
    );

    Ok(())
}

// ── Sato line filter ──────────────────────────────────────────────────────────────────────────────

/// Apply the Sato multi-scale line filter for curvilinear structure detection.
///
/// Computes the per-voxel maximum Hessian-eigenvalue-based tubularity response
/// across all provided scales (Sato et al. 1998).
fn run_sato(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::vesselness::{SatoConfig, SatoLineFilter};

    let image = read_image(&args.input)?;

    let scales: Vec<f64> = args
        .scales
        .split('')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    let scales = if scales.is_empty() {
        vec![1.0, 2.0, 3.0]
    } else {
        scales
    };

    let config = SatoConfig {
        scales: scales.clone(),
        alpha: args.alpha,
        bright_tubes: true,
    };
    let filter = SatoLineFilter::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied sato (scales={:?}, α={}) to {} → {}",
        scales,
        args.alpha,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        alpha = args.alpha,
        "filter: sato complete"
    );

    Ok(())
}

"""
