use anyhow::Result;
use tracing::info;

use super::{read_image, write_image_inferred, Backend, FilterArgs};

// ── Gaussian filter ───────────────────────────────────────────────────────────
/// Apply a Gaussian smoothing filter to the input image and write the result.
///
/// The sigma value from `args.sigma` is applied uniformly along all three
/// spatial dimensions. The `GaussianFilter` implementation skips any
/// dimension whose sigma is ≤ 1e-6, so `--sigma 0.0` is a valid no-op.
pub(super) fn run_gaussian(args: &FilterArgs) -> Result<()> {
    use ritk_filter::GaussianFilter;
    use ritk_filter::GaussianSigma;

    let image = read_image(&args.input)?;

    // sigma ≤ 0 is documented as a no-op at the CLI level; skip the filter
    // and return the image unmodified rather than constructing a near-zero sigma.
    let filtered = if args.sigma > 0.0 {
        let sigma = GaussianSigma::new(args.sigma)
            .ok_or_else(|| anyhow::anyhow!("--sigma must be > 0, got {}", args.sigma))?;
        let filter: GaussianFilter<Backend> = GaussianFilter::new(vec![sigma; 3]);
        filter.apply(&image)
    } else {
        image
    };

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied gaussian (\u{03c3}={}) to {} \u{2192} {}",
        args.sigma,
        args.input.display(),
        args.output.display(),
    );
    info!(
        "filter: gaussian complete input={} output={} sigma={}",
        args.input.display(),
        args.output.display(),
        args.sigma
    );

    Ok(())
}

// ── N4 bias field correction ──────────────────────────────────────────────────
pub(super) fn run_n4_bias(args: &FilterArgs) -> Result<()> {
    use ritk_filter::bias::N4Config;
    use ritk_filter::N4BiasFieldCorrectionFilter;

    let image = read_image(&args.input)?;

    let config = N4Config {
        num_fitting_levels: args.levels,
        num_iterations: args.iterations,
        ..Default::default()
    };
    let filter = N4BiasFieldCorrectionFilter::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied n4-bias (levels={}, iters={}) to {} \u{2192} {}",
        args.levels,
        args.iterations,
        args.input.display(),
        args.output.display(),
    );
    info!(
        "filter: n4-bias complete input={} output={} levels={} iterations={}",
        args.input.display(),
        args.output.display(),
        args.levels,
        args.iterations
    );

    Ok(())
}

// ── Anisotropic diffusion ─────────────────────────────────────────────────────
pub(super) fn run_anisotropic(args: &FilterArgs) -> Result<()> {
    use ritk_filter::diffusion::{ConductanceFunction, DiffusionConfig};
    use ritk_filter::AnisotropicDiffusionFilter;
    use ritk_filter::ExponentialConductance;

    let image = read_image(&args.input)?;

    let config = DiffusionConfig {
        num_iterations: args.iterations,
        conductance: args.conductance as f32,
        time_step: 0.0625,
        function: ConductanceFunction::Exponential,
    };
    let filter = AnisotropicDiffusionFilter::<ExponentialConductance>::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied anisotropic (iters={}, K={}) to {} \u{2192} {}",
        args.iterations,
        args.conductance,
        args.input.display(),
        args.output.display(),
    );
    info!(
        "filter: anisotropic complete input={} output={} iterations={} conductance={}",
        args.input.display(),
        args.output.display(),
        args.iterations,
        args.conductance
    );

    Ok(())
}

// -- Curvature anisotropic diffusion ------------------------------------------
pub(super) fn run_curvature(args: &FilterArgs) -> Result<()> {
    use ritk_filter::diffusion::{CurvatureAnisotropicDiffusionFilter, CurvatureConfig};

    let image = read_image(&args.input)?;

    let config = CurvatureConfig {
        num_iterations: args.iterations,
        time_step: args.time_step as f32,
    };
    let filter = CurvatureAnisotropicDiffusionFilter::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied curvature (iters={}, dt={}) to {} -> {}",
        args.iterations,
        args.time_step,
        args.input.display(),
        args.output.display()
    );
    info!("filter: curvature complete");

    Ok(())
}

// -- Sato line filter ---------------------------------------------------------
pub(super) fn run_sato(args: &FilterArgs) -> Result<()> {
    use ritk_filter::vesselness::{SatoConfig, SatoLineFilter};

    let image = read_image(&args.input)?;

    let scales = args.scales.clone();
    let scales = if scales.is_empty() {
        vec![1.0, 2.0, 3.0]
    } else {
        scales
    };

    let config = SatoConfig {
        scales,
        alpha: args.alpha,
        polarity: ritk_filter::vesselness::VesselPolarity::Bright,
    };
    let filter = SatoLineFilter::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied sato (scales={:?}, alpha={}) to {} -> {}",
        filter.config.scales,
        args.alpha,
        args.input.display(),
        args.output.display()
    );
    info!("filter: sato complete");

    Ok(())
}

pub(super) fn run_discrete_gaussian(args: &FilterArgs) -> Result<()> {
    use ritk_filter::{DiscreteGaussianFilter, GaussianSigma};

    let image = read_image(&args.input)?;

    // variance < 0 is invalid; variance = 0 is identity (no smoothing applied).
    if args.variance < 0.0 {
        anyhow::bail!("--variance must be non-negative, got {}", args.variance);
    }
    if args.variance == 0.0 {
        write_image_inferred(&args.output, &image)?;
        println!(
            "Applied discrete-gaussian (variance=0.0: identity) to {} -> {}",
            args.input.display(),
            args.output.display()
        );
        return Ok(());
    }

    // CLI accepts variance (σ²); DiscreteGaussianFilter API takes sigma (σ).
    let sigma = GaussianSigma::new(args.variance.sqrt())
        .expect("invariant: sqrt of positive variance yields positive sigma");
    let filter = DiscreteGaussianFilter::<Backend>::new(vec![sigma])
        .with_maximum_error(args.maximum_error)
        .with_spacing_mode(args.spacing_mode);
    let filtered = filter.apply(&image);

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied discrete-gaussian (variance={}, maximum_error={}, spacing_mode={}) to {} -> {}",
        args.variance,
        args.maximum_error,
        args.spacing_mode,
        args.input.display(),
        args.output.display()
    );
    info!("filter: discrete-gaussian complete");

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::filter::{default_args, make_test_image};
    use ritk_filter::SpacingMode;
    use tempfile::tempdir;

    // ── Positive: Gaussian creates output file ────────────────────────────
    /// Applying the Gaussian filter must create the output file.
    #[test]
    fn test_filter_gaussian_creates_output_file() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("filtered.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        run_gaussian(&default_args(input.clone(), output.clone(), "gaussian")).unwrap();
        assert!(output.exists(), "output file must be created");
    }

    // ── Positive: Gaussian preserves shape ───────────────────────────────
    /// The output image must have the same voxel dimensions as the input.
    #[test]
    fn test_filter_gaussian_preserves_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("filtered.mha");

        ritk_io::write_metaimage(&input, &make_test_image()).unwrap();

        run_gaussian(&default_args(input.clone(), output.clone(), "gaussian")).unwrap();

        let result = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            result.shape(),
            [5, 5, 5],
            "shape must be preserved after Gaussian filtering"
        );
    }

    // ── Positive: Gaussian with sigma=0 is a no-op ──────────────────────
    /// `--sigma 0.0` must leave voxel values unchanged (GaussianFilter skips
    /// dimensions with σ ≤ 1e-6).
    #[test]
    fn test_filter_gaussian_sigma_zero_is_noop() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("filtered.mha");

        let original = make_test_image();
        let original_data: Vec<f32> = original
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        ritk_io::write_metaimage(&input, &original).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "gaussian");
        args.sigma = 0.0;
        run_gaussian(&args).unwrap();

        let result = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let result_data: Vec<f32> = result
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        // Sigma = 0 → no convolution → values must be identical after round-trip.
        // NIfTI/MetaImage round-trip may reorder axes; compare sums as a
        // scalar invariant that is permutation-independent.
        let orig_sum: f32 = original_data.iter().sum();
        let result_sum: f32 = result_data.iter().sum();
        assert!(
            (orig_sum - result_sum).abs() < 1e-3 * orig_sum.abs().max(1.0),
            "voxel sum must be preserved under \u{03c3}=0 Gaussian (orig={orig_sum}, result={result_sum})"
        );
    }

    // ── Positive: N4 bias-field correction creates output file ───────────
    #[test]
    fn test_filter_n4_applies_correction() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "n4-bias");
        args.levels = 1;
        args.iterations = 5;
        let result = run_n4_bias(&args);
        assert!(result.is_ok(), "n4-bias must succeed: {:?}", result.err());
        assert!(output.exists(), "n4-bias must write output file");
    }

    // ── Positive: anisotropic diffusion creates output file ──────────────
    #[test]
    fn test_filter_anisotropic_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "anisotropic");
        args.iterations = 5;
        let result = run_anisotropic(&args);
        assert!(
            result.is_ok(),
            "anisotropic must succeed: {:?}",
            result.err()
        );
        assert!(output.exists(), "anisotropic must write output file");
    }

    #[test]
    fn test_filter_curvature_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "curvature");
        args.iterations = 3;
        let result = run_curvature(&args);
        assert!(result.is_ok(), "curvature must succeed: {:?}", result.err());
        assert!(output.exists(), "curvature must write output file");
        let out_img = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out_img.shape(), [5, 5, 5], "output shape must match input");
    }

    #[test]
    fn test_filter_discrete_gaussian_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "discrete-gaussian");
        args.variance = 1.0;
        args.maximum_error = 0.01;
        args.spacing_mode = SpacingMode::Physical;
        let result = run_discrete_gaussian(&args);
        assert!(
            result.is_ok(),
            "discrete-gaussian must succeed: {:?}",
            result.err()
        );
        assert!(output.exists(), "discrete-gaussian must write output file");
        let out_img = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out_img.shape(), [5, 5, 5], "output shape must match input");
    }

    #[test]
    fn test_filter_discrete_gaussian_sigma_zero_variance_is_noop_on_sum() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("out.mha");

        let image = make_test_image();
        let input_sum: f32 = image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .copied()
            .sum();
        ritk_io::write_metaimage(&input, &image).unwrap();

        let mut args = default_args(input, output.clone(), "discrete-gaussian");
        args.variance = 0.0;
        let result = run_discrete_gaussian(&args);
        assert!(
            result.is_ok(),
            "discrete-gaussian zero variance must succeed: {:?}",
            result.err()
        );
        let out_img = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let output_sum: f32 = out_img
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .copied()
            .sum();
        assert!(
            (input_sum - output_sum).abs() < 1e-3 * input_sum.abs().max(1.0),
            "discrete-gaussian zero variance must preserve voxel sum (input={input_sum}, output={output_sum})"
        );
    }

    #[test]
    fn test_filter_sato_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "sato");
        args.scales = vec![1.0];
        let result = run_sato(&args);
        assert!(result.is_ok(), "sato must succeed: {:?}", result.err());
        assert!(output.exists(), "sato must write output file");
        let out_img = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out_img.shape(), [5, 5, 5], "output shape must match input");
    }
}
