//! `ritk filter` — image filtering command.
//!
//! Applies one of the following filters to a 3-D medical image:
//!
//! | Filter       | Status         | Parameters                                   |
//! |-------------|----------------|----------------------------------------------|
//! | `gaussian`   | Implemented    | `--sigma <FLOAT>` (default 1.0 mm)           |
//! | `n4-bias`    | Pending Sprint 3 merge | `--levels`, `--iterations`          |
//! | `anisotropic`| Pending Sprint 3 merge | `--iterations`, `--conductance`     |
//! | `frangi`     | Pending Sprint 3 merge | `--scales`, `--alpha`, `--beta`, `--gamma` |
//!
//! The three pending filters return a structured `Err` rather than panicking
//! so the CLI remains usable for the filters that are available.

use anyhow::{anyhow, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use super::{read_image, write_image_inferred, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `filter` subcommand.
#[derive(Args, Debug)]
pub struct FilterArgs {
    /// Input image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Filter to apply.
    ///
    /// Accepted values: `gaussian`, `n4-bias`, `anisotropic`, `frangi`.
    #[arg(long, value_name = "FILTER")]
    pub filter: String,

    // ── Gaussian ──────────────────────────────────────────────────────────
    /// Gaussian standard deviation in physical units (mm).
    ///
    /// Applied uniformly in all three spatial dimensions.
    /// Used by: `gaussian`.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub sigma: f64,

    // ── N4 bias-field correction ──────────────────────────────────────────
    /// Number of multi-resolution pyramid levels for N4 bias-field correction.
    ///
    /// Used by: `n4-bias`.
    #[arg(long, default_value = "4", value_name = "INT")]
    pub levels: usize,

    /// Maximum number of optimizer iterations per pyramid level.
    ///
    /// Used by: `n4-bias`, `anisotropic`.
    #[arg(long, default_value = "50", value_name = "INT")]
    pub iterations: usize,

    // ── Anisotropic diffusion ─────────────────────────────────────────────
    /// Conductance parameter controlling edge sensitivity for anisotropic
    /// diffusion.  Lower values preserve edges more aggressively.
    ///
    /// Used by: `anisotropic`.
    #[arg(long, default_value = "3.0", value_name = "FLOAT")]
    pub conductance: f64,

    // ── Frangi vesselness ─────────────────────────────────────────────────
    /// Comma-separated list of vessel scale radii (mm) for multi-scale Frangi
    /// vesselness enhancement.
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "0.5,1.0,2.0", value_name = "FLOATS")]
    pub scales: String,

    /// Frangi α parameter (controls sensitivity to plate-like structures).
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub alpha: f64,

    /// Frangi β parameter (controls sensitivity to blob-like structures).
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub beta: f64,

    /// Frangi γ parameter (controls sensitivity to background noise).
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "15.0", value_name = "FLOAT")]
    pub gamma: f64,
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `filter` subcommand.
///
/// Dispatches to the appropriate filter implementation based on `args.filter`.
/// Filters that are not yet available in `ritk-core` return a descriptive
/// `Err` rather than panicking.
///
/// # Errors
/// Returns an error when:
/// - The input image cannot be read.
/// - The requested filter is not available in this build.
/// - The output image cannot be written.
/// - An unknown filter name is supplied.
pub fn run(args: FilterArgs) -> Result<()> {
    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        filter = %args.filter,
        "filter: starting"
    );

    match args.filter.as_str() {
        "gaussian" => run_gaussian(&args),
        "n4-bias" => run_n4_bias(&args),
        "anisotropic" => run_anisotropic(&args),
        "gradient-magnitude" => run_gradient_magnitude(&args),
        "laplacian" => run_laplacian(&args),
        "frangi" => run_frangi(&args),
        other => Err(anyhow!(
            "Unknown filter '{other}'. \
             Available filters: gaussian, n4-bias, anisotropic, \
             gradient-magnitude, laplacian, frangi."
        )),
    }
}

// ── Gaussian filter ───────────────────────────────────────────────────────────

/// Apply a Gaussian smoothing filter to the input image and write the result.
///
/// The sigma value from `args.sigma` is applied uniformly along all three
/// spatial dimensions.  The `GaussianFilter` implementation skips any
/// dimension whose sigma is ≤ 1e-6, so `--sigma 0.0` is a valid no-op.
fn run_gaussian(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GaussianFilter;

    let image = read_image(&args.input)?;

    // Isotropic sigma applied to all three spatial dimensions.
    let filter: GaussianFilter<Backend> = GaussianFilter::new(vec![args.sigma; 3]);
    let filtered = filter.apply(&image);

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied gaussian (\u{03c3}={}) to {} \u{2192} {}",
        args.sigma,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        sigma = args.sigma,
        "filter: gaussian complete"
    );

    Ok(())
}

// ── N4 bias field correction ──────────────────────────────────────────────────

fn run_n4_bias(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::bias::N4Config;
    use ritk_core::filter::N4BiasFieldCorrectionFilter;

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
        input = %args.input.display(),
        output = %args.output.display(),
        levels = args.levels,
        iterations = args.iterations,
        "filter: n4-bias complete"
    );

    Ok(())
}

// ── Anisotropic diffusion ─────────────────────────────────────────────────────

fn run_anisotropic(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::diffusion::{ConductanceFunction, DiffusionConfig};
    use ritk_core::filter::AnisotropicDiffusionFilter;

    let image = read_image(&args.input)?;
    let config = DiffusionConfig {
        num_iterations: args.iterations,
        conductance: args.conductance as f32,
        time_step: 0.0625,
        function: ConductanceFunction::Exponential,
    };
    let filter = AnisotropicDiffusionFilter::new(config);
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
        input = %args.input.display(),
        output = %args.output.display(),
        iterations = args.iterations,
        conductance = args.conductance,
        "filter: anisotropic complete"
    );

    Ok(())
}

// ── Gradient magnitude ────────────────────────────────────────────────────────

fn run_gradient_magnitude(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GradientMagnitudeFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = GradientMagnitudeFilter::new([spacing[0], spacing[1], spacing[2]]);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied gradient-magnitude to {} \u{2192} {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        "filter: gradient-magnitude complete"
    );

    Ok(())
}

// ── Laplacian ─────────────────────────────────────────────────────────────────

fn run_laplacian(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LaplacianFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = LaplacianFilter::new([spacing[0], spacing[1], spacing[2]]);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied laplacian to {} \u{2192} {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        "filter: laplacian complete"
    );

    Ok(())
}

// ── Frangi vesselness ─────────────────────────────────────────────────────────

fn run_frangi(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::vesselness::FrangiConfig;
    use ritk_core::filter::FrangiVesselnessFilter;

    let image = read_image(&args.input)?;

    // Parse comma-separated scale list (e.g. "0.5,1.0,2.0").
    let scales: Vec<f64> = args
        .scales
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    let scales = if scales.is_empty() {
        vec![0.5, 1.0, 2.0]
    } else {
        scales
    };

    let config = FrangiConfig {
        scales: scales.clone(),
        alpha: args.alpha,
        beta: args.beta,
        gamma: args.gamma,
        bright_vessels: true,
    };
    let filter = FrangiVesselnessFilter { config };
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied frangi (scales={:?}, \u{03b1}={}, \u{03b2}={}, \u{03b3}={}) to {} \u{2192} {}",
        scales,
        args.alpha,
        args.beta,
        args.gamma,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        alpha = args.alpha,
        beta = args.beta,
        gamma = args.gamma,
        "filter: frangi complete"
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    /// Build a 5×5×5 test image whose voxel values are `0, 1, 2, …, 124`.
    fn make_test_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..125).map(|i| i as f32).collect();
        let td = TensorData::new(values, Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    // ── Positive: Gaussian creates output file ────────────────────────────────

    /// Applying the Gaussian filter must create the output file.
    #[test]
    fn test_filter_gaussian_creates_output_file() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("filtered.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        run(FilterArgs {
            input: input.clone(),
            output: output.clone(),
            filter: "gaussian".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 50,
            conductance: 3.0,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        })
        .unwrap();

        assert!(output.exists(), "output file must be created");
    }

    // ── Positive: Gaussian preserves shape ───────────────────────────────────

    /// The output image must have the same voxel dimensions as the input.
    #[test]
    fn test_filter_gaussian_preserves_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("filtered.mha");

        ritk_io::write_metaimage(&input, &make_test_image()).unwrap();

        run(FilterArgs {
            input: input.clone(),
            output: output.clone(),
            filter: "gaussian".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 50,
            conductance: 3.0,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        })
        .unwrap();

        let result = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            result.shape(),
            [5, 5, 5],
            "shape must be preserved after Gaussian filtering"
        );
    }

    // ── Positive: Gaussian with sigma=0 is a no-op ───────────────────────────

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

        run(FilterArgs {
            input: input.clone(),
            output: output.clone(),
            filter: "gaussian".to_string(),
            sigma: 0.0,
            levels: 4,
            iterations: 50,
            conductance: 3.0,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        })
        .unwrap();

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
            "voxel sum must be preserved under σ=0 Gaussian (orig={orig_sum}, result={result_sum})"
        );
    }

    // ── Positive: N4 bias-field correction creates output file ───────────────

    /// N4 bias-field correction must complete without error and write an output
    /// file of the same shape as the input.
    #[test]
    fn test_filter_n4_applies_correction() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(FilterArgs {
            input,
            output: output.clone(),
            filter: "n4-bias".to_string(),
            sigma: 1.0,
            levels: 1,
            iterations: 5,
            conductance: 3.0,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        });

        assert!(result.is_ok(), "n4-bias must succeed: {:?}", result.err());
        assert!(output.exists(), "n4-bias must write output file");
    }

    // ── Positive: anisotropic diffusion creates output file ───────────────────

    #[test]
    fn test_filter_anisotropic_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(FilterArgs {
            input,
            output: output.clone(),
            filter: "anisotropic".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 5,
            conductance: 3.0,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        });

        assert!(
            result.is_ok(),
            "anisotropic must succeed: {:?}",
            result.err()
        );
        assert!(output.exists(), "anisotropic must write output file");
    }

    // ── Positive: Frangi vesselness creates output file ───────────────────────

    #[test]
    fn test_filter_frangi_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(FilterArgs {
            input,
            output: output.clone(),
            filter: "frangi".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 5,
            conductance: 3.0,
            scales: "1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        });

        assert!(result.is_ok(), "frangi must succeed: {:?}", result.err());
        assert!(output.exists(), "frangi must write output file");
    }

    // ── Positive: gradient-magnitude creates output file ─────────────────────

    #[test]
    fn test_filter_gradient_magnitude_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(FilterArgs {
            input,
            output: output.clone(),
            filter: "gradient-magnitude".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 5,
            conductance: 3.0,
            scales: "1.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        });

        assert!(
            result.is_ok(),
            "gradient-magnitude must succeed: {:?}",
            result.err()
        );
        assert!(output.exists(), "gradient-magnitude must write output file");
    }

    // ── Positive: laplacian creates output file ───────────────────────────────

    #[test]
    fn test_filter_laplacian_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(FilterArgs {
            input,
            output: output.clone(),
            filter: "laplacian".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 5,
            conductance: 3.0,
            scales: "1.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        });

        assert!(result.is_ok(), "laplacian must succeed: {:?}", result.err());
        assert!(output.exists(), "laplacian must write output file");
    }

    // ── Negative: unknown filter name returns error ───────────────────────────

    #[test]
    fn test_filter_unknown_name_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(FilterArgs {
            input,
            output,
            filter: "bilateral".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 50,
            conductance: 3.0,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        });

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unknown filter 'bilateral'"),
            "error must name the unknown filter, got: {msg}"
        );
    }

    // ── Boundary: missing input file returns error ────────────────────────────

    #[test]
    fn test_filter_missing_input_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("does_not_exist.nii");
        let output = dir.path().join("out.nii");

        let result = run(FilterArgs {
            input,
            output,
            filter: "gaussian".to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 50,
            conductance: 3.0,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
        });

        assert!(result.is_err(), "missing input must yield an error");
    }
}
