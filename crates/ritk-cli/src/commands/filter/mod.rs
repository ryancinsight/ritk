//! `ritk filter` â€” image filtering command.
//!
//! Applies one of 34 filters to a 3-D medical image. The CLI surface uses
//! `--filter <KIND>` for closed-set dispatch (typed by [`FilterKind`]) so
//! unknown filter names are rejected at parse time, not at runtime.

use anyhow::Result;
use tracing::info;

pub(crate) use super::{Backend, NativeBackend};

pub mod args;
pub use args::*;

mod intensity;
mod morphology;
mod smoothing;
mod spatial;
#[path = "spatial_impl.rs"]
mod spatial_file;

#[cfg(test)]
mod tests;

// â”€â”€ Command handler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Execute the `filter` subcommand.
///
/// Dispatches to the appropriate filter implementation based on `args.kind`.
/// Match is exhaustive (the [`FilterKind`] enum has no `_ =>` arm) so unknown
/// filter names are caught at CLI parse time by clap.
///
/// # Errors
/// Returns an error when:
/// - The input image cannot be read.
/// - The output image cannot be written.
pub fn run(args: FilterArgs) -> Result<()> {
    info!(
        "filter: starting input={} output={} kind={}",
        args.input.display(),
        args.output.display(),
        args.kind
    );

    match args.kind {
        FilterKind::Gaussian => smoothing::run_gaussian(&args),
        FilterKind::N4Bias => smoothing::run_n4_bias(&args),
        FilterKind::Anisotropic => smoothing::run_anisotropic(&args),
        FilterKind::GradientMagnitude => spatial::run_gradient_magnitude(&args),
        FilterKind::Laplacian => spatial::run_laplacian(&args),
        FilterKind::Frangi => spatial::run_frangi(&args),
        FilterKind::Median => spatial::run_median(&args),
        FilterKind::Bilateral => spatial::run_bilateral(&args),
        FilterKind::Canny => spatial::run_canny(&args),
        FilterKind::Sobel => spatial::run_sobel(&args),
        FilterKind::Log => spatial::run_log(&args),
        FilterKind::RecursiveGaussian => spatial::run_recursive_gaussian(&args),
        FilterKind::Curvature => smoothing::run_curvature(&args),
        FilterKind::Sato => smoothing::run_sato(&args),
        FilterKind::DiscreteGaussian => smoothing::run_discrete_gaussian(&args),
        FilterKind::BedSeparation => intensity::run_bed_separation(&args),
        FilterKind::Cpr => spatial::run_cpr(&args),
        FilterKind::RescaleIntensity => intensity::run_rescale_intensity(&args),
        FilterKind::IntensityWindowing => intensity::run_intensity_windowing(&args),
        FilterKind::ThresholdBelow => intensity::run_threshold_below(&args),
        FilterKind::ThresholdAbove => intensity::run_threshold_above(&args),
        FilterKind::ThresholdOutside => intensity::run_threshold_outside(&args),
        FilterKind::Sigmoid => intensity::run_sigmoid(&args),
        FilterKind::BinaryThreshold => intensity::run_binary_threshold(&args),
        FilterKind::GrayscaleErosion => morphology::run_grayscale_erosion(&args),
        FilterKind::GrayscaleDilation => morphology::run_grayscale_dilation(&args),
        FilterKind::WhiteTopHat => morphology::run_white_top_hat(&args),
        FilterKind::BlackTopHat => morphology::run_black_top_hat(&args),
        FilterKind::HitOrMiss => morphology::run_hit_or_miss(&args),
        FilterKind::LabelDilation => morphology::run_label_dilation(&args),
        FilterKind::LabelErosion => morphology::run_label_erosion(&args),
        FilterKind::LabelOpening => morphology::run_label_opening(&args),
        FilterKind::LabelClosing => morphology::run_label_closing(&args),
        FilterKind::MorphologicalReconstruction => {
            morphology::run_morphological_reconstruction(&args)
        }
    }
}

// â”€â”€ Test helpers (shared across leaf modules) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
use std::path::PathBuf;

/// Default `FilterArgs` builder â€” sets every per-family field to its
/// reasonable default and lets the caller override what is needed.
#[cfg(test)]
pub(crate) fn default_args(input: PathBuf, output: PathBuf, kind: FilterKind) -> FilterArgs {
    use ritk_filter::SpacingMode;
    FilterArgs {
        input,
        output,
        kind,
        smoothing: SmoothingArgs { sigma: 1.0 },
        diffusion: DiffusionArgs {
            levels: 4,
            iterations: 50,
            conductance: 3.0,
            time_step: 0.0625 },
        vesselness: VesselnessArgs {
            scales: vec![0.5, 1.0, 2.0],
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0 },
        edge: EdgeArgs {
            sigma_spatial: 3.0,
            sigma_range: 50.0,
            low: 0.1,
            high: 0.3 },
        discrete: DiscreteArgs {
            variance: 1.0,
            maximum_error: 0.01,
            spacing_mode: SpacingMode::Physical },
        kernel: KernelArgs { radius: 1 },
        recursive: RecursiveArgs {
            order: CliDerivativeOrder::Zero },
        bed: BedArgs {
            body_threshold: -350.0,
            closing_radius: 2,
            opening_radius: 1,
            outside_value: -1024.0 },
        range: RangeArgs {
            out_min: 0.0,
            out_max: 1.0 },
        window: WindowArgs {
            window_min: 0.0,
            window_max: 255.0 },
        band: BandArgs {
            lower_threshold: 0.0,
            upper_threshold: 1.0,
            outside_value: 0.0,
            foreground_value: 1.0,
            background_value: 0.0 },
        threshold: ThresholdArgs {
            threshold_value: 0.5 },
        mask_input: MaskInputArgs { mask: None },
        cpr: CprArgs {
            cpr_points: vec![],
            cpr_path_samples: 256,
            cpr_half_width: 10.0,
            cpr_cross_samples: 64 },
        sigmoid: SigmoidArgs {
            midpoint: 62.0,
            steepness: 20.0 } }
}

/// Build a 5Ã—5Ã—5 test image whose voxel values are `0, 1, 2, â€¦, 124`.
#[cfg(test)]
pub(crate) fn make_test_image() -> NativeImage<f32, NativeBackend, 3> {
    use crate::commands::NativeBackend;
    use ritk_image::native::Image as NativeImage;
    use ritk_spatial::{Direction, Point, Spacing};

    let values: Vec<f32> = (0..125).map(|i| i as f32).collect();
    NativeImage::from_flat_on(
        values,
        [5, 5, 5],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &NativeBackend::default(),
    )
    .expect("invariant: valid 5Ã—5Ã—5 ramp image")
}
