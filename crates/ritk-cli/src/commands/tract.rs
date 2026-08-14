//! `tract` subcommand group — streamline tractography.
//!
//! Fitting and the direction lookup are the library's
//! ([`ritk_diffusion::maps`], [`ritk_tractography`]); this module parses
//! arguments, seeds, converts the resulting streamlines into the image's
//! physical frame, and writes them. It holds no tracking logic of its own.

use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ritk_diffusion::maps::{
    fit_diffusion_maps, DiffusionMapsConfig, DirectionInterpolation, DtiVolume,
};
use ritk_image::Image;
use ritk_spatial::Point;
use ritk_tck::{TckHeader, TckTractogram};
use ritk_tractography::{
    dti_volume_direction_field, euler_tractography, TerminationReason, TrackingDirection,
    TractographyConfig,
};
use tracing::info;

use super::Backend;

/// Streamline tractography.
#[derive(clap::Args, Debug)]
pub struct TractArgs {
    #[command(subcommand)]
    pub command: TractCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum TractCommand {
    /// Track streamlines through a diffusion tensor field.
    Dti(DtiArgs),
}

/// Fit tensors, track through them, and write the streamlines.
///
/// # What this produces, and what it does not
///
/// Streamlines are short by the standards of a tuned pipeline. On OpenNeuro
/// ds002087 sub-01 the median track is about 16 mm, where anatomical bundles
/// run 30–150 mm. Two causes, neither of which is fixed by loosening a
/// threshold:
///
/// - **The data.** Single-shell b = 700 at 2 mm resolves one tensor per voxel,
///   so a track entering a crossing region follows an average orientation that
///   belongs to no fibre. Roughly a quarter of voxels survive masking and
///   physical rejection at all.
/// - **Nearest-neighbour direction lookup.** The orientation is constant within
///   a voxel and steps discontinuously at each boundary, so genuinely smooth
///   bundles can exceed the turn limit. Interpolating the field would soften
///   this, and is not simply a matter of averaging vectors: an eigenvector has
///   no sign, so neighbours must be aligned before they are combined or they
///   cancel.
///
/// Reported termination reasons distinguish the two — a field boundary means
/// the mask or anisotropy floor ended the track, a turning angle means the
/// field was rougher than the limit allows.
#[derive(clap::Args, Debug)]
pub struct DtiArgs {
    /// Diffusion-weighted series (4-D NIfTI, or any natively readable series).
    #[arg(long)]
    pub dwi: PathBuf,

    /// FSL `bval` file: one b-value per volume, in acquisition order.
    #[arg(long)]
    pub bval: PathBuf,

    /// FSL `bvec` file: three rows of gradient components, in image-axis order.
    #[arg(long)]
    pub bvec: PathBuf,

    /// Write the streamlines here, as MRtrix `.tck`.
    #[arg(long)]
    pub output: PathBuf,

    /// Fractional anisotropy at or above which a voxel is seeded.
    ///
    /// 0.25 is the conventional white-matter floor: grey matter and CSF sit
    /// well below it, coherent bundles well above.
    #[arg(long, default_value_t = 0.25)]
    pub seed_anisotropy: f64,

    /// Fractional anisotropy below which a streamline stops.
    ///
    /// Lower than the seeding floor by convention, so a track started in
    /// confident white matter may follow a bundle through a less certain
    /// stretch instead of terminating at the first dip.
    #[arg(long, default_value_t = 0.15)]
    pub track_anisotropy: f64,

    /// Cap on seeds. Zero seeds every qualifying voxel.
    ///
    /// Seeds are taken at a stride over the qualifying voxels rather than the
    /// first N, so a cap thins the whole volume evenly instead of covering
    /// whichever end is stored first.
    #[arg(long, default_value_t = 10_000)]
    pub max_seeds: usize,

    /// Integration step, in voxels.
    #[arg(long, default_value_t = 0.5)]
    pub step_size: f64,

    /// Cap on integration steps per streamline.
    #[arg(long, default_value_t = 1_000)]
    pub max_steps: usize,

    /// Sharpest turn a streamline may take between steps, in degrees.
    #[arg(long, default_value_t = 60.0)]
    pub max_turn_degrees: f64,

    /// How the orientation is sampled between voxel centres.
    ///
    /// `trilinear` interpolates the outer product of the surrounding voxels,
    /// which is sign-invariant and so cannot cancel two neighbours describing
    /// the same fibre with opposite eigenvector signs. `nearest` is the
    /// piecewise-constant baseline, kept so the two can be compared on the same
    /// data.
    #[arg(long, value_enum, default_value_t = Interpolation::Trilinear)]
    pub interpolation: Interpolation,

    /// Background threshold, as a fraction of the b = 0 signal's upper
    /// percentile. Pass `0` to fit every voxel.
    #[arg(long, default_value_t = DiffusionMapsConfig::default().background_fraction)]
    pub background_fraction: f64,
}

/// Orientation sampling mode, as a command-line value.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// The orientation of the nearest voxel centre.
    Nearest,
    /// Trilinear over the outer product of the surrounding voxels.
    Trilinear,
}

impl From<Interpolation> for DirectionInterpolation {
    fn from(value: Interpolation) -> Self {
        match value {
            Interpolation::Nearest => Self::Nearest,
            Interpolation::Trilinear => Self::Trilinear,
        }
    }
}

/// Execute the `tract` subcommand group.
///
/// # Errors
///
/// Propagates argument, IO, fitting, and tracking failures.
pub fn run(args: TractArgs) -> Result<()> {
    match args.command {
        TractCommand::Dti(args) => dti(args),
    }
}

/// Execute `tract dti`.
fn dti(args: DtiArgs) -> Result<()> {
    info!(
        "tract dti: dwi={} output={}",
        args.dwi.display(),
        args.output.display()
    );

    let scheme = super::dwi::read_scheme(&args.bval, &args.bvec)?;
    let series = ritk_io::read_image_series_native(&args.dwi)
        .with_context(|| format!("reading {}", args.dwi.display()))?;
    let reference = series
        .first()
        .context("the series contains no volumes")?
        .clone();
    anyhow::ensure!(
        series.len() == scheme.len(),
        "series has {} volumes but the scheme declares {}",
        series.len(),
        scheme.len()
    );

    let voxels: Vec<&[f32]> = series
        .iter()
        .map(|volume| {
            volume
                .data_slice()
                .context("series volume is not contiguous in host memory")
        })
        .collect::<Result<_>>()?;

    let config = DiffusionMapsConfig {
        background_fraction: args.background_fraction,
        ..DiffusionMapsConfig::default()
    };
    let maps = fit_diffusion_maps(&scheme, &voxels, &config).context("fitting the tensor field")?;
    info!("fitted {} of {} voxels", maps.fitted_count(), maps.len());

    let anisotropy = maps.fractional_anisotropy();
    let volume = DtiVolume::new(maps, reference.shape(), args.track_anisotropy)
        .context("placing the tensor field on the image grid")?
        .with_interpolation(args.interpolation.into());

    let seeds = seed(
        &anisotropy,
        volume.shape(),
        args.seed_anisotropy,
        args.max_seeds,
    );
    anyhow::ensure!(
        !seeds.is_empty(),
        "no voxel reached FA {}, so there is nothing to seed. The peak was {:.3}.",
        args.seed_anisotropy,
        anisotropy.iter().copied().fold(0.0_f64, f64::max)
    );
    info!("seeding {} voxels", seeds.len());

    let tracking = TractographyConfig::new(
        args.step_size,
        args.max_steps,
        args.max_turn_degrees,
        TrackingDirection::Bidirectional,
    )
    .context("validating the tracking configuration")?;

    let tracks = euler_tractography(&seeds, tracking, dti_volume_direction_field(&volume))
        .context("tracking through the tensor field")?;
    // Why tracking stopped is the first thing to look at when a tractogram is
    // shorter than expected: a field boundary means the mask or anisotropy
    // floor ended it, a turning angle means the direction field is rougher than
    // the turn limit allows.
    let mut boundary = 0_usize;
    let mut turning = 0_usize;
    let mut step_limit = 0_usize;
    for streamline in tracks.streamlines() {
        for reason in std::iter::once(streamline.forward_termination())
            .chain(streamline.backward_termination())
        {
            match reason {
                TerminationReason::FieldBoundary => boundary += 1,
                TerminationReason::TurningAngle => turning += 1,
                TerminationReason::StepLimit => step_limit += 1,
            }
        }
    }
    info!(
        "tracked {} streamlines from {} seeds; terminations: {boundary} at a field boundary, \
         {turning} on the turn limit, {step_limit} at the step limit",
        tracks.streamlines().len(),
        tracks.seeds_attempted()
    );

    write_tck(&args.output, &tracks, &reference)
        .with_context(|| format!("writing {}", args.output.display()))?;

    println!(
        "wrote {}: {} streamlines from {} seeds",
        args.output.display(),
        tracks.streamlines().len(),
        tracks.seeds_attempted()
    );
    Ok(())
}

/// Voxel indices whose anisotropy qualifies them as seeds.
///
/// Qualifying voxels are taken at a stride rather than truncated at `limit`, so
/// a cap thins the volume evenly. Truncating would seed only whichever end of
/// the volume is stored first, which reads as a tractogram covering half a
/// brain.
fn seed(anisotropy: &[f64], shape: [usize; 3], floor: f64, limit: usize) -> Vec<Point<3>> {
    let qualifying: Vec<usize> = anisotropy
        .iter()
        .enumerate()
        .filter(|(_, value)| **value >= floor)
        .map(|(voxel, _)| voxel)
        .collect();

    let stride = if limit == 0 || qualifying.len() <= limit {
        1
    } else {
        qualifying.len().div_ceil(limit)
    };

    let [_, rows, columns] = shape;
    let plane = rows * columns;
    qualifying
        .iter()
        .step_by(stride)
        .map(|voxel| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "voxel indices are far below f64's exact-integer range"
            )]
            let index = [
                (voxel / plane) as f64,
                ((voxel % plane) / columns) as f64,
                (voxel % columns) as f64,
            ];
            Point::new(index)
        })
        .collect()
}

/// Write streamlines as MRtrix `.tck`, in the image's physical frame.
///
/// Tracking runs in voxel indices, but a tractogram is only meaningful beside
/// the anatomy it came from, so every point is mapped through the image's own
/// transform before it is written.
fn write_tck(
    path: &Path,
    tracks: &ritk_tractography::TractographyResult,
    reference: &Image<f32, Backend, 3>,
) -> Result<()> {
    let streamlines = tracks
        .streamlines()
        .iter()
        .map(|streamline| {
            let points: Vec<_> = streamline
                .geometry()
                .points()
                .iter()
                .map(|index| {
                    let physical = reference.continuous_index_to_physical_point(&Point::new([
                        index.x, index.y, index.z,
                    ]));
                    leto::geometry::Point3::new(physical[0], physical[1], physical[2])
                })
                .collect();
            gaia::Polyline::new(points).context("streamline geometry in physical coordinates")
        })
        .collect::<Result<Vec<_>>>()?;

    #[expect(
        clippy::cast_possible_wrap,
        reason = "a tractogram holds far fewer streamlines than i64::MAX"
    )]
    let count = streamlines.len() as i64;
    let tractogram = TckTractogram {
        header: TckHeader {
            count: Some(count),
            ..TckHeader::default()
        },
        streamlines,
    };

    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    tractogram.write(&mut writer)?;
    Ok(())
}

#[cfg(test)]
mod tests;
