//! `tract` subcommand group — streamline tractography.
//!
//! Fitting and the direction lookup are the library's
//! ([`ritk_diffusion::maps`], [`ritk_tractography`]); this module parses
//! arguments, builds the validated DTI volume policy, converts the resulting
//! streamlines into the image's physical frame, and writes them. It holds no
//! seeding or tracking logic of its own.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: ritk-cli is the application output layer"
)]

use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ritk_diffusion::maps::{
    fit_diffusion_maps, DiffusionMapsConfig, DirectionInterpolation, DtiVolume,
};
use ritk_image::Image;
use ritk_tractography::{
    dti_volume_tractography, DtiTractographyConfig, TerminationReason, TrackingDirection,
    TractographyConfig,
};
use tracing::info;

use super::Backend;

mod connectome;

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
    /// Reduce a tractogram and a parcellation to a region connectome.
    Connectome(connectome::ConnectomeArgs),
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
///   so a track entering a crossing follows an average orientation belonging to
///   no fibre, and stops where the orientations genuinely disagree. Roughly a
///   quarter of voxels survive masking and physical rejection at all.
///
/// The discontinuous direction field that used to be the second cause is fixed:
/// `--interpolation trilinear` is the default and roughly halves turn-limit
/// terminations, taking the median track from 16 mm to 24 mm on this subject.
/// `--interpolation nearest` restores the old behaviour for comparison.
///
/// Reported termination reasons distinguish the causes — a field boundary means
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

    /// Write the streamlines here.
    ///
    /// The format follows the extension: `.tck` (MRtrix), `.trk` (TrackVis /
    /// DSI Studio) or `.trx`, which is a directory rather than a file. All
    /// three carry the streamlines in the reference image's physical frame.
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
        TractCommand::Connectome(args) => connectome::run(args),
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

    let volume = DtiVolume::new(maps, reference.shape(), args.track_anisotropy)
        .context("placing the tensor field on the image grid")?
        .with_interpolation(args.interpolation.into());

    let tracking = TractographyConfig::new(
        args.step_size,
        args.max_steps,
        args.max_turn_degrees,
        TrackingDirection::Bidirectional,
    )
    .context("validating the tracking configuration")?;

    let policy = DtiTractographyConfig::new(args.seed_anisotropy, args.max_seeds, tracking)
        .context("validating the DTI seeding configuration")?;
    let tracks = dti_volume_tractography(&volume, policy)
        .context("seeding and tracking through the tensor field")?;
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

    write_tracks(&args.output, &tracks, &reference)
        .with_context(|| format!("writing {}", args.output.display()))?;

    println!(
        "wrote {}: {} streamlines from {} seeds",
        args.output.display(),
        tracks.streamlines().len(),
        tracks.seeds_attempted()
    );
    Ok(())
}

/// Track file formats `tract` can write.
///
/// Inferred from the output path's extension, matching how the rest of the CLI
/// selects image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackFormat {
    /// MRtrix `.tck`.
    Tck,
    /// TrackVis / DSI Studio `.trk`.
    Trk,
    /// Tractography Reference eXchange `.trx`, written as a directory.
    Trx,
}

impl TrackFormat {
    /// Infer the format from an output path.
    fn from_path(path: &Path) -> Option<Self> {
        match path
            .extension()
            .and_then(|extension| extension.to_str())?
            .to_ascii_lowercase()
            .as_str()
        {
            "tck" => Some(Self::Tck),
            "trk" => Some(Self::Trk),
            "trx" => Some(Self::Trx),
            _ => None,
        }
    }
}

/// Write streamlines in the image's physical frame, in the inferred format.
///
/// Tracking runs in voxel indices, but a tractogram is only meaningful beside
/// the anatomy it came from, so every point is mapped through the image's own
/// transform before it is written. The conversion happens once here rather than
/// per format.
fn write_tracks(
    path: &Path,
    tracks: &ritk_tractography::TractographyResult,
    reference: &Image<f32, Backend, 3>,
) -> Result<()> {
    let format = TrackFormat::from_path(path).with_context(|| {
        format!(
            "cannot infer a track format from {}: expected .tck, .trk or .trx",
            path.display()
        )
    })?;

    let physical = tracks
        .map_points(|index| reference.continuous_index_to_physical_point(index))
        .context("mapping streamlines into the image's physical frame")?;

    match format {
        TrackFormat::Tck => write_to_file(path, |writer| {
            physical.to_tck().write(writer).map_err(anyhow::Error::from)
        }),
        TrackFormat::Trk => write_to_file(path, |writer| {
            trk(&physical, reference)
                .write(writer)
                .map_err(anyhow::Error::from)
        }),
        // .trx is a directory of arrays, not a single stream.
        TrackFormat::Trx => physical
            .to_trx()
            .write_dir(path)
            .map_err(anyhow::Error::from),
    }
}

/// Create `path` and hand a buffered writer to `emit`.
fn write_to_file(path: &Path, emit: impl FnOnce(&mut BufWriter<File>) -> Result<()>) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    emit(&mut writer)?;
    writer.flush()?;
    Ok(())
}

/// Build a `.trk` tractogram carrying the reference image's geometry.
///
/// Two conversions, both easy to get wrong and neither visible in the resulting
/// coordinates.
///
/// **Axis order.** `.trk` describes the volume in voxel `(i, j, k)` order,
/// fastest axis first, which is what every NIfTI reader reports. RITK orders
/// `shape`, `spacing` and the `Direction` columns `[depth, row, column]`,
/// slowest first. So `dim`, `voxel_size` and the affine's columns are all
/// reversed here. Emitting RITK's order instead still round-trips — the writer
/// inverts whatever affine it is given — but declares a `72 x 104 x 104` volume
/// where the source image is `104 x 104 x 72`, and a viewer asked to overlay the
/// two would disagree about the bounding box.
///
/// **Handedness.** `.trk` embeds a voxel-to-**RAS** affine; RITK geometry is
/// LPS, and the two differ by the sign of the first two axes, so those rows are
/// negated. Leaving the identity affine that `to_trk` alone would write yields
/// a file that loads and displays in the wrong place.
///
/// `ritk_trk` applies the inverse of this affine to every point on write, so the
/// streamlines are stored in the voxel space it describes and a reader
/// recovers the physical coordinates by applying it back.
fn trk(
    tracks: &ritk_tractography::TractographyResult,
    reference: &Image<f32, Backend, 3>,
) -> ritk_trk::TrkTractogram {
    let shape = reference.shape();
    let spacing = reference.spacing();
    let direction = reference.direction();
    let origin = reference.origin();

    /// RITK axis for a `.trk` axis: the two orders are reverses of each other.
    const fn ritk_axis(trk_axis: usize) -> usize {
        2 - trk_axis
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "image extents and millimetre spacings are far inside i16 and f32"
    )]
    let dim = [
        shape[ritk_axis(0)] as i16,
        shape[ritk_axis(1)] as i16,
        shape[ritk_axis(2)] as i16,
    ];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "millimetre voxel spacings are far inside f32"
    )]
    let voxel_size = [
        spacing[ritk_axis(0)] as f32,
        spacing[ritk_axis(1)] as f32,
        spacing[ritk_axis(2)] as f32,
    ];

    // Column c is that voxel axis scaled by its spacing; the last column is the
    // origin. Rows 0 and 1 are negated to take LPS to RAS.
    let mut affine = [[0.0_f32; 4]; 4];
    affine[3][3] = 1.0;
    for row in 0..3 {
        let flip = if row < 2 { -1.0 } else { 1.0 };
        for column in 0..3 {
            let axis = ritk_axis(column);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "direction cosines scaled by millimetre spacing are far inside f32"
            )]
            let value = (direction.0[(row, axis)] * spacing[axis]) as f32;
            affine[row][column] = flip * value;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "millimetre origins are far inside f32"
        )]
        let offset = origin[row] as f32;
        affine[row][3] = flip * offset;
    }

    tracks.to_trk_header(dim, voxel_size, Some(affine))
}

#[cfg(test)]
mod tests;
