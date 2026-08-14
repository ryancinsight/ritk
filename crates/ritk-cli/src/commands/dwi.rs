//! `dwi` subcommand group — diffusion-weighted image processing.
//!
//! Fitting is the library's ([`ritk_diffusion::maps`]); this module parses
//! arguments, reads the series and its gradient scheme, and writes the
//! requested maps as images. It holds no estimation logic of its own.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ritk_diffusion::maps::{fit_diffusion_maps, DiffusionMaps, DiffusionMapsConfig};
use ritk_diffusion_scheme::read_fsl_scheme;
use ritk_image::Image;
use tracing::info;

use super::{write_image_inferred, Backend};

/// Diffusion-weighted image processing.
#[derive(clap::Args, Debug)]
pub struct DwiArgs {
    #[command(subcommand)]
    pub command: DwiCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum DwiCommand {
    /// Fit a diffusion tensor per voxel and write its scalar maps.
    Tensor(TensorArgs),
}

/// Fit diffusion tensors and write the requested scalar maps.
///
/// At least one output must be requested; fitting a volume and writing nothing
/// is a mistake worth reporting rather than several minutes of silence.
#[derive(clap::Args, Debug)]
pub struct TensorArgs {
    /// Diffusion-weighted series (4-D NIfTI, or any natively readable series).
    #[arg(long)]
    pub dwi: PathBuf,

    /// FSL `bval` file: one b-value per volume, in acquisition order.
    #[arg(long)]
    pub bval: PathBuf,

    /// FSL `bvec` file: three rows of gradient components, in image-axis order.
    #[arg(long)]
    pub bvec: PathBuf,

    /// Write fractional anisotropy here.
    #[arg(long)]
    pub fa: Option<PathBuf>,

    /// Write mean diffusivity here, in mm²/s.
    #[arg(long)]
    pub md: Option<PathBuf>,

    /// Write axial diffusivity (λ₁) here, in mm²/s.
    #[arg(long)]
    pub ad: Option<PathBuf>,

    /// Write radial diffusivity ((λ₂ + λ₃) / 2) here, in mm²/s.
    #[arg(long)]
    pub rd: Option<PathBuf>,

    /// Background threshold, as a fraction of the b = 0 signal's upper
    /// percentile.
    ///
    /// Outside the head the signal is noise, and a tensor fitted to noise is
    /// strongly anisotropic, so an unmasked map is dominated by a bright rim
    /// tracing the skull. Pass `0` to fit every voxel.
    #[arg(long, default_value_t = DiffusionMapsConfig::default().background_fraction)]
    pub background_fraction: f64,
}

/// Execute the `dwi` subcommand group.
///
/// # Errors
///
/// Propagates argument, IO, and fitting failures.
pub fn run(args: DwiArgs) -> Result<()> {
    match args.command {
        DwiCommand::Tensor(args) => tensor(args),
    }
}

/// Execute `dwi tensor`.
fn tensor(args: TensorArgs) -> Result<()> {
    let requested: [(&str, Option<&Path>); 4] = [
        ("fa", args.fa.as_deref()),
        ("md", args.md.as_deref()),
        ("ad", args.ad.as_deref()),
        ("rd", args.rd.as_deref()),
    ];
    anyhow::ensure!(
        requested.iter().any(|(_, path)| path.is_some()),
        "no output requested: pass at least one of --fa, --md, --ad, --rd"
    );

    info!(
        "dwi tensor: dwi={} bval={} bvec={}",
        args.dwi.display(),
        args.bval.display(),
        args.bvec.display()
    );

    let scheme = read_scheme(&args.bval, &args.bvec)?;
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

    let volumes: Vec<&[f32]> = series
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
    let maps =
        fit_diffusion_maps(&scheme, &volumes, &config).context("fitting the tensor field")?;

    info!("fitted {} of {} voxels", maps.fitted_count(), maps.len());
    anyhow::ensure!(
        maps.fitted_count() > 0,
        "no voxel yielded an admissible tensor. Either the series and the scheme \
         are misaligned, or --background-fraction {} masked the whole volume.",
        args.background_fraction
    );

    for (name, path) in requested {
        let Some(path) = path else { continue };
        let values = measure(&maps, name);
        write_map(path, &values, &reference)
            .with_context(|| format!("writing {name} map to {}", path.display()))?;
        info!("wrote {name} to {}", path.display());
    }
    Ok(())
}

/// Load an FSL gradient scheme from its two sidecar files.
///
/// `ritk-diffusion-scheme` is deliberately free of filesystem IO — every entry
/// point takes contents, and `write_fsl_scheme` returns strings rather than
/// writing — so reading the files is the caller's job. This is four lines of
/// glue rather than duplicated parsing.
fn read_scheme(bval: &Path, bvec: &Path) -> Result<ritk_diffusion_scheme::GradientScheme> {
    let bval_contents =
        std::fs::read_to_string(bval).with_context(|| format!("reading {}", bval.display()))?;
    let bvec_contents =
        std::fs::read_to_string(bvec).with_context(|| format!("reading {}", bvec.display()))?;
    read_fsl_scheme(&bval_contents, &bvec_contents)
        .context("building the gradient scheme from its FSL sidecars")
}

/// The named scalar map.
///
/// Exhaustive over the same names the argument list declares, so a new output
/// flag cannot reach here without a matching arm.
fn measure(maps: &DiffusionMaps, name: &str) -> Vec<f64> {
    match name {
        "fa" => maps.fractional_anisotropy(),
        "md" => maps.mean_diffusivity(),
        "ad" => maps.axial_diffusivity(),
        "rd" => maps.radial_diffusivity(),
        other => unreachable!("invariant: {other} is not one of the declared outputs"),
    }
}

/// Write a scalar map carrying the input series' geometry.
///
/// The map is a measurement of the same voxels, so it inherits the reference
/// volume's origin, spacing, and direction — a map written into a default frame
/// would not overlay the anatomy it was computed from.
fn write_map(path: &Path, values: &[f64], reference: &Image<f32, Backend, 3>) -> Result<()> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "diffusion scalars are written at image precision, matching every other RITK map"
    )]
    let data: Vec<f32> = values.iter().map(|value| *value as f32).collect();

    let image = Image::from_flat_on(
        data,
        reference.shape(),
        *reference.origin(),
        *reference.spacing(),
        *reference.direction(),
        &Backend::default(),
    )?;
    write_image_inferred(path, &image)
}

#[cfg(test)]
mod tests;
