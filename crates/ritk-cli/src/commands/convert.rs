//! `ritk convert` — format conversion command.
//!
//! Reads an image from any supported input format (inferred from extension)
//! and writes it to any supported output format (inferred from extension or
//! overridden via `--format`).
//!
//! # Supported input formats
//! Uses RITK readers for NIfTI, MetaImage, NRRD, PNG, DICOM, MINC2, MGH, TIFF,
//! VTK, JPEG, and Analyze. This command converts scalar `f32` 3-D images;
//! rank-4 NIfTI, NRRD, and MGH series use RITK's separate series APIs. RGB DICOM
//! uses RITK's color-volume API rather than this scalar converter. DICOM files
//! and series directories are accepted; a directory with multiple image series
//! requires `--series-uid`.
//!
//! # Supported output formats
//! Uses RITK writers for NIfTI, MetaImage, NRRD, PNG, MINC2, MGH, TIFF, VTK,
//! JPEG, Analyze, and DICOM. PNG output is one grayscale slice with exact unsigned
//! 8-bit or 16-bit integer samples and no physical-space metadata. DICOM output
//! is a derived Secondary Capture series: each slice is quantized to unsigned
//! 16-bit pixels with a rescale, and source patient and study metadata is not
//! copied.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: ritk-cli is the application output layer"
)]

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use ritk_image::Image;
use ritk_io::ImageFormat;
use std::path::PathBuf;
use tracing::info;

use super::{infer_format, is_read_capable, is_write_capable, read_image, write_image, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Override output format.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Neuroimaging Informatics Technology Initiative.
    #[value(name = "nifti")]
    Nifti,
    /// MetaIO's MetaImage format.
    #[value(name = "metaimage")]
    MetaImage,
    /// Nearly Raw Raster Data.
    #[value(name = "nrrd")]
    Nrrd,
    /// Grayscale Portable Network Graphics.
    Png,
    /// Medical Imaging NetCDF, version 2.
    Minc,
    /// FreeSurfer volume.
    Mgh,
    /// Tagged Image File Format.
    Tiff,
    /// Legacy VTK structured points.
    Vtk,
    /// Joint Photographic Experts Group.
    Jpeg,
    /// Analyze 7.5 volume.
    Analyze,
    /// Digital Imaging and Communications in Medicine Secondary Capture.
    #[value(name = "dicom")]
    Dicom,
}

/// Arguments for the `convert` subcommand.
#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Input image file (format inferred from extension) or DICOM series directory.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output file path, or a directory for DICOM output. Format is inferred
    /// from the extension unless `--format` is supplied.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Override the output format.
    #[arg(
        long,
        value_enum,
        value_name = "FORMAT",
        long_help = "Output format: nifti, metaimage, nrrd, png, minc, mgh, tiff, vtk, jpeg, analyze, or dicom. This command converts scalar f32 3-D images; rank-4 NIfTI, NRRD, and MGH series use RITK's separate series APIs. RGB DICOM uses RITK's color-volume API. PNG is one grayscale slice with exact unsigned 8-bit or 16-bit integer samples and no physical-space metadata. DICOM output is a derived Secondary Capture series in a directory, with unsigned 16-bit per-slice rescaling; source patient and study metadata is not copied. JPEG output is lossy 8-bit grayscale and accepts one slice. TIFF and JPEG do not preserve physical-space metadata; Analyze does not store direction. VTK output requires identity direction. Conversion uses RITK's f32 image model, so wide integer samples are not guaranteed exact."
    )]
    pub format: Option<OutputFormat>,

    /// Select a DICOM SeriesInstanceUID when the input directory contains multiple image series.
    #[arg(long, value_name = "UID")]
    pub series_uid: Option<String>,
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `convert` subcommand.
///
/// 1. Reads the image at `args.input` (file format inferred from extension, or
///    DICOM series selected from a directory).
/// 2. Writes the image to `args.output` (format taken from `--format` or
///    inferred from the output extension).
/// 3. Prints a one-line summary: path pair, shape in ZxYxX, and spacing.
///
/// # Errors
/// Returns an error when the input cannot be read, the output cannot be
/// written, or neither the `--format` flag nor the output extension resolves
/// to a writable format.
pub fn run(args: ConvertArgs) -> Result<()> {
    info!(
        "convert: starting input={} output={}",
        args.input.display(),
        args.output.display()
    );

    let in_fmt = if args.input.is_dir() {
        ImageFormat::Dicom
    } else {
        infer_format(&args.input).ok_or_else(|| {
            anyhow!(
                "Cannot infer input format from path: {}",
                args.input.display()
            )
        })?
    };

    if args.series_uid.is_some() && !args.input.is_dir() {
        bail!("--series-uid requires a DICOM input directory");
    }

    // Resolve output format: explicit flag takes precedence over extension.
    let out_fmt: ImageFormat = match args.format {
        Some(fmt) => match fmt {
            OutputFormat::Nifti => ImageFormat::NIfTI,
            OutputFormat::MetaImage => ImageFormat::MetaImage,
            OutputFormat::Nrrd => ImageFormat::Nrrd,
            OutputFormat::Png => ImageFormat::Png,
            OutputFormat::Minc => ImageFormat::Minc,
            OutputFormat::Mgh => ImageFormat::Mgh,
            OutputFormat::Tiff => ImageFormat::Tiff,
            OutputFormat::Vtk => ImageFormat::Vtk,
            OutputFormat::Jpeg => ImageFormat::Jpeg,
            OutputFormat::Analyze => ImageFormat::Analyze,
            OutputFormat::Dicom => ImageFormat::Dicom,
        },
        None => infer_format(&args.output).ok_or_else(|| {
            anyhow!(
                "Cannot infer output format from path '{}'. Specify --format for directory outputs such as DICOM.",
                args.output.display()
            )
        })?,
    };

    anyhow::ensure!(
        is_read_capable(in_fmt),
        "convert does not support {:?} input until its native reader exists",
        in_fmt
    );
    anyhow::ensure!(
        is_write_capable(out_fmt),
        "convert does not support {:?} output until its native writer exists",
        out_fmt
    );
    let image = if args.input.is_dir() {
        read_dicom_directory(&args.input, args.series_uid.as_deref())?
    } else {
        read_image(&args.input)?
    };
    let shape = image.shape();
    let spacing = *image.spacing();
    write_image(&args.output, &image, out_fmt)?;

    println!(
        "Converted {} \u{2192} {} (shape: {}x{}x{}, spacing: {:.4}\u{d7}{:.4}\u{d7}{:.4})",
        args.input.display(),
        args.output.display(),
        shape[0],
        shape[1],
        shape[2],
        spacing[0],
        spacing[1],
        spacing[2],
    );

    match out_fmt {
        ImageFormat::Png => println!(
            "PNG output stores one grayscale slice as unsigned 8-bit or 16-bit integer samples and does not preserve physical-space metadata."
        ),
        ImageFormat::Dicom => println!(
            "DICOM output is derived Secondary Capture: unsigned 16-bit per-slice rescaling; source patient and study metadata is not copied."
        ),
        ImageFormat::Jpeg => println!(
            "JPEG output is lossy 8-bit grayscale, limited to one slice, and does not preserve physical-space metadata."
        ),
        ImageFormat::Tiff => println!(
            "TIFF output does not preserve physical-space metadata; readers use default geometry."
        ),
        ImageFormat::Analyze => println!(
            "Analyze output does not store image direction; readers assume identity direction."
        ),
        _ => {}
    }

    info!(
        "convert: complete input={} output={} shape={:?}",
        args.input.display(),
        args.output.display(),
        shape
    );

    Ok(())
}

fn read_dicom_directory(
    path: &std::path::Path,
    selected_uid: Option<&str>,
) -> Result<Image<f32, Backend, 3>> {
    match selected_uid {
        Some(uid) => ritk_io::format::dicom::read_native_dicom_series_with_uid(
            path,
            uid,
            &Backend::default(),
        ),
        None => ritk_io::format::dicom::read_native_dicom_series(path, &Backend::default()),
    }
    .with_context(|| format!("Failed to read DICOM series from {}", path.display()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "convert/tests.rs"]
mod tests;
