//! `ritk convert` — format conversion command.
//!
//! Reads an image from any supported input format (inferred from extension)
//! and writes it to any supported output format (inferred from extension or
//! overridden via `--format`).
//!
//! # Supported input formats
//! NIfTI (`.nii`, `.nii.gz`), MetaImage (`.mha`, `.mhd`), NRRD (`.nrrd`,
//! `.nhdr`), PNG (`.png`), DICOM (`.dcm`).
//!
//! # Supported output formats
//! NIfTI, MetaImage, NRRD.  PNG and DICOM output are not supported because
//! `ritk-io` does not export write implementations for those formats.

use anyhow::{anyhow, Result};
use clap::Args;
use ritk_io::ImageFormat;
use std::path::PathBuf;
use tracing::info;

use super::{infer_format, is_read_capable, is_write_capable, read_image, write_image};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Override output format.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    #[value(name = "nifti")]
    Nifti,
    #[value(name = "metaimage")]
    MetaImage,
    #[value(name = "nrrd")]
    Nrrd,
    Mgh,
    Tiff,
    Vtk,
    Jpeg,
    Analyze,
}

/// Arguments for the `convert` subcommand.
#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Input image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image path.  Format is inferred from the file extension unless
    /// `--format` is supplied.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Override the output format.
    #[arg(long, value_enum, value_name = "FORMAT")]
    pub format: Option<OutputFormat>,
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `convert` subcommand.
///
/// 1. Reads the image at `args.input` (format inferred from extension).
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

    let in_fmt = infer_format(&args.input).ok_or_else(|| {
        anyhow!(
            "Cannot infer input format from path: {}",
            args.input.display()
        )
    })?;

    // Resolve output format: explicit flag takes precedence over extension.
    let out_fmt: ImageFormat = match args.format {
        Some(fmt) => match fmt {
            OutputFormat::Nifti => ImageFormat::NIfTI,
            OutputFormat::MetaImage => ImageFormat::MetaImage,
            OutputFormat::Nrrd => ImageFormat::Nrrd,
            OutputFormat::Mgh => ImageFormat::Mgh,
            OutputFormat::Tiff => ImageFormat::Tiff,
            OutputFormat::Vtk => ImageFormat::Vtk,
            OutputFormat::Jpeg => ImageFormat::Jpeg,
            OutputFormat::Analyze => ImageFormat::Analyze,
        },
        None => infer_format(&args.output).ok_or_else(|| {
            anyhow!(
                "Cannot infer output format from path '{}'. \
                     Specify --format nifti|metaimage|nrrd.",
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
    let image = read_image(&args.input)?;
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

    info!(
        "convert: complete input={} output={} shape={:?}",
        args.input.display(),
        args.output.display(),
        shape
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    use crate::commands::Backend;

    /// Build a small deterministic 3-D image for testing.
    ///
    /// Shape is [3, 4, 5] (nz=3, ny=4, nx=5).  Voxel value at flat index i is
    /// `i as f32`.  Origin and spacing are identity.
    fn make_test_image() -> Image<f32, Backend, 3> {
        let n = 3 * 4 * 5;
        let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Image::from_flat_on(
            values,
            [3, 4, 5],
            Point::new([0.0; 3]),
            Spacing::new([1.0, 1.5, 2.0]),
            Direction::identity(),
            &Backend::default(),
        )
        .expect("invariant: image data matches shape")
    }

    // ── Positive: NIfTI round-trip ────────────────────────────────────────────

    /// Writing a NIfTI file and converting it back to another NIfTI must
    /// produce an output file with the same shape.
    #[test]
    fn test_convert_nifti_to_nifti_round_trip() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.nii");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::NIfTI).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists(), "output NIfTI must be created");
        let recovered = read_image(&output).unwrap();
        assert_eq!(
            recovered.shape(),
            [3, 4, 5],
            "shape must survive the round-trip"
        );
    }

    // ── Positive: NIfTI → MetaImage ───────────────────────────────────────────

    /// Converting a NIfTI to a MetaImage must produce a `.mha` file with the
    /// same shape.
    #[test]
    fn test_convert_nifti_to_metaimage() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.mha");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::NIfTI).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists(), "output MHA must be created");
        let recovered = read_image(&output).unwrap();
        assert_eq!(recovered.shape(), [3, 4, 5]);
    }

    // ── Positive: NIfTI → NRRD ───────────────────────────────────────────────

    /// Converting a NIfTI to NRRD must produce a `.nrrd` file with the
    /// same shape.
    #[test]
    fn test_convert_nifti_to_nrrd() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.nrrd");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::NIfTI).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists(), "output NRRD must be created");
        let recovered = read_image(&output).unwrap();
        assert_eq!(recovered.shape(), [3, 4, 5]);
    }

    // ── Positive: explicit --format overrides extension ───────────────────────

    /// When `--format nifti` is passed the output extension is ignored and a
    /// valid NIfTI file is produced.
    #[test]
    fn test_convert_explicit_format_flag_overrides_extension() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        // Deliberately give the output a non-NIfTI extension.
        let output = dir.path().join("output.nii");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::NIfTI).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            format: Some(OutputFormat::Nifti),
        })
        .unwrap();

        assert!(output.exists());
    }

    // ── Negative: unknown output extension without --format ───────────────────

    /// When the output path has an unrecognised extension and no `--format`
    /// flag is provided, the command must return an error (not panic).
    #[test]
    fn test_convert_unknown_output_extension_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.xyz");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::NIfTI).unwrap();

        let result = run(ConvertArgs {
            input,
            output,
            format: None,
        });
        assert!(
            result.is_err(),
            "unknown output extension must yield an error"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Cannot infer output format"),
            "error must explain the problem, got: {msg}"
        );
    }

    // ── Negative: non-existent input file ─────────────────────────────────────

    /// Attempting to convert a path that does not exist must return an error.
    #[test]
    fn test_convert_missing_input_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("does_not_exist.nii");
        let output = dir.path().join("output.nii");

        let result = run(ConvertArgs {
            input,
            output,
            format: None,
        });
        assert!(result.is_err(), "missing input must yield an error");
    }

    // ── Boundary: MetaImage round-trip ────────────────────────────────────────

    /// Writing a MetaImage and converting it back to NIfTI must preserve shape.
    #[test]
    fn test_convert_metaimage_to_nifti() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("output.nii");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::MetaImage).unwrap();

        run(ConvertArgs {
            input,
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists());
        let recovered = read_image(&output).unwrap();
        assert_eq!(recovered.shape(), [3, 4, 5]);
    }

    // ── ADR 0003 Phase A: native-dispatch coverage ────────────────────────────

    /// `is_read_capable`/`is_write_capable` must agree exactly
    /// with the native format matrix: VTK is now read and written through its
    /// native adapter, while PNG and DICOM remain read-only. A drift here would silently misroute a
    /// command onto the wrong substrate without any other test catching it.
    #[test]
    fn test_native_capability_predicates_match_adr_0003_matrix() {
        use super::super::{is_read_capable, is_write_capable};

        let read_and_write = [
            ImageFormat::NIfTI,
            ImageFormat::Nrrd,
            ImageFormat::Analyze,
            ImageFormat::Mgh,
            ImageFormat::MetaImage,
            ImageFormat::Tiff,
            ImageFormat::Jpeg,
            ImageFormat::Vtk,
        ];
        for fmt in read_and_write {
            assert!(is_read_capable(fmt), "{fmt:?} must read natively");
            assert!(is_write_capable(fmt), "{fmt:?} must write natively");
        }

        assert!(is_read_capable(ImageFormat::Png), "PNG reads natively");
        assert!(
            !is_write_capable(ImageFormat::Png),
            "PNG has no native writer"
        );

        assert!(is_read_capable(ImageFormat::Dicom), "DICOM reads natively");
        assert!(
            !is_write_capable(ImageFormat::Dicom),
            "DICOM has no native writer yet"
        );

        assert!(is_read_capable(ImageFormat::Vtk), "VTK reads natively");
        assert!(is_write_capable(ImageFormat::Vtk), "VTK writes natively");
    }

    /// Conversion preserves native NIfTI serialization bytes when no format
    /// conversion is requested.
    #[test]
    fn test_native_convert_output_is_byte_identical_to_coeus_writer() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let native_output = dir.path().join("via_convert.nii");
        let direct_output = dir.path().join("via_native_direct.nii");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::NIfTI).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: native_output.clone(),
            format: None,
        })
        .unwrap();

        write_image(&direct_output, &image, ImageFormat::NIfTI).unwrap();

        assert_eq!(
            std::fs::read(&native_output).unwrap(),
            std::fs::read(&direct_output).unwrap(),
            "convert must preserve the native NIfTI serialization contract"
        );
    }

    /// VTK input is decoded through the native reader and converted to NIfTI.
    #[test]
    fn test_convert_vtk_input_native_round_trip() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.vtk");
        let output = dir.path().join("output.nii");
        let image = make_test_image();
        write_image(&input, &image, ImageFormat::Vtk).unwrap();

        run(ConvertArgs {
            input,
            output: output.clone(),
            format: None,
        })
        .expect("native VTK input conversion");
        let recovered = read_image(&output).unwrap();
        assert_eq!(recovered.shape(), image.shape());
        assert_eq!(recovered.data_slice().unwrap(), image.data_slice().unwrap());
    }

    /// VTK output is encoded through the native writer.
    #[test]
    fn test_convert_vtk_output_native_round_trip() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.vtk");

        let image = make_test_image();
        write_image(&input, &image, ImageFormat::NIfTI).unwrap();

        run(ConvertArgs {
            input,
            output: output.clone(),
            format: Some(OutputFormat::Vtk),
        })
        .expect("native VTK output conversion");
        let recovered = read_image(&output).unwrap();
        assert_eq!(recovered.shape(), image.shape());
        assert_eq!(recovered.data_slice().unwrap(), image.data_slice().unwrap());
    }
}
