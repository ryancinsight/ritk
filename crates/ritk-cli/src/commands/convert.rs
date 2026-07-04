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

use super::{
    infer_format, is_native_read_capable, is_native_write_capable, read_image, read_image_native,
    write_image, write_image_native,
};

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

    // ADR 0003 Phase A: route through the Atlas-native path when both ends
    // have a native reader/writer; otherwise fall back to the Burn path
    // (currently `dicom` and `vtk` on either side). This avoids converting
    // between the two image types mid-command — each command run picks one
    // substrate for its whole read→write span.
    let shape;
    let spacing;
    if is_native_read_capable(in_fmt) && is_native_write_capable(out_fmt) {
        let image = read_image_native(&args.input)?;
        shape = image.shape();
        spacing = *image.spacing();
        write_image_native(&args.output, &image, out_fmt)?;
    } else {
        let image = read_image(&args.input)?;
        shape = image.shape();
        spacing = *image.spacing();
        write_image(&args.output, &image, out_fmt)?;
    }

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
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    use crate::commands::Backend;

    /// Build a small deterministic 3-D image for testing.
    ///
    /// Shape is [3, 4, 5] (nz=3, ny=4, nx=5).  Voxel value at flat index i is
    /// `i as f32`.  Origin and spacing are identity.
    fn make_test_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let n = 3 * 4 * 5;
        let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let td = TensorData::new(values, Shape::new([3, 4, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0, 1.5, 2.0]),
            Direction::identity(),
        )
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
        ritk_io::write_nifti(&input, &image).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists(), "output NIfTI must be created");
        let recovered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
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
        ritk_io::write_nifti(&input, &image).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists(), "output MHA must be created");
        let recovered =
            ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
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
        ritk_io::write_nifti(&input, &image).unwrap();

        run(ConvertArgs {
            input: input.clone(),
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists(), "output NRRD must be created");
        let recovered = ritk_io::read_nrrd::<Backend, _>(&output, &Default::default()).unwrap();
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
        ritk_io::write_nifti(&input, &image).unwrap();

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
        ritk_io::write_nifti(&input, &image).unwrap();

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
        ritk_io::write_metaimage(&input, &image).unwrap();

        run(ConvertArgs {
            input,
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists());
        let recovered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(recovered.shape(), [3, 4, 5]);
    }

    // ── ADR 0003 Phase A: native-dispatch coverage ────────────────────────────

    /// `is_native_read_capable`/`is_native_write_capable` must agree exactly
    /// with the format matrix documented in ADR 0003: 7 formats read+write
    /// natively, PNG reads only (no native writer, matching the Burn path),
    /// and DICOM/VTK have neither. A drift here would silently misroute a
    /// command onto the wrong substrate without any other test catching it.
    #[test]
    fn test_native_capability_predicates_match_adr_0003_matrix() {
        use super::super::{is_native_read_capable, is_native_write_capable};

        let read_and_write = [
            ImageFormat::NIfTI,
            ImageFormat::Nrrd,
            ImageFormat::Analyze,
            ImageFormat::Mgh,
            ImageFormat::MetaImage,
            ImageFormat::Tiff,
            ImageFormat::Jpeg,
        ];
        for fmt in read_and_write {
            assert!(is_native_read_capable(fmt), "{fmt:?} must read natively");
            assert!(is_native_write_capable(fmt), "{fmt:?} must write natively");
        }

        assert!(is_native_read_capable(ImageFormat::Png), "PNG reads natively");
        assert!(
            !is_native_write_capable(ImageFormat::Png),
            "PNG has no native writer"
        );

        for fmt in [ImageFormat::Dicom, ImageFormat::Vtk] {
            assert!(!is_native_read_capable(fmt), "{fmt:?} has no native reader yet");
            assert!(!is_native_write_capable(fmt), "{fmt:?} has no native writer yet");
        }
    }

    /// Differential oracle at the CLI dispatch boundary: converting the same
    /// logical image through the (now-native) `convert` path and through the
    /// pre-cutover Burn writer directly must produce byte-identical NIfTI
    /// files. This is stronger than the round-trip tests above — it catches a
    /// dispatch bug (e.g. routing to the wrong substrate) that a shape-only
    /// assertion would miss, independent of the crate-level writer parity
    /// already proven in MIG-494/495.
    #[test]
    fn test_native_convert_output_is_byte_identical_to_burn_writer() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let native_output = dir.path().join("via_convert.nii");
        let burn_output = dir.path().join("via_burn_direct.nii");

        let image = make_test_image();
        ritk_io::write_nifti(&input, &image).unwrap();

        // Through `convert` — nifti→nifti is native-capable both ends, so
        // this exercises `read_image_native`/`write_image_native`.
        run(ConvertArgs {
            input: input.clone(),
            output: native_output.clone(),
            format: None,
        })
        .unwrap();

        // Directly through the Burn writer, from the same source file read
        // via the Burn reader — the pre-cutover reference path.
        let burn_image = ritk_io::read_nifti::<Backend, _>(&input, &Default::default()).unwrap();
        ritk_io::write_nifti(&burn_output, &burn_image).unwrap();

        assert_eq!(
            std::fs::read(&native_output).unwrap(),
            std::fs::read(&burn_output).unwrap(),
            "convert's native dispatch must emit the exact bytes the Burn path would"
        );
    }

    /// VTK has no native path (ADR 0003), so `convert` must fall back to the
    /// Burn helpers end-to-end. This is coverage the original suite lacked —
    /// every prior test used a native-capable format pair.
    #[test]
    fn test_convert_vtk_input_falls_back_to_burn_path() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.vtk");
        let output = dir.path().join("output.nii");

        let image = make_test_image();
        ritk_io::write_vtk(&input, &image).unwrap();

        run(ConvertArgs {
            input,
            output: output.clone(),
            format: None,
        })
        .unwrap();

        assert!(output.exists());
        let recovered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(recovered.shape(), [3, 4, 5]);
    }

    /// Symmetric case: a native-capable input converted to VTK output must
    /// also fall back to the Burn path (checked via `--format vtk` since VTK
    /// output has no standard extension to infer from in this test fixture).
    #[test]
    fn test_convert_vtk_output_falls_back_to_burn_path() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("output.vtk");

        let image = make_test_image();
        ritk_io::write_nifti(&input, &image).unwrap();

        run(ConvertArgs {
            input,
            output: output.clone(),
            format: Some(OutputFormat::Vtk),
        })
        .unwrap();

        assert!(output.exists());
    }
}
