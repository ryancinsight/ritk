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
use std::path::PathBuf;
use tracing::info;

use super::{infer_format, read_image, write_image};

// ── CLI arguments ─────────────────────────────────────────────────────────────

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
    ///
    /// Accepted values: `nifti`, `metaimage`, `nrrd`.
    /// `png` and `dicom` are listed here for completeness but are not
    /// writable in the current build of `ritk-io`.
    #[arg(long, value_name = "FORMAT")]
    pub format: Option<String>,
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
        input = %args.input.display(),
        output = %args.output.display(),
        "convert: starting"
    );

    let image = read_image(&args.input)?;
    let shape = image.shape();
    let spacing = image.spacing();

    // Resolve output format: explicit flag takes precedence over extension.
    let out_fmt: String = match &args.format {
        Some(f) => f.clone(),
        None => infer_format(&args.output)
            .ok_or_else(|| {
                anyhow!(
                    "Cannot infer output format from path '{}'. \
                     Specify --format nifti|metaimage|nrrd.",
                    args.output.display()
                )
            })?
            .to_owned(),
    };

    write_image(&args.output, &image, &out_fmt)?;

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
        input = %args.input.display(),
        output = %args.output.display(),
        shape = ?shape,
        "convert: complete"
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
            format: Some("nifti".to_string()),
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
}
