//! Shared command infrastructure for the RITK CLI.
//!
//! Declares the five subcommand modules and provides the shared IO helpers
//! (`infer_format`, `read_image`, `write_image`, `write_image_inferred`) and
//! the concrete `Backend` type alias used throughout all command handlers.

pub mod convert;
pub mod filter;
pub mod register;
pub mod segment;
pub mod stats;
pub mod resample;

use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend as BurnBackend;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_io::read_dicom_series;
use std::path::Path;

// ── Shared backend ────────────────────────────────────────────────────────────

/// CPU backend used by every CLI command.
///
/// `NdArray<f32>` requires no GPU runtime and produces deterministic results,
/// which is appropriate for a CLI tool that must run on any host.
pub(crate) type Backend = NdArray<f32>;

// ── Format inference ──────────────────────────────────────────────────────────

/// Infer the image format string from a file-system path.
///
/// Returns one of `"nifti"`, `"metaimage"`, `"nrrd"`, `"png"`, `"dicom"`,
/// `"mgh"`, `"tiff"`, `"vtk"`, `"jpeg"`, or `None` when the extension is
/// not recognised.
///
/// `.nii.gz` is detected before the generic extension check so that the
/// compound suffix is handled correctly.
pub(crate) fn infer_format(path: &Path) -> Option<&'static str> {
    let name = path.file_name()?.to_str()?;

    // Compound suffix must be tested before the single-extension fallback.
    if name.ends_with(".nii.gz") || name.ends_with(".nii") {
        return Some("nifti");
    }

    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "mha" | "mhd" => Some("metaimage"),
        "nrrd" | "nhdr" => Some("nrrd"),
        "png" => Some("png"),
        "dcm" | "dicom" => Some("dicom"),
        "mgz" | "mgh" => Some("mgh"),
        "tif" | "tiff" => Some("tiff"),
        "vtk" => Some("vtk"),
        "jpg" | "jpeg" => Some("jpeg"),
        _ => None,
    }
}

// ── Read helper ───────────────────────────────────────────────────────────────

/// Read a 3-D medical image from `path`, inferring the format from the
/// file extension.
///
/// # Errors
/// Returns an error when the extension is unrecognised or the underlying
/// reader fails.
pub(crate) fn read_image(path: &Path) -> Result<Image<Backend, 3>> {
    let device: <Backend as BurnBackend>::Device = Default::default();

    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer input format from path: {}", path.display()))?;

    match fmt {
        "nifti" => ritk_io::read_nifti::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read NIfTI file: {}", path.display())),
        "metaimage" => ritk_io::read_metaimage::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read MetaImage file: {}", path.display())),
        "nrrd" => ritk_io::read_nrrd::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read NRRD file: {}", path.display())),
        "png" => ritk_io::read_png_to_image::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read PNG file: {}", path.display())),
        "dicom" => read_dicom_series::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read DICOM series from: {}", path.display())),
        "mgh" => ritk_io::read_mgh::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read MGH file: {}", path.display())),
        "tiff" => ritk_io::read_tiff::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read TIFF file: {}", path.display())),
        "vtk" => ritk_io::read_vtk::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read VTK file: {}", path.display())),
        "jpeg" => ritk_io::read_jpeg::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read JPEG file: {}", path.display())),
        other => unreachable!("infer_format returned unexpected value: {other}"),
    }
}

// ── Write helpers ─────────────────────────────────────────────────────────────

/// Write `image` to `path` using the explicitly supplied `format` string.
///
/// Accepted format strings: `"nifti"`, `"metaimage"`, `"nrrd"`, `"mgh"`,
/// `"tiff"`.
/// `"png"`, `"dicom"`, `"vtk"`, and `"jpeg"` are recognised but unsupported;
/// they return a descriptive `Err`.
///
/// # Errors
/// Returns an error when the format is unsupported, unknown, or the writer
/// fails.
pub(crate) fn write_image(path: &Path, image: &Image<Backend, 3>, format: &str) -> Result<()> {
    match format {
        "nifti" => ritk_io::write_nifti::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write NIfTI file: {}", path.display())),
        "metaimage" => ritk_io::write_metaimage::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write MetaImage file: {}", path.display())),
        "nrrd" => ritk_io::write_nrrd::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write NRRD file: {}", path.display())),
        "mgh" => ritk_io::write_mgh::<Backend, _>(image, path)
            .with_context(|| format!("Failed to write MGH file: {}", path.display())),
        "tiff" => ritk_io::write_tiff::<Backend, _>(image, path)
            .with_context(|| format!("Failed to write TIFF file: {}", path.display())),
        "png" => Err(anyhow!(
            "PNG output is not supported: ritk-io has no write_png implementation. \
             Convert to NIfTI, MetaImage, or NRRD instead."
        )),
        "dicom" => ritk_io::write_dicom_series::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write DICOM series to: {}", path.display())),
        "vtk" => ritk_io::write_vtk::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write VTK file: {}", path.display())),
        "jpeg" => ritk_io::write_jpeg::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write JPEG file: {}", path.display())),
        other => Err(anyhow!(
            "Unrecognised output format '{other}'. \
             Supported write formats: nifti, metaimage, nrrd, mgh, tiff."
        )),
    }
}

/// Write `image` to `path`, inferring the output format from the path extension.
///
/// Delegates to [`write_image`] after resolving the format string.
///
/// # Errors
/// Returns an error when the extension is unrecognised or the writer fails.
pub(crate) fn write_image_inferred(path: &Path, image: &Image<Backend, 3>) -> Result<()> {
    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer output format from path: {}", path.display()))?;
    write_image(path, image, fmt)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_format_nifti_single_ext() {
        assert_eq!(infer_format(Path::new("brain.nii")), Some("nifti"));
    }

    #[test]
    fn test_infer_format_nifti_compound_ext() {
        assert_eq!(infer_format(Path::new("brain.nii.gz")), Some("nifti"));
    }

    #[test]
    fn test_infer_format_metaimage_mha() {
        assert_eq!(infer_format(Path::new("scan.mha")), Some("metaimage"));
    }

    #[test]
    fn test_infer_format_metaimage_mhd() {
        assert_eq!(infer_format(Path::new("scan.mhd")), Some("metaimage"));
    }

    #[test]
    fn test_infer_format_nrrd() {
        assert_eq!(infer_format(Path::new("volume.nrrd")), Some("nrrd"));
    }

    #[test]
    fn test_infer_format_nhdr() {
        assert_eq!(infer_format(Path::new("volume.nhdr")), Some("nrrd"));
    }

    #[test]
    fn test_infer_format_png() {
        assert_eq!(infer_format(Path::new("slice.png")), Some("png"));
    }

    #[test]
    fn test_infer_format_dicom_dcm() {
        assert_eq!(infer_format(Path::new("001.dcm")), Some("dicom"));
    }

    #[test]
    fn test_infer_format_mgh() {
        assert_eq!(infer_format(Path::new("brain.mgh")), Some("mgh"));
    }

    #[test]
    fn test_infer_format_mgz() {
        assert_eq!(infer_format(Path::new("brain.mgz")), Some("mgh"));
    }

    #[test]
    fn test_infer_format_tiff() {
        assert_eq!(infer_format(Path::new("scan.tiff")), Some("tiff"));
    }

    #[test]
    fn test_infer_format_tif() {
        assert_eq!(infer_format(Path::new("scan.tif")), Some("tiff"));
    }

    #[test]
    fn test_infer_format_vtk() {
        assert_eq!(infer_format(Path::new("model.vtk")), Some("vtk"));
    }

    #[test]
    fn test_infer_format_jpeg() {
        assert_eq!(infer_format(Path::new("photo.jpeg")), Some("jpeg"));
    }

    #[test]
    fn test_infer_format_jpg() {
        assert_eq!(infer_format(Path::new("photo.jpg")), Some("jpeg"));
    }

    #[test]
    fn test_infer_format_unknown_returns_none() {
        assert_eq!(infer_format(Path::new("data.xyz")), None);
    }

    #[test]
    fn test_infer_format_no_extension_returns_none() {
        assert_eq!(infer_format(Path::new("scandata")), None);
    }

    #[test]
    fn test_write_image_png_returns_err() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(Path::new("out.png"), &image, "png");
        assert!(result.is_err(), "PNG write must return an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("PNG output is not supported"),
            "error message must explain PNG limitation, got: {msg}"
        );
    }

    #[test]
    fn test_write_image_dicom_creates_directory() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("dicom_series");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(tensor, Point::new([0.0; 3]),
            Spacing::new([1.0; 3]), Direction::identity());
        let result = write_image(&out_path, &image, "dicom");
        assert!(result.is_ok(), "DICOM write must succeed: {:?}", result.err());
        assert!(out_path.is_dir(), "DICOM output directory must exist");
    }

    #[test]
    fn test_write_image_vtk_succeeds() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};

        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("out.vtk");

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(&out_path, &image, "vtk");
        assert!(result.is_ok(), "VTK write must succeed: {:?}", result.err());
        assert!(out_path.exists(), "VTK output file must exist");
        assert!(
            out_path.metadata().unwrap().len() > 0,
            "VTK file must be non-empty"
        );
    }

    #[test]
    fn test_write_image_jpeg_nz_gt_1_returns_err() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};

        let device: <Backend as BurnBackend>::Device = Default::default();
        // nz=2 is invalid for JPEG — must be 1
        let td = TensorData::new(vec![128.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("out.jpg");
        let result = write_image(&out_path, &image, "jpeg");
        assert!(result.is_err(), "JPEG write with nz=2 must fail");
        let err = result.unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("nz=2"),
            "error message must mention nz constraint, got: {msg}"
        );
    }

    #[test]
    fn test_write_image_jpeg_2d_succeeds() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};

        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("out.jpg");

        let device: <Backend as BurnBackend>::Device = Default::default();
        // nz=1 is valid for JPEG
        let td = TensorData::new(vec![128.0f32; 4], Shape::new([1, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(&out_path, &image, "jpeg");
        assert!(
            result.is_ok(),
            "JPEG write with nz=1 must succeed: {:?}",
            result.err()
        );
        assert!(out_path.exists(), "JPEG output file must exist");
        assert!(
            out_path.metadata().unwrap().len() > 0,
            "JPEG file must be non-empty"
        );
    }

    #[test]
    fn test_write_image_unknown_format_returns_err() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(Path::new("out.xyz"), &image, "xyz");
        assert!(result.is_err(), "unknown format must return an error");
    }
}
