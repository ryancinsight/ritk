//! Shared command infrastructure for the RITK CLI.
//!
//! Declares the five subcommand modules and provides the shared IO helpers
//! (`infer_format`, `read_image`, `write_image`, `write_image_inferred`) and
//! the concrete `Backend` type alias used throughout all command handlers.

pub mod convert;
pub mod filter;
pub mod normalize;
pub mod register;
pub mod resample;
pub mod segment;
pub mod stats;
pub mod viewer;

use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::backend::Backend as BurnBackend;
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_io::{is_rgb_dicom_series, read_dicom_series, ImageFormat};
use std::path::Path;

// ── Shared backend ────────────────────────────────────────────────────────────

/// CPU backend used by every CLI command.
///
/// `NdArray<f32>` requires no GPU runtime and produces deterministic results,
/// which is appropriate for a CLI tool that must run on any host.
pub(crate) type Backend = NdArray<f32>;

// ── Format inference ──────────────────────────────────────────────────────────

/// Infer the image format from a file-system path.
///
/// Delegates to [`ritk_io::ImageFormat::from_path`] as the SSOT for extension→format mapping.
pub(crate) fn infer_format(path: &Path) -> Option<ImageFormat> {
    ImageFormat::from_path(path)
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
        ImageFormat::NIfTI => ritk_io::read_nifti::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read NIfTI file: {}", path.display())),
        ImageFormat::MetaImage => ritk_io::read_metaimage::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read MetaImage file: {}", path.display())),
        ImageFormat::Nrrd => ritk_io::read_nrrd::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read NRRD file: {}", path.display())),
        ImageFormat::Png => ritk_io::read_png_to_image::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read PNG file: {}", path.display())),
        ImageFormat::Dicom => {
            if is_rgb_dicom_series(path).unwrap_or(false) {
                bail!(
                    "RGB DICOM colour series are not supported by the CLI. \
                     Use `ritk-snap` (the graphical viewer) to load and inspect RGB DICOM volumes."
                );
            }
            read_dicom_series::<Backend, _>(path, &device)
                .with_context(|| format!("Failed to read DICOM series from: {}", path.display()))
        }
        ImageFormat::Mgh => ritk_io::read_mgh::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read MGH file: {}", path.display())),
        ImageFormat::Tiff => ritk_io::read_tiff::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read TIFF file: {}", path.display())),
        ImageFormat::Vtk => ritk_io::read_vtk::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read VTK file: {}", path.display())),
        ImageFormat::Jpeg => ritk_io::read_jpeg::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read JPEG file: {}", path.display())),
        ImageFormat::Analyze => ritk_io::read_analyze::<Backend, _>(path, &device)
            .with_context(|| format!("Failed to read Analyze file: {}", path.display())),
    }
}

// ── Write helpers ─────────────────────────────────────────────────────────────

/// Write `image` to `path` using the explicitly supplied `format`.
///
/// Accepted formats: `NIfTI`, `MetaImage`, `Nrrd`, `Mgh`, `Tiff`, `Vtk`,
/// `Jpeg`, `Analyze`.
/// `Png` is recognised but unsupported (returns a descriptive `Err`).
/// `Dicom` write is supported via [`ritk_io::write_dicom_series`].
///
/// # Errors
/// Returns an error when the format is unsupported or the writer fails.
pub(crate) fn write_image(
    path: &Path,
    image: &Image<Backend, 3>,
    format: ImageFormat,
) -> Result<()> {
    match format {
        ImageFormat::NIfTI => ritk_io::write_nifti::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write NIfTI file: {}", path.display())),
        ImageFormat::MetaImage => ritk_io::write_metaimage::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write MetaImage file: {}", path.display())),
        ImageFormat::Nrrd => ritk_io::write_nrrd::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write NRRD file: {}", path.display())),
        ImageFormat::Mgh => ritk_io::write_mgh::<Backend, _>(image, path)
            .with_context(|| format!("Failed to write MGH file: {}", path.display())),
        ImageFormat::Tiff => ritk_io::write_tiff::<Backend, _>(image, path)
            .with_context(|| format!("Failed to write TIFF file: {}", path.display())),
        ImageFormat::Png => Err(anyhow!(
            "PNG output is not supported: ritk-io has no write_png implementation. \
             Convert to NIfTI, MetaImage, or NRRD instead."
        )),
        ImageFormat::Dicom => ritk_io::write_dicom_series::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write DICOM series to: {}", path.display())),
        ImageFormat::Vtk => ritk_io::write_vtk::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write VTK file: {}", path.display())),
        ImageFormat::Jpeg => ritk_io::write_jpeg::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write JPEG file: {}", path.display())),
        ImageFormat::Analyze => ritk_io::write_analyze::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write Analyze file: {}", path.display())),
    }
}

/// Write `image` to `path`, inferring the output format from the path extension.
///
/// Delegates to [`write_image`] after resolving the format.
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
    use ritk_io::ImageFormat;

    #[test]
    fn test_infer_format_nifti_single_ext() {
        assert_eq!(
            infer_format(Path::new("brain.nii")),
            Some(ImageFormat::NIfTI)
        );
    }

    #[test]
    fn test_infer_format_nifti_compound_ext() {
        assert_eq!(
            infer_format(Path::new("brain.nii.gz")),
            Some(ImageFormat::NIfTI)
        );
    }

    #[test]
    fn test_infer_format_metaimage_mha() {
        assert_eq!(
            infer_format(Path::new("scan.mha")),
            Some(ImageFormat::MetaImage)
        );
    }

    #[test]
    fn test_infer_format_metaimage_mhd() {
        assert_eq!(
            infer_format(Path::new("scan.mhd")),
            Some(ImageFormat::MetaImage)
        );
    }

    #[test]
    fn test_infer_format_nrrd() {
        assert_eq!(
            infer_format(Path::new("volume.nrrd")),
            Some(ImageFormat::Nrrd)
        );
    }

    #[test]
    fn test_infer_format_nhdr() {
        assert_eq!(
            infer_format(Path::new("volume.nhdr")),
            Some(ImageFormat::Nrrd)
        );
    }

    #[test]
    fn test_infer_format_png() {
        assert_eq!(infer_format(Path::new("slice.png")), Some(ImageFormat::Png));
    }

    #[test]
    fn test_infer_format_dicom_dcm() {
        assert_eq!(infer_format(Path::new("001.dcm")), Some(ImageFormat::Dicom));
    }

    #[test]
    fn test_infer_format_mgh() {
        assert_eq!(infer_format(Path::new("brain.mgh")), Some(ImageFormat::Mgh));
    }

    #[test]
    fn test_infer_format_mgz() {
        assert_eq!(infer_format(Path::new("brain.mgz")), Some(ImageFormat::Mgh));
    }

    #[test]
    fn test_infer_format_tiff() {
        assert_eq!(
            infer_format(Path::new("scan.tiff")),
            Some(ImageFormat::Tiff)
        );
    }

    #[test]
    fn test_infer_format_tif() {
        assert_eq!(infer_format(Path::new("scan.tif")), Some(ImageFormat::Tiff));
    }

    #[test]
    fn test_infer_format_vtk() {
        assert_eq!(infer_format(Path::new("model.vtk")), Some(ImageFormat::Vtk));
    }

    #[test]
    fn test_infer_format_jpeg() {
        assert_eq!(
            infer_format(Path::new("photo.jpeg")),
            Some(ImageFormat::Jpeg)
        );
    }

    #[test]
    fn test_infer_format_jpg() {
        assert_eq!(
            infer_format(Path::new("photo.jpg")),
            Some(ImageFormat::Jpeg)
        );
    }

    #[test]
    fn test_infer_format_analyze_hdr() {
        assert_eq!(
            infer_format(Path::new("brain.hdr")),
            Some(ImageFormat::Analyze)
        );
    }

    #[test]
    fn test_infer_format_analyze_img() {
        assert_eq!(
            infer_format(Path::new("brain.img")),
            Some(ImageFormat::Analyze)
        );
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
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(Path::new("out.png"), &image, ImageFormat::Png);
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
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("dicom_series");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(&out_path, &image, ImageFormat::Dicom);
        assert!(
            result.is_ok(),
            "DICOM write must succeed: {:?}",
            result.err()
        );
        assert!(out_path.is_dir(), "DICOM output directory must exist");
    }

    #[test]
    fn test_write_image_vtk_succeeds() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

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
        let result = write_image(&out_path, &image, ImageFormat::Vtk);
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
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

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
        let result = write_image(&out_path, &image, ImageFormat::Jpeg);
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
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

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
        let result = write_image(&out_path, &image, ImageFormat::Jpeg);
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
}
