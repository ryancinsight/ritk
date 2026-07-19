//! Shared command infrastructure for the RITK CLI.
//!
//! Declares the subcommand modules and provides the shared IO helpers
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
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_io::{is_rgb_dicom_series, ImageFormat};
use ritk_io::{ImageReader, ImageWriter};
use std::path::Path;

// â”€â”€ Shared backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// CPU backend used by every CLI command.
///
/// `SequentialBackend` requires no GPU runtime and produces deterministic results,
/// which is appropriate for a CLI tool that must run on any host.
pub(crate) type Backend = SequentialBackend;

// â”€â”€ Format inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Infer the image format from a file-system path.
///
/// Delegates to [`ritk_io::ImageFormat::from_path`] as the SSOT for extensionâ†’format mapping.
pub(crate) fn infer_format(path: &Path) -> Option<ImageFormat> {
    ImageFormat::from_path(path)
}

// â”€â”€ Read helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Read a 3-D medical image from `path`, inferring the format from the
/// file extension.
///
/// # Errors
/// Returns an error when the extension is unrecognised or the underlying
/// reader fails.
pub(crate) fn read_image(path: &Path) -> Result<Image<f32, Backend, 3>> {
    let backend = Backend::default();
    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer input format from path: {}", path.display()))?;

    match fmt {
        ImageFormat::Dicom => {
            if is_rgb_dicom_series(path).unwrap_or(false) {
                bail!(
                    "RGB DICOM colour series are not supported by the CLI. \
                     Use `ritk-snap` (the graphical viewer) to load and inspect RGB DICOM volumes."
                );
            }
            ImageReader::read(
                &ritk_io::format::dicom::native::DicomReader::new(backend),
                path,
            )
            .with_context(|| format!("Failed to read DICOM series (native): {}", path.display()))
        }
        ImageFormat::Vtk => {
            ImageReader::read(&ritk_io::format::vtk::native::VtkReader::new(backend), path)
                .with_context(|| format!("Failed to read VTK file (native): {}", path.display()))
        }
        ImageFormat::NIfTI => ImageReader::read(
            &ritk_io::format::nifti::native::NiftiReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read NIfTI file (native): {}", path.display())),
        ImageFormat::MetaImage => ImageReader::read(
            &ritk_io::format::metaimage::native::MetaImageReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read MetaImage file (native): {}", path.display())),
        ImageFormat::Nrrd => ImageReader::read(
            &ritk_io::format::nrrd::native::NrrdReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read NRRD file (native): {}", path.display())),
        ImageFormat::Png => {
            ImageReader::read(&ritk_io::format::png::native::PngReader::new(backend), path)
                .with_context(|| format!("Failed to read PNG file (native): {}", path.display()))
        }
        ImageFormat::Mgh => {
            ImageReader::read(&ritk_io::format::mgh::native::MghReader::new(backend), path)
                .with_context(|| format!("Failed to read MGH file (native): {}", path.display()))
        }
        ImageFormat::Tiff => ImageReader::read(
            &ritk_io::format::tiff::native::TiffReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read TIFF file (native): {}", path.display())),
        ImageFormat::Jpeg => ImageReader::read(
            &ritk_io::format::jpeg::native::JpegReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read JPEG file (native): {}", path.display())),
        ImageFormat::Analyze => {
            ImageReader::read(&ritk_io::format::analyze::AnalyzeReader::new(backend), path)
                .with_context(|| {
                    format!("Failed to read Analyze file (native): {}", path.display())
                })
        }
    }
}

// â”€â”€ Write helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    image: &Image<f32, Backend, 3>,
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
        ImageFormat::Tiff => ImageWriter::write(
            &ritk_io::format::tiff::native::TiffWriter::new(Backend::default()),
            path,
            image,
        )
        .with_context(|| format!("Failed to write TIFF file: {}", path.display())),
        ImageFormat::Jpeg => ImageWriter::write(
            &ritk_io::format::jpeg::native::JpegWriter::new(Backend::default()),
            path,
            image,
        )
        .with_context(|| format!("Failed to write JPEG file: {}", path.display())),
        ImageFormat::Vtk => ImageWriter::write(
            &ritk_io::format::vtk::native::VtkWriter::new(Backend::default()),
            path,
            image,
        )
        .with_context(|| format!("Failed to write VTK file: {}", path.display())),
        ImageFormat::Png => Err(anyhow!(
            "PNG output is not supported: ritk-io has no write_png implementation. \
             Convert to NIfTI, MetaImage, or NRRD instead."
        )),
        ImageFormat::Dicom => ritk_io::write_dicom_series::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write DICOM series to: {}", path.display())),
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
pub(crate) fn write_image_inferred(path: &Path, image: &Image<f32, Backend, 3>) -> Result<()> {
    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer output format from path: {}", path.display()))?;
    write_image(path, image, fmt)
}

// â”€â”€ Capability helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// True when `fmt` has an Atlas-native reader (ADR 0003 Phase A coverage).
pub(crate) fn is_read_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::Nrrd
            | ImageFormat::Analyze
            | ImageFormat::Mgh
            | ImageFormat::MetaImage
            | ImageFormat::Tiff
            | ImageFormat::Jpeg
            | ImageFormat::Vtk
            | ImageFormat::Png
            | ImageFormat::Dicom
    )
}

/// True when `fmt` has an Atlas-native writer (ADR 0003 Phase A coverage).
///
/// Narrower than [`is_read_capable`]: PNG has no native writer.
pub(crate) fn is_write_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::Nrrd
            | ImageFormat::Analyze
            | ImageFormat::Mgh
            | ImageFormat::MetaImage
            | ImageFormat::Tiff
            | ImageFormat::Jpeg
            | ImageFormat::Vtk
    )
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    fn test_native_read_capability_tracks_dicom_cutover() {
        assert!(
            is_read_capable(ImageFormat::Dicom),
            "DICOM reads must route through the native reader"
        );
        assert!(
            !is_write_capable(ImageFormat::Dicom),
            "DICOM writes remain on the legacy writer until a native writer exists"
        );
        assert!(is_read_capable(ImageFormat::Vtk));
    }
}
