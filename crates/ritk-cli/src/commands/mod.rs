//! Shared command infrastructure for the RITK CLI.
//!
//! Declares the subcommand modules and provides the shared IO helpers
//! (`infer_format`, `read_image`, `write_image`, `write_image_inferred`) and
//! the concrete `Backend` type alias used throughout all command handlers.

pub mod convert;
pub mod dwi;
pub mod filter;
pub mod normalize;
pub mod parcellate;
pub mod register;
pub mod resample;
pub mod segment;
pub mod stats;
pub mod tract;
pub mod viewer;

use anyhow::{anyhow, Context, Result};
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_io::ImageFormat;
use std::path::Path;

// ── Shared backend ────────────────────────────────────────────────────────────

/// CPU backend used by every CLI command.
///
/// `SequentialBackend` requires no GPU runtime and produces deterministic results,
/// which is appropriate for a CLI tool that must run on any host.
pub(crate) type Backend = SequentialBackend;

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
pub(crate) fn read_image(path: &Path) -> Result<Image<f32, Backend, 3>> {
    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer input format from path: {}", path.display()))?;
    ritk_io::read_image_native(path)
        .with_context(|| format!("Failed to read {fmt:?} file (native): {}", path.display()))
}

// ── Write helpers ─────────────────────────────────────────────────────────────

/// Write `image` to `path` using the explicitly supplied `format`.
///
/// Dispatches through `ritk-io`'s native writer contract. DICOM writes a
/// directory of derived Secondary Capture slices. PNG writes one grayscale
/// slice with finite integer samples in the unsigned 16-bit range.
/// JPEG output is lossy and limited to one grayscale slice; TIFF and JPEG do
/// not retain physical-space metadata, and Analyze does not retain direction.
///
/// # Errors
/// Returns an error when the format is unsupported or the writer fails.
pub(crate) fn write_image(
    path: &Path,
    image: &Image<f32, Backend, 3>,
    format: ImageFormat,
) -> Result<()> {
    ritk_io::write_image_native_with_format(path, image, format)
        .with_context(|| format!("Failed to write {format:?} file: {}", path.display()))
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

// ── Capability helpers ────────────────────────────────────────────────────────────

/// True when `fmt` has an Atlas-native reader (ADR 0003 Phase A coverage).
pub(crate) fn is_read_capable(fmt: ImageFormat) -> bool {
    ritk_io::is_native_read_capable(fmt)
}

/// True when `fmt` has an Atlas-native writer (ADR 0003 Phase A coverage).
pub(crate) fn is_write_capable(fmt: ImageFormat) -> bool {
    ritk_io::is_native_write_capable(fmt)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

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
    fn test_infer_format_minc() {
        assert_eq!(
            infer_format(Path::new("volume.mnc")),
            Some(ImageFormat::Minc)
        );
        assert_eq!(
            infer_format(Path::new("volume.mnc2")),
            Some(ImageFormat::Minc)
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
        assert!(is_write_capable(ImageFormat::Dicom));
        assert!(is_read_capable(ImageFormat::Vtk));
    }
}
