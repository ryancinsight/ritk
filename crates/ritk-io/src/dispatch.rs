//! Native image format dispatch.

use crate::format;

// ── Image format enumeration ──────────────────────────────────────────────────

/// Canonical medical image format.
///
/// Used as the single source of truth for path-to-format inference, shared by
/// the CLI, Python bindings, and any other consumer that needs to infer a format
/// from a file path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    NIfTI,
    MetaImage,
    Nrrd,
    Png,
    Dicom,
    Mgh,
    Tiff,
    Vtk,
    Jpeg,
    Analyze,
}

impl ImageFormat {
    /// Infer the image format from a file-system path.
    ///
    /// Returns `Some(format)` when the extension is recognised, `None` otherwise.
    ///
    /// `.nii.gz` is detected before the generic extension check so that the
    /// compound suffix is handled correctly.
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();

        // Compound suffix must be tested before the single-extension fallback.
        if name.ends_with(".nii.gz") || name.ends_with(".nii") {
            return Some(Self::NIfTI);
        }
        if name.ends_with(".mgh.gz") {
            return Some(Self::Mgh);
        }

        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "mha" | "mhd" => Some(Self::MetaImage),
            "nrrd" | "nhdr" => Some(Self::Nrrd),
            "png" => Some(Self::Png),
            "dcm" | "dicom" | "ima" => Some(Self::Dicom),
            "mgz" | "mgh" => Some(Self::Mgh),
            "tif" | "tiff" => Some(Self::Tiff),
            "vtk" => Some(Self::Vtk),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "hdr" | "img" => Some(Self::Analyze),
            _ => None,
        }
    }

    /// The canonical string name of this format.
    ///
    /// The returned string matches the format strings expected by
    /// `ritk-io` reader/writer dispatch in the CLI and Python bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NIfTI => "nifti",
            Self::MetaImage => "metaimage",
            Self::Nrrd => "nrrd",
            Self::Png => "png",
            Self::Dicom => "dicom",
            Self::Mgh => "mgh",
            Self::Tiff => "tiff",
            Self::Vtk => "vtk",
            Self::Jpeg => "jpeg",
            Self::Analyze => "analyze",
        }
    }

    /// Map the canonical format name string to its [`ImageFormat`] variant.
    ///
    /// Accepts the same strings produced by [`ImageFormat::as_str`].
    /// Returns `None` for unrecognised names.
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s {
            "nifti" => Some(Self::NIfTI),
            "metaimage" => Some(Self::MetaImage),
            "nrrd" => Some(Self::Nrrd),
            "png" => Some(Self::Png),
            "dicom" => Some(Self::Dicom),
            "mgh" => Some(Self::Mgh),
            "tiff" => Some(Self::Tiff),
            "vtk" => Some(Self::Vtk),
            "jpeg" => Some(Self::Jpeg),
            "analyze" => Some(Self::Analyze),
            _ => None,
        }
    }
}

// ── Native image dispatch ─────────────────────────────────────────────────────

/// Native CPU backend used by consumer-level image I/O.
///
/// `SequentialBackend` keeps file I/O deterministic and avoids pulling a device
/// runtime into CLI or Python boundary code.
pub type NativeBackend = coeus_core::SequentialBackend;

/// Native 3-D f32 image used by consumer-level image I/O.
pub type NativeImage = ritk_image::Image<f32, NativeBackend, 3>;

/// Native 3-D f32 acquisition series — one image per volume, sharing one
/// spatial grid — used by consumer-level series I/O.
pub type NativeSeries = Vec<NativeImage>;

/// True when `fmt` has a native reader in the unified `ritk-io` contract.
#[must_use]
pub fn is_native_read_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::MetaImage
            | ImageFormat::Nrrd
            | ImageFormat::Png
            | ImageFormat::Dicom
            | ImageFormat::Mgh
            | ImageFormat::Tiff
            | ImageFormat::Vtk
            | ImageFormat::Jpeg
            | ImageFormat::Analyze
    )
}

/// True when `fmt` has a native writer in the unified `ritk-io` contract.
///
/// PNG has no image writer and DICOM writes still target the legacy series
/// writer.
#[must_use]
pub fn is_native_write_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::MetaImage
            | ImageFormat::Nrrd
            | ImageFormat::Mgh
            | ImageFormat::Tiff
            | ImageFormat::Vtk
            | ImageFormat::Jpeg
            | ImageFormat::Analyze
    )
}

/// Read a 3-D f32 image through the native reader contract.
///
/// DICOM directories are accepted before extension inference because a series
/// directory has no image extension. Its ordered slices become one 3-D image
/// in a one-volume acquisition series.
///
/// # Errors
///
/// Returns an error when the path has no supported native reader or the selected
/// format reader fails.
pub fn read_image_native<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<NativeImage> {
    let path = path.as_ref();
    if path.is_dir() {
        return crate::ImageReader::read(
            &format::dicom::native::DicomReader::new(NativeBackend::default()),
            path,
        )
        .map_err(anyhow::Error::from);
    }

    let fmt = ImageFormat::from_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot infer native image input format from path: {}",
            path.display()
        )
    })?;

    match fmt {
        ImageFormat::NIfTI => crate::ImageReader::read(
            &format::nifti::native::NiftiReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::MetaImage => crate::ImageReader::read(
            &format::metaimage::native::MetaImageReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Nrrd => crate::ImageReader::read(
            &format::nrrd::native::NrrdReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Png => crate::ImageReader::read(
            &format::png::native::PngReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Dicom => crate::ImageReader::read(
            &format::dicom::native::DicomReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Mgh => crate::ImageReader::read(
            &format::mgh::native::MghReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Tiff => crate::ImageReader::read(
            &format::tiff::native::TiffReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Jpeg => crate::ImageReader::read(
            &format::jpeg::native::JpegReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Analyze => crate::ImageReader::read(
            &format::analyze::AnalyzeReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Vtk => crate::ImageReader::read(
            &format::vtk::native::VtkReader::new(NativeBackend::default()),
            path,
        ),
    }
    .map_err(anyhow::Error::from)
}

/// Write a 3-D f32 image through the native writer contract.
///
/// # Errors
///
/// Returns an error when the path has no supported native writer or the selected
/// format writer fails.
pub fn write_image_native<P: AsRef<std::path::Path>>(
    path: P,
    image: &NativeImage,
) -> anyhow::Result<()> {
    let path = path.as_ref();
    let fmt = ImageFormat::from_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot infer native image output format from path: {}",
            path.display()
        )
    })?;

    match fmt {
        ImageFormat::NIfTI => crate::ImageWriter::write(
            &format::nifti::native::NiftiWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::MetaImage => crate::ImageWriter::write(
            &format::metaimage::native::MetaImageWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Nrrd => crate::ImageWriter::write(
            &format::nrrd::native::NrrdWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Mgh => crate::ImageWriter::write(
            &format::mgh::native::MghWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Tiff => crate::ImageWriter::write(
            &format::tiff::native::TiffWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Jpeg => crate::ImageWriter::write(
            &format::jpeg::native::JpegWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Analyze => crate::ImageWriter::write(
            &format::analyze::AnalyzeWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Png => Err(std::io::Error::other(
            "PNG image writing is not implemented on the native substrate",
        )),
        ImageFormat::Dicom => Err(std::io::Error::other(
            "DICOM image writing is not implemented on the native substrate",
        )),
        ImageFormat::Vtk => crate::ImageWriter::write(
            &format::vtk::native::VtkWriter::new(NativeBackend::default()),
            path,
            image,
        ),
    }
    .map_err(anyhow::Error::from)
}

/// Write a series of volumes to `path`, inferring the format from its extension.
///
/// The counterpart to [`read_image_series_native`], over the same three formats.
/// A caller that can read a 4-D series through this module can now write one
/// back; before, only the format-specific writers could, which forced a
/// dependency on the format crate for what the dispatch already knew how to do.
///
/// Every volume must share one grid — the format writers enforce that — and the
/// first volume's geometry describes the series.
///
/// # Errors
///
/// Returns an error when the path has no supported native series writer or the
/// selected format series writer fails.
pub fn write_image_series_native<P: AsRef<std::path::Path>>(
    path: P,
    volumes: &[NativeImage],
) -> anyhow::Result<()> {
    let path = path.as_ref();
    let backend = NativeBackend::default();

    let fmt = ImageFormat::from_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot infer native series output format from path: {}",
            path.display()
        )
    })?;

    match fmt {
        ImageFormat::NIfTI => ritk_nifti::write_nifti_series(path, volumes, &backend),
        ImageFormat::Nrrd => ritk_nrrd::write_nrrd_series(path, volumes, &backend),
        ImageFormat::Mgh => ritk_mgh::write_mgh_series(path, volumes, &backend),
        other => Err(anyhow::anyhow!(
            "series I/O is not yet supported for {other:?} through the native              dispatch; use the format-specific series writer directly"
        )),
    }
}

/// Read a 3-D f32 acquisition series through the native reader dispatch.
///
/// Each returned image shares one spatial grid and is in acquisition order.
/// A rank-3 file is a one-volume series, so this reader accepts an ordinary
/// volume; [`read_image_native`] does not accept the converse.
///
/// DICOM directories are accepted before extension inference because a series
/// directory has no image extension.
///
/// # Errors
///
/// Returns an error when the path has no supported native series reader or
/// the selected format series reader fails.
pub fn read_image_series_native<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<NativeSeries> {
    let path = path.as_ref();
    let backend = NativeBackend::default();
    if path.is_dir() {
        let image = format::dicom::read_native_dicom_series(path, &backend)?;
        return Ok(vec![image]);
    }

    let fmt = ImageFormat::from_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot infer native series input format from path: {}",
            path.display()
        )
    })?;

    match fmt {
        ImageFormat::NIfTI => ritk_nifti::read_nifti_series(path, &backend),
        ImageFormat::Nrrd => ritk_nrrd::read_nrrd_series(path, &backend),
        ImageFormat::Mgh => ritk_mgh::read_mgh_series(path, &backend),
        other => Err(anyhow::anyhow!(
            "series I/O is not yet supported for {other:?} through the native \
             dispatch; use the format-specific series reader directly"
        )),
    }
}

#[cfg(test)]
mod native_dispatch_tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use ritk_spatial::{Direction, Point, Spacing};

    fn native_volume() -> NativeImage {
        let dims = [2usize, 2, 3];
        let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
        NativeImage::from_flat(
            values,
            dims,
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([0.5, 0.75, 1.25]),
            Direction::identity(),
        )
        .expect("test image")
    }

    #[test]
    fn native_capability_matrix_matches_dispatch() {
        for fmt in [
            ImageFormat::NIfTI,
            ImageFormat::MetaImage,
            ImageFormat::Nrrd,
            ImageFormat::Mgh,
            ImageFormat::Tiff,
            ImageFormat::Vtk,
            ImageFormat::Jpeg,
            ImageFormat::Analyze,
        ] {
            assert!(is_native_read_capable(fmt), "{fmt:?} must read natively");
            assert!(is_native_write_capable(fmt), "{fmt:?} must write natively");
        }
        assert!(is_native_read_capable(ImageFormat::Png));
        assert!(is_native_read_capable(ImageFormat::Dicom));
        assert!(!is_native_write_capable(ImageFormat::Png));
        assert!(!is_native_write_capable(ImageFormat::Dicom));
    }

    #[test]
    fn native_dispatch_round_trips_nrrd_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("native.nrrd");
        let image = native_volume();

        write_image_native(&path, &image).expect("native write");
        let loaded = read_image_native(&path).expect("native read");

        assert_eq!(loaded.shape(), image.shape());
        assert_eq!(loaded.data_slice().unwrap(), image.data_slice().unwrap());
        assert_eq!(loaded.origin(), image.origin());
        assert_eq!(loaded.spacing(), image.spacing());
    }

    #[test]
    fn native_dispatch_round_trips_vtk_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("native.vtk");
        let image = native_volume();

        write_image_native(&path, &image).expect("native VTK write");
        let loaded = read_image_native(&path).expect("native VTK read");
        assert_eq!(loaded.shape(), image.shape());
        assert_eq!(loaded.data_slice().unwrap(), image.data_slice().unwrap());
        assert_eq!(loaded.origin(), image.origin());
        assert_eq!(loaded.spacing(), image.spacing());
    }
}
