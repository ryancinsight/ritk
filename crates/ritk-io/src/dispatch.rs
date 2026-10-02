//! Native image format dispatch.

use crate::format;
use ritk_codecs::sample::Exact;

// ── Image format enumeration ──────────────────────────────────────────────────

/// Canonical medical image format.
///
/// Used as the single source of truth for path-to-format inference, shared by
/// the CLI, Python bindings, and any other consumer that needs to infer a format
/// from a file path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ImageFormat {
    /// NIfTI-1 or NIfTI-2 volume (`.nii` or `.nii.gz`).
    NIfTI,
    /// MetaImage volume (`.mha` or `.mhd`).
    MetaImage,
    /// Nearly Raw Raster Data volume (`.nrrd` or `.nhdr`).
    Nrrd,
    /// Grayscale Portable Network Graphics image (`.png`).
    Png,
    /// Digital Imaging and Communications in Medicine series or instance.
    Dicom,
    /// MINC2 volume (`.mnc` or `.mnc2`).
    Minc,
    /// FreeSurfer volume (`.mgh` or `.mgz`).
    Mgh,
    /// Tagged Image File Format image (`.tif` or `.tiff`).
    Tiff,
    /// Legacy VTK structured-points volume (`.vtk`).
    Vtk,
    /// Joint Photographic Experts Group image (`.jpg` or `.jpeg`).
    Jpeg,
    /// Analyze 7.5 volume (`.hdr` and `.img`).
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
            "mnc" | "mnc2" => Some(Self::Minc),
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
            Self::Minc => "minc",
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
            "minc" => Some(Self::Minc),
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
            | ImageFormat::Minc
            | ImageFormat::Mgh
            | ImageFormat::Tiff
            | ImageFormat::Vtk
            | ImageFormat::Jpeg
            | ImageFormat::Analyze
    )
}

/// True when `fmt` has a native writer in the unified `ritk-io` contract.
///
/// Format limits are enforced by each writer. PNG accepts one grayscale slice;
/// DICOM writes a derived Secondary Capture series to a directory.
#[must_use]
pub fn is_native_write_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::MetaImage
            | ImageFormat::Nrrd
            | ImageFormat::Png
            | ImageFormat::Dicom
            | ImageFormat::Minc
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
        ImageFormat::Minc => crate::ImageReader::read(
            &format::minc::native::MincReader::new(NativeBackend::default()),
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

    write_image_native_with_format(path, image, fmt)
}

/// Write a 3-D image through the native writer selected by `format`.
///
/// This form supports directory outputs and paths whose extension does not
/// identify the selected format. DICOM writes a derived Secondary Capture
/// series into `path`; it does not copy source patient or study metadata. PNG
/// writes one grayscale slice of unsigned integer samples and does not store
/// physical-space metadata.
///
/// # Errors
///
/// Returns an error when the writer rejects the image, output path, or format.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use ritk_io::{write_image_native_with_format, ImageFormat, NativeImage};
/// use ritk_spatial::{Direction, Point, Spacing};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let backend = SequentialBackend;
/// let image = NativeImage::from_flat_on(
///     vec![0.0_f32],
///     [1, 1, 1],
///     Point::new([0.0; 3]),
///     Spacing::new([1.0; 3]),
///     Direction::identity(),
///     &backend,
/// )?;
/// let directory = tempfile::tempdir()?;
/// write_image_native_with_format(
///     directory.path().join("slice.png"),
///     &image,
///     ImageFormat::Png,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_image_native_with_format<P: AsRef<std::path::Path>>(
    path: P,
    image: &NativeImage,
    format: ImageFormat,
) -> anyhow::Result<()> {
    let path = path.as_ref();
    let backend = NativeBackend::default();

    let result = match format {
        ImageFormat::Dicom => {
            return crate::format::dicom::write_dicom_series_native(path, image);
        }
        ImageFormat::NIfTI => crate::ImageWriter::write(
            &format::nifti::native::NiftiWriter::new(backend),
            path,
            image,
        ),
        ImageFormat::MetaImage => crate::ImageWriter::write(
            &format::metaimage::native::MetaImageWriter::new(backend),
            path,
            image,
        ),
        ImageFormat::Nrrd => {
            crate::ImageWriter::write(&format::nrrd::native::NrrdWriter::new(backend), path, image)
        }
        ImageFormat::Minc => {
            crate::ImageWriter::write(&format::minc::native::MincWriter::new(backend), path, image)
        }
        ImageFormat::Mgh => {
            crate::ImageWriter::write(&format::mgh::native::MghWriter::new(backend), path, image)
        }
        ImageFormat::Tiff => {
            crate::ImageWriter::write(&format::tiff::native::TiffWriter::new(backend), path, image)
        }
        ImageFormat::Jpeg => {
            crate::ImageWriter::write(&format::jpeg::native::JpegWriter::new(backend), path, image)
        }
        ImageFormat::Analyze => {
            crate::ImageWriter::write(&format::analyze::AnalyzeWriter::new(backend), path, image)
        }
        ImageFormat::Png => {
            crate::ImageWriter::write(&format::png::native::PngWriter::new(backend), path, image)
        }
        ImageFormat::Vtk => {
            crate::ImageWriter::write(&format::vtk::native::VtkWriter::new(backend), path, image)
        }
    };
    result.map_err(anyhow::Error::from)
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
            "series I/O is not yet supported for {other:?} through the native \
             dispatch; use the format-specific series writer directly"
        )),
    }
}

/// Read a 3-D f32 acquisition series through the native reader dispatch.
///
/// Each returned image shares one spatial grid and is in acquisition order.
/// A rank-3 file is a one-volume series, so this reader accepts an ordinary
/// volume; [`read_image_native`] does not accept the converse.
/// NIfTI, NRRD, and MGH stored samples use exact conversion to `f32`; values
/// that cannot be represented exactly are refused.
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
        ImageFormat::NIfTI => {
            ritk_nifti::read_nifti_series(path, &backend, Exact)
        }
        ImageFormat::Nrrd => ritk_nrrd::read_nrrd_series(path, &backend, Exact),
        ImageFormat::Mgh => ritk_mgh::read_mgh_series(path, &backend, Exact),
        other => Err(anyhow::anyhow!(
            "series I/O is not yet supported for {other:?} through the native \
             dispatch; use the format-specific series reader directly"
        )),
    }
}

#[cfg(test)]
mod native_dispatch_tests;
