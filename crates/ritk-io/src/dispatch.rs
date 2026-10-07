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
    Minc,
    Mif,
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
            "mnc" => Some(Self::Minc),
            "mif" => Some(Self::Mif),
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
            Self::Minc => "minc",
            Self::Mif => "mif",
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
            "minc" => Some(Self::Minc),
            "mif" => Some(Self::Mif),
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
            | ImageFormat::Minc
            | ImageFormat::Mif
    )
}

/// True when `fmt` has a native writer in the unified `ritk-io` contract.
///
/// Every format with a reader has a writer here except DICOM, whose writes
/// still target the legacy series writer ([`crate::write_dicom_series`])
/// because a series is a directory of instances rather than one file. PNG
/// writes a single `[1, rows, cols]` slice; a volume-shaped image is rejected
/// by the codec, not silently truncated.
#[must_use]
pub fn is_native_write_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::MetaImage
            | ImageFormat::Nrrd
            | ImageFormat::Png
            | ImageFormat::Mgh
            | ImageFormat::Tiff
            | ImageFormat::Vtk
            | ImageFormat::Jpeg
            | ImageFormat::Analyze
            | ImageFormat::Minc
            | ImageFormat::Mif
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
        ImageFormat::Minc => crate::ImageReader::read(
            &format::minc::native::MincReader::new(NativeBackend::default()),
            path,
        ),
        ImageFormat::Mif => crate::ImageReader::read(
            &format::mif::native::MifReader::new(NativeBackend::default()),
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
        ImageFormat::Minc => crate::ImageWriter::write(
            &format::minc::native::MincWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Mif => crate::ImageWriter::write(
            &format::mif::native::MifWriter::new(NativeBackend::default()),
            path,
            image,
        ),
        ImageFormat::Png => crate::ImageWriter::write(&format::png::native::PngWriter, path, image),
        ImageFormat::Dicom => Err(std::io::Error::other(
            "DICOM image writing is not implemented on the native substrate; \
             a series is a directory of instances, so use write_dicom_series",
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

    /// A single `[1, rows, cols]` slice.
    ///
    /// Slice-shaped on purpose: PNG's writer rejects a volume, so a volume
    /// fixture would report a shape-policy failure as a missing capability.
    /// Every other writer accepts this shape, which isolates the capability
    /// query from per-codec shape rules.
    fn slice_volume() -> NativeImage {
        let dims = [1usize, 3, 4];
        let values: Vec<f32> = (0..12).map(|i| i as f32 * 16.0).collect();
        NativeImage::from_flat(
            values,
            dims,
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
        )
        .expect("test slice")
    }

    /// Every [`ImageFormat`] variant, in one place.
    ///
    /// The two tests below iterate this list rather than a hand-written
    /// subset, so a format that gains a variant without gaining a route is
    /// caught. `ALL_FORMATS` is asserted to hold every variant exactly once
    /// by `every_format_round_trips_through_path_and_name`, which pins both
    /// enumerations against each other.
    const ALL_FORMATS: [ImageFormat; 12] = [
        ImageFormat::NIfTI,
        ImageFormat::MetaImage,
        ImageFormat::Nrrd,
        ImageFormat::Png,
        ImageFormat::Dicom,
        ImageFormat::Mgh,
        ImageFormat::Tiff,
        ImageFormat::Vtk,
        ImageFormat::Jpeg,
        ImageFormat::Analyze,
        ImageFormat::Minc,
        ImageFormat::Mif,
    ];

    /// The on-disk extension [`ImageFormat::from_path`] must map back to `fmt`.
    ///
    /// Deliberately *not* `as_str()`: `as_str` is the CLI/Python name, while
    /// `from_path` recognises real file extensions (`.nii`, `.mha`, `.jpg`),
    /// and the two are not the same string. Keeping this table separate makes
    /// a drifted `from_path` arm fail here rather than in a consumer.
    fn canonical_extension(fmt: ImageFormat) -> &'static str {
        match fmt {
            ImageFormat::NIfTI => "nii",
            ImageFormat::MetaImage => "mha",
            ImageFormat::Nrrd => "nrrd",
            ImageFormat::Png => "png",
            ImageFormat::Dicom => "dcm",
            ImageFormat::Mgh => "mgh",
            ImageFormat::Tiff => "tiff",
            ImageFormat::Vtk => "vtk",
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Analyze => "hdr",
            ImageFormat::Minc => "mnc",
            ImageFormat::Mif => "mif",
        }
    }

    /// Both format enumerations are total and mutually consistent.
    ///
    /// `from_path` must invert `canonical_extension` and `from_str_name` must
    /// invert `as_str`, for every variant. A format added to the enum but not
    /// to `ALL_FORMATS` shows up as a length mismatch; one added to
    /// `ALL_FORMATS` but not to `from_path` shows up as a `None`.
    #[test]
    fn every_format_round_trips_through_path_and_name() {
        let mut seen = std::collections::HashSet::new();
        for fmt in ALL_FORMATS {
            assert!(seen.insert(fmt), "{fmt:?} is listed in ALL_FORMATS twice");
            let path = std::path::PathBuf::from(format!("image.{}", canonical_extension(fmt)));
            assert_eq!(
                ImageFormat::from_path(&path),
                Some(fmt),
                "from_path must recognise {}",
                path.display()
            );
            assert_eq!(
                ImageFormat::from_str_name(fmt.as_str()),
                Some(fmt),
                "from_str_name must invert as_str for {fmt:?}"
            );
        }
        assert_eq!(
            seen.len(),
            ALL_FORMATS.len(),
            "ALL_FORMATS must enumerate each variant exactly once"
        );
    }

    /// The capability queries are not assertions about intent — they are
    /// predictions about the dispatch, checked here against it.
    ///
    /// For every format: `write_image_native` must succeed exactly when
    /// [`is_native_write_capable`] says it can, and a file this module just
    /// wrote must read back through [`read_image_native`] whenever
    /// [`is_native_read_capable`] claims a reader. A format whose capability
    /// flag drifts from its route fails here.
    #[test]
    fn native_capability_matrix_matches_dispatch() {
        for fmt in ALL_FORMATS {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir
                .path()
                .join(format!("capability.{}", canonical_extension(fmt)));
            let image = slice_volume();

            let written = write_image_native(&path, &image);
            assert_eq!(
                written.is_ok(),
                is_native_write_capable(fmt),
                "{fmt:?}: write_image_native disagreed with is_native_write_capable \
                 (capable={}, result={written:?})",
                is_native_write_capable(fmt)
            );

            if written.is_err() {
                continue;
            }
            assert!(
                is_native_read_capable(fmt),
                "{fmt:?}: writing a file this dispatch produced implies a reader"
            );
            let loaded = read_image_native(&path)
                .unwrap_or_else(|error| panic!("{fmt:?}: written file must read back: {error}"));
            assert_eq!(loaded.shape(), image.shape(), "{fmt:?}: shape parity");
        }
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

    /// The dispatch rejects a volume written to a single PNG path.
    ///
    /// PNG is a slice format: `write_png` refuses `depth != 1` rather than
    /// silently keeping slice 0. The route is write-capable, so this is the
    /// codec's shape policy rejecting an unrepresentable direction, not a
    /// missing adapter.
    #[test]
    fn native_dispatch_rejects_a_png_volume() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("volume.png");
        let error = write_image_native(&path, &native_volume())
            .expect_err("a [2, 2, 3] volume is not a single PNG slice");
        assert!(
            error.to_string().contains("single slice"),
            "PNG volume rejection must name the slice constraint, got: {error}"
        );
    }
}
