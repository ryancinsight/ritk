pub mod domain;
pub mod format;

pub use domain::{ImageReader, ImageWriter};

pub use domain::{
    AttributeArray, VtkDataObject, VtkFilter, VtkPipeline, VtkPolyData, VtkSink, VtkSource,
    VtkStructuredGrid, VtkUnstructuredGrid,
};
pub use format::analyze::{read_analyze, write_analyze, AnalyzeReader, AnalyzeWriter};
pub use format::dicom::{
    anonymize_dicom_directory, anonymize_dicom_file, anonymize_object, dicom_echo, dicom_find,
    dicom_retrieve, dicom_retrieve_series, dicom_seg_to_label_map, dicom_store, is_private_tag,
    is_rgb_dicom_series, label_map_to_dicom_seg, label_map_to_rt_struct, literal_arraystring,
    load_atlas_color_multiframe, load_color_multiframe_flat, load_color_volume_flat,
    load_color_volume_flat_from_path, load_dicom_from_series, load_dicom_multiframe,
    load_dicom_multiframe_flat, load_dicom_multiframe_native, load_dicom_series,
    load_dicom_series_with_metadata, load_native_dicom_series, model_to_in_mem,
    read_dicom_gradient_scheme_from_file, read_dicom_gradient_scheme_from_series, read_dicom_seg,
    read_dicom_series, read_dicom_series_with_metadata, read_multiframe_info,
    read_native_dicom_series, read_rt_dose, read_rt_plan, read_rt_struct, rt_roi_to_polydata,
    scan_dicom_directory, scan_dicom_instances, scan_dicom_part10_bytes, write_dicom_multiframe,
    write_dicom_multiframe_native, write_dicom_multiframe_native_with_config,
    write_dicom_multiframe_native_with_options, write_dicom_multiframe_with_config,
    write_dicom_multiframe_with_options, write_dicom_object, write_dicom_seg, write_dicom_series,
    write_dicom_series_native, write_dicom_series_with_metadata, write_rt_dose, write_rt_plan,
    write_rt_struct, AeTitle, AnonymizationProfile, AnonymizeOptions, AnonymizeResult,
    AnonymizeStats, AssociationConfig, CleaningPolicy, ColorMultiFrameVolume, ContourGeometricType,
    DicomAddress, DicomObjectModel, DicomObjectNode, DicomPreservationSet, DicomPreservedElement,
    DicomReadMetadata, DicomSegmentInfo, DicomSegmentation, DicomSequenceItem, DicomSeriesInfo,
    DicomSliceMetadata, DicomTag, DicomValue, DicomWriter, EchoResponse, FindLevel, FindQuery,
    FindResult, MoveDestination, MoveResponse, MultiFrameInfo, MultiFrameSpatialMetadata,
    MultiFrameVolume, MultiFrameWriterConfig, NetworkingError, PatientPosition, PixelSignedness,
    RtBeamInfo, RtContour, RtDoseGrid, RtDoseSummationType, RtDoseType, RtFractionGroup,
    RtPlanInfo, RtRoiInfo, RtRoiInterpretedType, RtStructureSet, ScannedDicomSeries, ScpConfig,
    SegEncoding, SegmentAlgorithmType, SegmentationType, StoreResponse, StoreScp, StoreScpHandle,
    StoredInstance, TagAction, TransferSyntaxKind, RT_DOSE_SOP_CLASS_UID, RT_PLAN_SOP_CLASS_UID,
};
pub use format::dicomweb::{DicomWebClient, QidoSearchParams, StowFailure, StowResponse};
pub use format::jpeg::{read_jpeg_color_to_volume, JpegColorReader};
pub use format::metaimage::{
    read_metaimage, write_metaimage, write_metaimage_with_data, MetaImageReader, MetaImageWriter,
};
pub use format::mgh::{read_mgh, read_mgh_series, write_mgh, MghReader, MghWriter};
pub use format::nifti::{
    read_nifti, read_nifti_from_bytes, read_nifti_from_bytes_native, read_nifti_labels,
    read_nifti_series, write_nifti, write_nifti_labels, NiftiReader, NiftiWriter,
};
pub use format::nrrd::{
    read_nrrd, read_nrrd_series, write_nrrd, write_nrrd_with_data, NrrdReader, NrrdWriter,
};
pub use format::png::{
    read_png_color_series, read_png_color_to_volume, PngColorReader, PngColorSeriesReader,
};
pub use format::tiff::{read_tiff_color_to_volume, TiffColorReader};
pub use format::vtk::image_xml::{
    read_vti_binary_appended, read_vti_binary_appended_bytes, write_vti_binary_appended_bytes,
    write_vti_binary_appended_to_file,
};
pub use format::vtk::{mesh_to_vtk_string, write_mesh_as_vtk};
pub use format::vtk::{
    read_obj_mesh, read_ply_mesh, read_stl_mesh, read_vtk_polydata, read_vtp_polydata, write_gltf,
    write_obj_mesh, write_ply_ascii, write_ply_binary_le, write_stl_ascii, write_stl_binary,
    write_vtk_polydata, write_vtp_polydata,
};

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
        let name = path.file_name()?.to_str()?;

        // Compound suffix must be tested before the single-extension fallback.
        if name.ends_with(".nii.gz") || name.ends_with(".nii") {
            return Some(Self::NIfTI);
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
/// directory has no image extension.
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
    if path.is_dir() {
        return Err(anyhow::anyhow!(
            "series I/O from a DICOM directory is not yet supported through \
             the native series dispatch; use `read_native_dicom_series` directly"
        ));
    }

    let fmt = ImageFormat::from_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot infer native series input format from path: {}",
            path.display()
        )
    })?;

    let backend = NativeBackend::default();
    match fmt {
        ImageFormat::NIfTI => {
            ritk_nifti::read_nifti_series(path, &backend).map_err(anyhow::Error::from)
        }
        ImageFormat::Nrrd => {
            ritk_nrrd::read_nrrd_series(path, &backend).map_err(anyhow::Error::from)
        }
        ImageFormat::Mgh => ritk_mgh::read_mgh_series(path, &backend).map_err(anyhow::Error::from),
        other => Err(anyhow::anyhow!(
            "series I/O is not yet supported for {other:?} through the native \
             dispatch; use the format-specific series reader directly"
        )),
    }
    .map_err(anyhow::Error::from)
}

#[cfg(test)]
mod native_dispatch_tests {
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

    // ── Native series dispatch round-trip tests ──────────────────────────────

    /// Build `volumes` images on one spatial grid.
    ///
    /// Volume `v` is filled with distinct per-voxel values so an ordering or
    /// offset error in the series reader is detectable by value, not only by
    /// length.
    fn native_series_fixture(volumes: usize, dims: [usize; 3]) -> NativeSeries {
        let n = dims[0] * dims[1] * dims[2];
        let backend = NativeBackend::default();
        (0..volumes)
            .map(|v| {
                let values: Vec<f32> = (0..n).map(|i| (v * 100 + i) as f32 * 0.5 - 1.0).collect();
                NativeImage::from_flat_on(
                    values,
                    dims,
                    Point::new([1.0, 2.0, 3.0]),
                    Spacing::new([0.5, 0.75, 1.25]),
                    Direction::identity(),
                    &backend,
                )
                .expect("series fixture image")
            })
            .collect()
    }

    fn assert_series_matches(actual: &NativeSeries, expected: &NativeSeries, context: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context}: volume count round-trip"
        );
        for (position, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                got.shape(),
                want.shape(),
                "{context}: volume {position} shape"
            );
            assert_eq!(
                got.data_slice().expect("contiguous host voxels"),
                want.data_slice().expect("contiguous host voxels"),
                "{context}: volume {position} voxels"
            );
            assert_eq!(
                got.origin(),
                want.origin(),
                "{context}: volume {position} origin"
            );
            assert_eq!(
                got.spacing(),
                want.spacing(),
                "{context}: volume {position} spacing"
            );
            assert_eq!(
                got.direction(),
                want.direction(),
                "{context}: volume {position} direction"
            );
        }
    }

    #[test]
    fn native_dispatch_reads_nifti_series() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("series.nii");
        let backend = NativeBackend::default();
        let expected = native_series_fixture(4, [2, 3, 4]);

        ritk_nifti::write_nifti_series(&path, &expected, &backend).expect("write NIfTI series");
        let actual = read_image_series_native(&path).expect("read via dispatch");

        assert_series_matches(&actual, &expected, "NIfTI series dispatch");
    }

    #[test]
    fn native_dispatch_reads_nifti_gzip_series() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("series.nii.gz");
        let backend = NativeBackend::default();
        let expected = native_series_fixture(3, [2, 2, 3]);

        ritk_nifti::write_nifti_series(&path, &expected, &backend)
            .expect("write gzipped NIfTI series");
        let actual = read_image_series_native(&path).expect("read via dispatch");

        assert_series_matches(&actual, &expected, "gzipped NIfTI series dispatch");
    }

    #[test]
    fn native_dispatch_reads_nrrd_series() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("series.nrrd");
        let backend = NativeBackend::default();
        let expected = native_series_fixture(5, [2, 3, 4]);

        ritk_nrrd::write_nrrd_series(&path, &expected, &backend).expect("write NRRD series");
        let actual = read_image_series_native(&path).expect("read via dispatch");

        assert_series_matches(&actual, &expected, "NRRD series dispatch");
    }

    #[test]
    fn native_dispatch_reads_mgh_series() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("series.mgh");
        let backend = NativeBackend::default();
        let expected = native_series_fixture(6, [2, 3, 2]);

        ritk_mgh::write_mgh_series(&path, &expected, &backend).expect("write MGH series");
        let actual = read_image_series_native(&path).expect("read via dispatch");

        assert_series_matches(&actual, &expected, "MGH series dispatch");
    }

    #[test]
    fn native_dispatch_reads_mgz_series() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("series.mgz");
        let backend = NativeBackend::default();
        let expected = native_series_fixture(3, [2, 2, 3]);

        ritk_mgh::write_mgh_series(&path, &expected, &backend).expect("write MGZ series");
        let actual = read_image_series_native(&path).expect("read via dispatch");

        assert_series_matches(&actual, &expected, "gzipped MGH series dispatch");
    }

    #[test]
    fn native_dispatch_reads_single_volume_series() {
        // A rank-3 file is a one-volume series; every codec must accept it
        // through the series dispatcher.
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = NativeBackend::default();
        let expected = native_series_fixture(1, [2, 3, 4]);

        let nifti_path = dir.path().join("one.nii");
        ritk_nifti::write_nifti_series(&nifti_path, &expected, &backend)
            .expect("write single-volume NIfTI series");
        let nifti_actual =
            read_image_series_native(&nifti_path).expect("read single-volume NIfTI series");
        assert_series_matches(&nifti_actual, &expected, "NIfTI single-volume series");

        let nrrd_path = dir.path().join("one.nrrd");
        ritk_nrrd::write_nrrd_series(&nrrd_path, &expected, &backend)
            .expect("write single-volume NRRD series");
        let nrrd_actual =
            read_image_series_native(&nrrd_path).expect("read single-volume NRRD series");
        assert_series_matches(&nrrd_actual, &expected, "NRRD single-volume series");

        let mgh_path = dir.path().join("one.mgh");
        ritk_mgh::write_mgh_series(&mgh_path, &expected, &backend)
            .expect("write single-volume MGH series");
        let mgh_actual =
            read_image_series_native(&mgh_path).expect("read single-volume MGH series");
        assert_series_matches(&mgh_actual, &expected, "MGH single-volume series");
    }

    #[test]
    fn cross_codec_series_differential_nifti_nrrd_mgh() {
        // ADR 0036 verification condition 8: write the same series to all
        // three format codecs, read each back, and assert they produce
        // identical voxel values and spatial metadata.
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = NativeBackend::default();
        let expected = native_series_fixture(4, [3, 4, 5]);

        // Write the same fixture to all three formats.
        let nii_path = dir.path().join("differential.nii");
        let nrrd_path = dir.path().join("differential.nrrd");
        let mgh_path = dir.path().join("differential.mgh");

        ritk_nifti::write_nifti_series(&nii_path, &expected, &backend).expect("write NIfTI");
        ritk_nrrd::write_nrrd_series(&nrrd_path, &expected, &backend).expect("write NRRD");
        ritk_mgh::write_mgh_series(&mgh_path, &expected, &backend).expect("write MGH");

        // Read each back through the unified dispatch.
        let nii = read_image_series_native(&nii_path).expect("read NIfTI");
        let nrrd = read_image_series_native(&nrrd_path).expect("read NRRD");
        let mgh = read_image_series_native(&mgh_path).expect("read MGH");

        // All three must agree with the original fixture.
        assert_series_matches(&nii, &expected, "NIfTI vs fixture");
        assert_series_matches(&nrrd, &expected, "NRRD vs fixture");
        assert_series_matches(&mgh, &expected, "MGH vs fixture");

        // Cross-codec: all three must agree with each other.
        assert_series_matches(&nii, &nrrd, "NIfTI vs NRRD");
        assert_series_matches(&nii, &mgh, "NIfTI vs MGH");
        assert_series_matches(&nrrd, &mgh, "NRRD vs MGH");
    }

    #[test]
    fn native_dispatch_rejects_unsupported_series_format() {
        // Only NIfTI, NRRD, and MGH are routed through the series dispatch.
        let dir = tempfile::tempdir().expect("tempdir");
        let vtk_path = dir.path().join("image.vtk");

        // Write a single 3-D VTK image so the path exists and has a valid
        // extension; the dispatch must reject it, not the reader.
        let image = native_volume();
        write_image_native(&vtk_path, &image).expect("write VTK image");

        let err = read_image_series_native(&vtk_path)
            .expect_err("VTK has no series reader in the native dispatch");
        assert!(
            format!("{err:#}").contains("not yet supported"),
            "error must name the unsupported format, got: {err:#}"
        );
    }
}
