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
    load_dicom_series_with_metadata, load_native_dicom_series, model_to_in_mem, read_dicom_seg,
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
pub use format::mgh::{read_mgh, write_mgh, MghReader, MghWriter};
pub use format::nifti::{
    read_nifti, read_nifti_from_bytes, read_nifti_from_bytes_native, read_nifti_labels,
    write_nifti, write_nifti_labels, NiftiReader, NiftiWriter,
};
pub use format::nrrd::{read_nrrd, write_nrrd, write_nrrd_with_data, NrrdReader, NrrdWriter};
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

// â”€â”€ Image format enumeration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ Native image dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Native CPU backend used by consumer-level image I/O.
///
/// `SequentialBackend` keeps file I/O deterministic and avoids pulling a device
/// runtime into CLI or Python boundary code.
pub type NativeBackend = coeus_core::SequentialBackend;

/// Native 3-D f32 image used by consumer-level image I/O.
pub type NativeImage = ritk_image::Image<f32, NativeBackend, 3>;

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
}
