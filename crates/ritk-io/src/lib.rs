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
    load_dicom_color_from_series, load_dicom_color_series, load_dicom_from_series,
    load_dicom_multiframe, load_dicom_series, load_dicom_series_with_metadata, model_to_in_mem,
    read_dicom_color_series, read_dicom_seg, read_dicom_series, read_dicom_series_with_metadata,
    read_multiframe_info, read_rt_dose, read_rt_plan, read_rt_struct, rt_roi_to_polydata,
    scan_dicom_directory, scan_dicom_instances, scan_dicom_part10_bytes, write_dicom_multiframe,
    write_dicom_multiframe_with_config, write_dicom_multiframe_with_options, write_dicom_object,
    write_dicom_seg, write_dicom_series, write_dicom_series_with_metadata, write_rt_dose,
    write_rt_plan, write_rt_struct, AeTitle, AnonymizationProfile, AnonymizeOptions,
    AnonymizeResult, AnonymizeStats, AssociationConfig, CleaningPolicy, DicomAddress,
    DicomObjectModel, DicomObjectNode, DicomPreservationSet, DicomPreservedElement,
    DicomReadMetadata, DicomSegmentInfo, DicomSegmentation, DicomSequenceItem, DicomSeriesInfo,
    DicomSliceMetadata, DicomTag, DicomValue, DicomWriter, EchoResponse, FindLevel, FindQuery,
    FindResult, MoveDestination, MoveResponse, MultiFrameInfo, MultiFrameSpatialMetadata,
    MultiFrameWriterConfig, NetworkingError, PatientPosition, PixelSignedness, RtBeamInfo,
    RtContour, RtDoseGrid, RtFractionGroup, RtPlanInfo, RtRoiInfo, RtStructureSet,
    ScannedDicomSeries, ScpConfig, SegEncoding, StoreResponse, StoreScp, StoreScpHandle,
    StoredInstance, TagAction, TransferSyntaxKind, RT_DOSE_SOP_CLASS_UID, RT_PLAN_SOP_CLASS_UID,
};
pub use format::dicomweb::{DicomWebClient, QidoSearchParams, StowFailure, StowResponse};
pub use format::jpeg::{read_jpeg, write_jpeg, JpegReader, JpegWriter};
pub use format::metaimage::{read_metaimage, write_metaimage, MetaImageReader, MetaImageWriter};
pub use format::mgh::{read_mgh, write_mgh, MghReader, MghWriter};
pub use format::minc::{read_minc, write_minc, MincReader, MincWriter};
pub use format::nifti::{
    read_nifti, read_nifti_from_bytes, read_nifti_labels, write_nifti, write_nifti_labels,
    NiftiReader, NiftiWriter,
};
pub use format::nrrd::{read_nrrd, write_nrrd, NrrdReader, NrrdWriter};
pub use format::png::{read_png_series, read_png_to_image, PngReader, PngSeriesReader};
pub use format::tiff::{read_tiff, write_tiff, TiffReader, TiffWriter};
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
pub use format::vtk::{read_vtk, write_vtk, VtkReader, VtkWriter};

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
            "dcm" | "dicom" => Some(Self::Dicom),
            "mgz" | "mgh" => Some(Self::Mgh),
            "tif" | "tiff" => Some(Self::Tiff),
            "vtk" => Some(Self::Vtk),
            "jpg" | "jpeg" => Some(Self::Jpeg),
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
        }
    }
}
