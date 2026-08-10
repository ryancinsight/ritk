pub mod domain;
pub mod format;

pub use domain::{ImageReader, ImageWriter};

pub use domain::{
    AttributeArray, VtkDataObject, VtkFilter, VtkPipeline, VtkPolyData, VtkSink, VtkSource,
    VtkStructuredGrid, VtkUnstructuredGrid,
};
pub use format::analyze::{read_analyze, write_analyze, AnalyzeReader, AnalyzeWriter};
pub use format::dicom::{
    anonymize_dicom_directory, anonymize_dicom_directory_verified, anonymize_dicom_file,
    anonymize_dicom_file_verified, anonymize_object, dicom_echo, dicom_find, dicom_retrieve,
    dicom_retrieve_series, dicom_seg_to_label_map, dicom_store, is_private_tag,
    is_rgb_dicom_series, label_map_to_dicom_seg, label_map_to_rt_struct, literal_arraystring,
    load_atlas_color_multiframe, load_color_multiframe_flat, load_color_volume_flat,
    load_color_volume_flat_from_path, load_dicom_from_series, load_dicom_multiframe,
    load_dicom_multiframe_flat, load_dicom_multiframe_native, load_dicom_series,
    load_dicom_series_with_metadata, load_native_dicom_series, model_to_in_mem,
    read_dicom_gradient_scheme_from_file, read_dicom_gradient_scheme_from_files, read_dicom_seg,
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

mod dispatch;
pub use dispatch::{
    is_native_read_capable, is_native_write_capable, read_image_native, read_image_series_native,
    write_image_native, ImageFormat, NativeBackend, NativeImage, NativeSeries,
};
