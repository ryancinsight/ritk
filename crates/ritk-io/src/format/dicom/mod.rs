mod anonymize;
mod codec;
mod color;
mod color_common;
mod color_multiframe;
mod helpers;
mod multiframe;
pub mod networking;
mod object_model;
mod reader;
mod rt_dose;
mod rt_plan;
mod rt_struct;
mod seg;
mod series;
mod sop_class;
mod transfer_syntax;
mod writer;
mod writer_object;

pub use anonymize::{
    anonymize_dicom_directory, anonymize_dicom_file, anonymize_object, AnonymizationProfile,
    AnonymizeOptions, AnonymizeResult, AnonymizeStats, CleaningPolicy, TagAction,
};
pub use color::{
    is_rgb_dicom_series, load_dicom_color_from_series, load_dicom_color_series,
    read_dicom_color_series,
};
pub use color_multiframe::{load_dicom_color_multiframe, read_dicom_color_multiframe};
pub use multiframe::{
    load_dicom_multiframe, read_multiframe_info, write_dicom_multiframe,
    write_dicom_multiframe_with_config, write_dicom_multiframe_with_options, MultiFrameInfo,
    MultiFrameSpatialMetadata, MultiFrameWriterConfig,
};
pub use networking::dimse::{CommandField, DimseMessage, DimseStatus};
pub use networking::pdu::{AssociateAcPdu, AssociateRqPdu, Pdu};
pub use networking::{
    echo as dicom_echo, find as dicom_find, retrieve as dicom_retrieve,
    retrieve_series as dicom_retrieve_series, store as dicom_store, AeTitle, Association,
    AssociationConfig, DicomAddress, EchoResponse, FindLevel, FindQuery, FindResult,
    MoveDestination, MoveResponse, MoveResult, NetworkingError, ScpConfig, StoreResponse, StoreScp,
    StoreScpHandle, StoredInstance,
};
pub use object_model::{
    is_private_tag, DicomObjectModel, DicomObjectNode, DicomPreservationSet, DicomPreservedElement,
    DicomSequenceItem, DicomTag, DicomValue,
};
pub use reader::{
    literal_arraystring, load_dicom_from_series, load_dicom_series_with_metadata,
    load_native_dicom_from_series, load_native_dicom_series_with_metadata,
    read_dicom_series_with_metadata, read_native_dicom_series_with_metadata, scan_dicom_instances,
    scan_dicom_part10_bytes, DicomReadMetadata, DicomSliceMetadata, PatientPosition,
    ScannedDicomSeries,
};
pub use rt_dose::{
    read_rt_dose, write_rt_dose, RtDoseGrid, RtDoseSummationType, RtDoseType, RT_DOSE_SOP_CLASS_UID,
};
pub use rt_plan::{
    read_rt_plan, write_rt_plan, RtBeamInfo, RtFractionGroup, RtPlanInfo, RT_PLAN_SOP_CLASS_UID,
};
pub use rt_struct::{
    label_map_to_rt_struct, read_rt_struct, rt_roi_to_polydata, write_rt_struct,
    ContourGeometricType, RtContour, RtRoiInfo, RtRoiInterpretedType, RtStructureSet,
};
pub use seg::{
    dicom_seg_to_label_map, label_map_to_dicom_seg, read_dicom_seg, write_dicom_seg,
    DicomSegmentInfo, DicomSegmentation, SegEncoding, SegmentAlgorithmType, SegmentationType,
};
// Re-export series types and functions from the series submodule.
pub use ritk_dicom::PixelSignedness;
pub use series::{
    load_dicom_series, load_native_dicom_series, read_dicom_series, read_native_dicom_series,
    scan_dicom_directory, DicomReader, DicomSeriesInfo,
};
pub use transfer_syntax::TransferSyntaxKind;
pub use writer::{write_dicom_series, write_dicom_series_with_metadata, DicomWriter};
pub use writer_object::{model_to_in_mem, write_object as write_dicom_object};

/// Atlas-native-substrate DICOM reader implementing the unified image reader contract.
pub mod native {
    use crate::domain::{to_io_err, ImageReader};
    use coeus_core::ComputeBackend;
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound DICOM series reader.
    pub struct DicomReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> DicomReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for DicomReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            super::read_native_dicom_series_with_metadata(path, &self.backend)
                .map(|(image, _metadata)| image)
                .map_err(to_io_err)
        }
    }
}
