//! DICOM image format support.
//!
//! This module provides the canonical DICOM series entry points for `ritk-io`.
//! The actual reader and writer implementations live in the sibling `reader`
//! and `writer` modules.
//!
//! # Public API
//!
//! ## Series I/O
//! - `scan_dicom_directory`
//! - `read_dicom_series`
//! - `load_dicom_series`
//! - `read_dicom_series_with_metadata`
//! - `load_dicom_series_with_metadata`
//! - `write_dicom_series`
//! - `write_dicom_series_with_metadata`
//! - `DicomSeriesInfo`
//! - `DicomReadMetadata`
//! - `DicomSliceMetadata`
//!
//! ## Multi-Frame I/O
//! - `read_multiframe_info`
//! - `load_dicom_multiframe`
//! - `write_dicom_multiframe`
//! - `write_dicom_multiframe_with_options`
//! - `write_dicom_multiframe_with_config`
//! - `MultiFrameInfo`
//! - `MultiFrameSpatialMetadata`
//! - `MultiFrameWriterConfig`
//!
//! ## Object Model
//! - `DicomObjectModel`, `DicomObjectNode`, `DicomSequenceItem`
//! - `DicomTag`, `DicomValue`
//! - `DicomPreservationSet`, `DicomPreservedElement`
//! - `is_private_tag`
//!
//! # Example
//!
//! ```rust,ignore
//! use ritk_io::format::dicom::{read_dicom_series, scan_dicom_directory};
//!
//! let series = scan_dicom_directory("study/")?;
//! let image = read_dicom_series::<burn_ndarray::NdArray<f32>, _>("study/", &device)?;
//! ```
//!
//! The module re-exports the reader and writer types so `ritk_io::read_*`
//! and `ritk_io::write_*` remain the authoritative crate-level entry points.

mod codec;
pub mod multiframe;
pub mod object_model;
pub mod reader;
pub mod rt_dose;
pub mod rt_plan;
pub mod rt_struct;
pub mod seg;
pub mod sop_class;
pub mod transfer_syntax;
pub mod writer;
pub mod writer_object;

pub use multiframe::{
    load_dicom_multiframe, read_multiframe_info, write_dicom_multiframe,
    write_dicom_multiframe_with_config, write_dicom_multiframe_with_options, MultiFrameInfo,
    MultiFrameSpatialMetadata, MultiFrameWriterConfig,
};
pub use object_model::{
    is_private_tag, DicomObjectModel, DicomObjectNode, DicomPreservationSet, DicomPreservedElement,
    DicomSequenceItem, DicomTag, DicomValue,
};
pub use reader::{
    load_dicom_series, load_dicom_series_with_metadata, read_dicom_series,
    read_dicom_series_with_metadata, scan_dicom_directory, DicomReadMetadata, DicomSeriesInfo,
    DicomSliceMetadata,
};
pub use rt_dose::{read_rt_dose, write_rt_dose, RtDoseGrid, RT_DOSE_SOP_CLASS_UID};
pub use rt_plan::{
    read_rt_plan, write_rt_plan, RtBeamInfo, RtFractionGroup, RtPlanInfo, RT_PLAN_SOP_CLASS_UID,
};
pub use rt_struct::{read_rt_struct, rt_roi_to_polydata, RtContour, RtRoiInfo, RtStructureSet};
pub use seg::{read_dicom_seg, write_dicom_seg, DicomSegmentInfo, DicomSegmentation};
pub use sop_class::{classify_sop_class, is_image_sop_class, SopClassKind};
pub use transfer_syntax::TransferSyntaxKind;
pub use writer::{write_dicom_series, write_dicom_series_with_metadata, DicomWriter};
pub use writer_object::{model_to_in_mem, write_object as write_dicom_object};
