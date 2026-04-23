//! DICOM image format support.
//!
//! This module provides the canonical DICOM series entry points for `ritk-io`.
//! The actual reader and writer implementations live in the sibling `reader`
//! and `writer` modules.
//!
//! # Public API
//!
//! - `scan_dicom_directory`
//! - `read_dicom_series`
//! - `load_dicom_series`
//! - `read_dicom_series_with_metadata`
//! - `load_dicom_series_with_metadata`
//! - `DicomSeriesInfo`
//! - `DicomReadMetadata`
//! - `DicomSliceMetadata`
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

pub mod multiframe;
pub mod object_model;
pub mod reader;
pub mod transfer_syntax;
pub mod writer;
pub mod sop_class;
pub mod writer_object;

pub use object_model::{
    is_private_tag, DicomObjectModel, DicomObjectNode, DicomPreservationSet, DicomPreservedElement,
    DicomSequenceItem, DicomTag, DicomValue,
};
pub use reader::{
    load_dicom_series, load_dicom_series_with_metadata, read_dicom_series,
    read_dicom_series_with_metadata, scan_dicom_directory, DicomReadMetadata, DicomSeriesInfo,
    DicomSliceMetadata,
};
pub use multiframe::{load_dicom_multiframe, read_multiframe_info, write_dicom_multiframe, MultiFrameInfo};
pub use sop_class::{classify_sop_class, is_image_sop_class, SopClassKind};
pub use transfer_syntax::TransferSyntaxKind;
pub use writer::{write_dicom_series, write_dicom_series_with_metadata, DicomWriter};
pub use writer_object::{model_to_in_mem, write_object as write_dicom_object};
