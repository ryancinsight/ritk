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

pub mod reader;
pub mod writer;

pub use reader::{
    load_dicom_series, load_dicom_series_with_metadata, read_dicom_series,
    read_dicom_series_with_metadata, scan_dicom_directory, DicomReadMetadata, DicomSeriesInfo,
    DicomSliceMetadata,
};
pub use writer::{write_dicom_series, write_dicom_series_with_metadata, DicomWriter};
