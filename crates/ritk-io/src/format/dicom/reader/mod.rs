//! DICOM series reader and metadata API.
//!
//! The reader is split by responsibility:
//! - `scan`: directory discovery, SOP filtering, and series geometry assembly.
//! - `parse`: per-file DICOM metadata extraction and preservation capture.
//! - `pixel`: per-slice scalar pixel decode through the `ritk-dicom` backend.
//! - `loader`: conversion from scanned series metadata to `Image<B, 3>`.
//!
//! # Invariants
//!
//! - The input path must resolve to a directory containing at least one DICOM file.
//! - All returned slices are image-bearing SOP classes after scan filtering.
//! - Scalar volume loading accepts `SamplesPerPixel == 1`; color paths use the
//!   dedicated color loaders.
//! - Pixel transfer syntax handling is centralized in `ritk-dicom`.

mod dicomdir;
pub(super) mod geometry;
pub(super) mod loader;
mod parse;
pub(super) mod pixel;
mod preservation;
pub(super) mod scan;
pub(crate) mod types;
pub(super) mod utils;

#[cfg(test)]
mod tests;

pub use loader::{
    load_dicom_from_series, load_dicom_series_with_metadata, read_dicom_series_with_metadata,
};
pub use scan::{scan_dicom_instances, scan_dicom_part10_bytes};
pub(super) use types::DicomSeriesInfo;
// scan::scan_dicom_directory is accessed directly via `reader::scan::scan_dicom_directory`
// by sibling modules (color.rs). No re-export needed.
pub use types::literal_arraystring;
pub use types::{
    DicomReadMetadata, DicomSeriesInfo as ScannedDicomSeries, DicomSliceMetadata, PatientPosition,
};

pub(super) use geometry::{
    analyze_slice_spacing, dot, normalize, resample_frames_linear, slice_normal_from_iop,
};
