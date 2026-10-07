//! DICOM series reader and metadata API.
//!
//! The reader is split by responsibility:
//! - `scan`: directory discovery, SOP filtering, and series geometry assembly.
//! - `parse`: per-file DICOM metadata extraction and preservation capture.
//! - `pixel`: per-slice scalar pixel decode through the `ritk-dicom` backend.
//! - `loader`: conversion from scanned series metadata to `Image<f32, B, 3>`.
//!
//! # Invariants
//!
//! - The input path must resolve to a directory containing at least one DICOM file.
//! - All returned slices are image-bearing SOP classes after scan filtering.
//! - Scalar volume loading accepts `SamplesPerPixel == 1`; color paths use the
//!   dedicated color loaders.
//! - Pixel transfer syntax handling is centralized in `ritk-dicom`.

mod budget;
pub(super) mod detection;
pub(super) mod dicomdir;
mod dicomdir_bytes;
pub(super) mod geometry;
pub(super) mod loader;
mod parse;
pub(super) mod pixel;
mod preservation;
pub(super) mod scan;
mod stored;
pub(crate) mod types;

#[cfg(test)]
mod tests;

pub use loader::{
    load_dicom_from_series, load_dicom_from_series_with_budget, load_dicom_series_with_metadata,
    load_dicom_series_with_metadata_with_budget, read_dicom_series_with_metadata,
    read_dicom_series_with_metadata_with_budget,
};
pub use scan::{
    scan_dicom_directory_with_budget, scan_dicom_files, scan_dicom_files_with_budget,
    scan_dicom_instances, scan_dicom_instances_with_budget, scan_dicom_part10_bytes,
    scan_dicom_part10_bytes_with_budget, scan_dicom_path, scan_dicom_path_with_budget,
};
pub use stored::{
    load_dicom_stored_series, load_dicom_stored_series_with_budget, read_dicom_stored_series,
    read_dicom_stored_series_with_budget, StoredDicomError,
};
// scan::scan_dicom_directory is accessed directly via `reader::scan::scan_dicom_directory`
// by sibling modules (color.rs). No re-export needed.
pub use budget::DicomReadBudget;
pub use types::literal_arraystring;
pub use types::{
    DicomReadMetadata, DicomSeriesInfo as ScannedDicomSeries, DicomSliceMetadata, PatientPosition,
};

pub(super) use geometry::{
    analyze_slice_spacing, dot, normalize, resample_frames_linear, resampled_frame_count,
    slice_normal_from_iop,
};
