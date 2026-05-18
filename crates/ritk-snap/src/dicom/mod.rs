//! DICOM series discovery, hierarchical organisation, and volume loading.
//!
//! # Sub-modules
//! - [`series_tree`] — flat `SeriesEntry` records and the patient→study→series
//!   tree used by the series browser sidebar.
//! - [`loader`]      — volume loading from DICOM folders and NIfTI files;
//!   wraps `ritk-io` and produces [`crate::LoadedVolume`] values.
//! - [`metadata_table`] — presentation-neutral DICOM tag inspector rows.
//! - [`input_path`] — DICOM folder and DICOMDIR file normalization.
//! - [`hanging_protocol`] — deterministic load-time protocol selection.

pub mod hanging_protocol;
pub mod input_path;
pub mod loader;
pub mod metadata_table;
pub mod pet;
pub mod series_tree;
pub mod suv;

pub use hanging_protocol::{select_hanging_protocol, HangingProtocolDecision};
pub use input_path::{classify_dicom_input_path, DicomInputPath};
pub use loader::{
    load_dicom_volume, load_nifti_volume, load_volume_from_bytes, load_volume_from_path,
    scan_folder_for_series,
};
pub use metadata_table::{build_metadata_rows, MetadataRow, MetadataScope};
pub use series_tree::{PatientNode, SeriesEntry, SeriesTree, StudyNode};
