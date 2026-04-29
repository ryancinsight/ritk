//! DICOM series discovery, hierarchical organisation, and volume loading.
//!
//! # Sub-modules
//! - [`series_tree`] — flat `SeriesEntry` records and the patient→study→series
//!   tree used by the series browser sidebar.
//! - [`loader`]      — volume loading from DICOM folders and NIfTI files;
//!   wraps `ritk-io` and produces [`crate::LoadedVolume`] values.

pub mod loader;
pub mod series_tree;

pub use loader::{
    load_dicom_volume, load_nifti_volume, load_volume_from_path, scan_folder_for_series,
};
pub use series_tree::{PatientNode, SeriesEntry, SeriesTree, StudyNode};
