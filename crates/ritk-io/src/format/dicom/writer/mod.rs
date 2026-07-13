//! DICOM series writer using dicom-rs v0.8.2.
//! Transfer syntax: Explicit VR LE. Each .dcm has 128-byte preamble + DICM magic.
//!
//! Stage 1 scope:
//! - preserve metadata-driven tags during series write
//! - keep pixel-module ordering stable
//! - verify private tag propagation for supported scalar tags

mod metadata;
mod preservation;
mod series;
pub(crate) mod utils;

#[cfg(test)]
mod tests;

pub use metadata::{write_dicom_series_with_metadata, DicomWriter};
pub use series::{write_dicom_series, write_dicom_series_native};

#[cfg(test)]
pub(crate) use utils::generate_series_uid;
