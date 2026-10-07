//! DICOM image writers with pixel and metadata consistency checks.
//! Transfer syntax: Explicit VR LE. Each .dcm has 128-byte preamble + DICM magic.
//!
//! Stage 1 scope:
//! - preserve metadata-driven tags during series write
//! - keep pixel-module ordering stable
//! - verify private tag propagation for supported scalar tags

pub(crate) mod decimal_string;
pub(crate) mod elements;
mod error;
mod metadata;
pub(crate) mod output;
pub(crate) mod pixel_encoding;
pub(crate) mod pixel_preflight;
mod preservation;
mod series;

#[cfg(test)]
mod tests;

pub use error::DicomWriteError;
pub use metadata::{write_dicom_series_with_metadata, DicomWriter};
pub use series::{write_dicom_series, write_dicom_series_native};

#[cfg(test)]
pub(crate) use pixel_encoding::generate_series_uid;
