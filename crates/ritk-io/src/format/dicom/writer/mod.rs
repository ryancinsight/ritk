
//! DICOM series writer using dicom-rs v0.8.2.
//! Transfer syntax: Explicit VR LE. Each .dcm has 128-byte preamble + DICM magic.

pub(crate) mod elements;
mod error;
mod metadata;
pub(crate) mod output;
pub(crate) mod pixel_encoding;
mod preservation;
mod series;

#[cfg(test)]
mod tests;

pub use error::DicomWriteError;
pub use metadata::{write_dicom_series_with_metadata, DicomWriter};
pub use series::{write_dicom_series, write_dicom_series_native};

#[cfg(test)]
pub(crate) use pixel_encoding::generate_series_uid;