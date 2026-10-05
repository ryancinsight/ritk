#![doc = include_str!("../README.md")]

mod header;
mod reader;
mod shape;
mod spatial;
mod stored;
mod typed;
mod writer;

pub use header::NiftiHeaderError;
pub use reader::{
    read_nifti, read_nifti_from_bytes, read_nifti_labels, read_nifti_series,
    read_nifti_series_from_bytes,
};
pub use stored::{read_nifti_stored, read_nifti_stored_from_bytes, NiftiStoredReadError};
pub use typed::{NiftiReader, NiftiWriter};
pub use writer::{
    write_nifti, write_nifti2, write_nifti2_labels, write_nifti2_series, write_nifti_labels,
    write_nifti_series, write_nifti_stored, NiftiStoredWriteError,
};

#[cfg(test)]
mod tests;
