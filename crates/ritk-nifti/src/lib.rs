#![doc = include_str!("../README.md")]

mod document;
mod header;
mod reader;
mod shape;
mod spatial;
mod typed;
mod writer;

pub use document::{
    transcode_nifti_document, NiftiDocument, NiftiDocumentError, NiftiDocumentHeader, NiftiVersion,
    SpatialFormRelation,
};
pub use reader::{
    read_nifti, read_nifti_from_bytes, read_nifti_labels, read_nifti_series,
    read_nifti_series_from_bytes,
};
pub use typed::{NiftiReader, NiftiWriter};
pub use writer::{
    write_nifti, write_nifti2, write_nifti2_labels, write_nifti2_series, write_nifti_labels,
    write_nifti_series,
};

#[cfg(test)]
mod tests;
