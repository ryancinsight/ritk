//! Re-export NIfTI functionality from ritk-nifti.
//!
//! This module provides backward-compatible access to NIfTI readers and writers
//! that are now implemented in the dedicated ritk-nifti crate.

pub use ritk_nifti::{
    read_nifti, read_nifti_from_bytes, read_nifti_labels, write_nifti, write_nifti_labels,
    NiftiReader, NiftiWriter,
};
