//! NIfTI (Neuroimaging Informatics Technology Initiative) I/O for RITK.
//!
//! This crate provides native, canonical single-source-of-truth implementations
//! for reading and writing NIfTI-1 single-file payloads. It separates NIfTI
//! byte-level logic from the polymorphic I/O dispatch layer in `ritk-io` and
//! does not depend on `nifti-rs` or ndarray conversion surfaces.
//!
//! The current codec supports uncompressed `.nii` and gzip-wrapped `.nii.gz`
//! streams for the RITK image contracts used in this workspace: 3-D Float32
//! images, UInt32 label maps, sform/qform spatial metadata, checked shape
//! products, and bounded payload reads before allocation.
//!
//! # Key APIs
//!
//! - [`read_nifti`]: Read a NIfTI file as a Burn tensor-backed Image with spatial metadata
//! - [`write_nifti`]: Write an Image to a NIfTI file with full sform affine encoding
//! - [`read_nifti_labels`]: Read label maps (segmentations) as ZYX-ordered u32 vectors
//! - [`write_nifti_labels`]: Write label maps to NIfTI with spatial metadata
//!
//! # Spatial Convention
//!
//! - RITK tensors: `[Z, Y, X]` (depth, row, column)
//! - NIfTI storage: `[X, Y, Z]` (file axes)
//! - All read/write functions handle the permutation automatically
//! - RITK physical metadata is LPS; NIfTI affines are encoded/read as RAS
//!
//! # Affine Metadata
//!
//! RITK's internal affine maps `[depth,row,col]` to LPS:
//! ```ignore
//! A_lps = [D[:,0] * spacing[0], D[:,1] * spacing[1], D[:,2] * spacing[2], origin]
//! ```
//! NIfTI sform rows map file `[x,y,z]` to RAS, so columns are emitted as
//! `[internal_col, internal_row, internal_depth]` and the first two physical
//! rows are sign-flipped between LPS and RAS.

mod header;
mod reader;
mod shape;
mod spatial;
mod writer;

pub use reader::{read_nifti, read_nifti_from_bytes, read_nifti_labels};
pub use writer::{write_nifti, write_nifti_labels};

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NiftiReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_nifti(path, &self.device).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiWriter<B: Backend> {
    _marker: std::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> Default for NiftiWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> NiftiWriter<B> {
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_nifti(path, image).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests;
