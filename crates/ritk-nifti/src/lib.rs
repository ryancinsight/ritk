//! NIfTI (Neuroimaging Informatics Technology Initiative) I/O for RITK.
//!
//! This crate provides native, canonical single-source-of-truth implementations
//! for reading NIfTI-1/NIfTI-2 and writing explicit NIfTI-1 or NIfTI-2
//! single-file payloads. It separates NIfTI
//! byte-level logic from the polymorphic I/O dispatch layer in `ritk-io` and
//! does not depend on `nifti-rs` or ndarray conversion surfaces.
//!
//! The current codec supports uncompressed `.nii` and gzip-wrapped `.nii.gz`
//! streams for the RITK image contracts used in this workspace: 3-D Float32
//! images, UInt32 label maps, sform/qform spatial metadata, checked shape
//! products, and bounded payload reads before allocation.
//!
//! Analyze 7.5 `.hdr`/`.img` pairs are owned by `ritk-analyze`. Paired NIfTI
//! headers (`ni1`/`ni2`) are a distinct NIfTI extension point and are not mixed
//! into this single-file codec.
//!
//! # Key APIs
//!
//! - [`read_nifti`]: Read a NIfTI file as a native image with spatial metadata
//! - [`write_nifti`]: Write an Image to a NIfTI file with full sform affine encoding
//! - [`write_nifti2`]: Write an Image to a NIfTI-2 file with full sform affine encoding
//! - [`read_nifti_labels`]: Read label maps (segmentations) as ZYX-ordered u32 vectors
//! - [`write_nifti_labels`]: Write label maps to NIfTI with spatial metadata
//! - [`write_nifti2_labels`]: Write label maps to NIfTI-2 with spatial metadata
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

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use std::path::Path;
pub use writer::{write_nifti, write_nifti2, write_nifti2_labels, write_nifti_labels};

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiReader<B: ComputeBackend> {
    backend: B }

impl<B: ComputeBackend> NiftiReader<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
        read_nifti(path, &self.backend).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiWriter<B: ComputeBackend> {
    backend: B }

impl<B: ComputeBackend> NiftiWriter<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_nifti(path, image, &self.backend).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests;
