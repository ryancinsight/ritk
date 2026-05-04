//! NIfTI (Neuroimaging Informatics Technology Initiative) I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for reading and writing
//! NIfTI-1 files (medical imaging format). It separates NIfTI logic from the polymorphic I/O
//! dispatch layer in `ritk-io`.
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
//! - RITK tensors: `[Z, Y, X]` (ZYX ordering)
//! - NIfTI storage: `[X, Y, Z]` (XYZ ordering as per NIfTI standard)
//! - All read/write functions handle the permutation automatically
//!
//! # Affine Metadata
//!
//! Sform affine is encoded with the NIfTI convention:
//! ```ignore
//! srow_x = [M_col0[0], M_col1[0], M_col2[0], origin[0]]
//! srow_y = [M_col0[1], M_col1[1], M_col2[1], origin[1]]
//! srow_z = [M_col0[2], M_col1[2], M_col2[2], origin[2]]
//! ```
//! where `M_colJ = direction.column(J) * spacing[J]`.

mod reader;
mod writer;

pub use reader::{read_nifti, read_nifti_labels};
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
        read_nifti(path, &self.device)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiWriter<B: Backend> {
    _marker: std::marker::PhantomData<B>,
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
        write_nifti(path, image)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests;

