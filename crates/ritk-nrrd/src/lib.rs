//! NRRD (Nearly Raw Raster Data) I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for reading and writing
//! NRRD files. It separates NRRD logic from the polymorphic I/O dispatch layer in `ritk-io`.
//!
//! # Key APIs
//!
//! - [`read_nrrd`]: Read a NRRD file as a Burn tensor-backed Image with spatial metadata
//! - [`write_nrrd`]: Write an Image to a NRRD file with full space directions and origin encoding
//!
//! # Spatial Convention
//!
//! - RITK tensors: `[Z, Y, X]` (ZYX ordering)
//! - NRRD storage: `[X, Y, Z]` (XYZ ordering via ITK convention)
//! - All read/write functions handle the permutation automatically
//!
//! # Spatial Metadata
//!
//! Space directions encode per-axis transformations:
//! ```ignore
//! space_direction[i] = direction.column(i) * spacing[i]
//! spacing[i] = |space_direction[i]|
//! direction[:, i] = space_direction[i] / |space_direction[i]|
//! ```
//!
//! Space origin encodes the physical starting point in [X, Y, Z] space.

pub mod reader;
pub mod writer;

pub use reader::{read_nrrd, NrrdReader};
pub use writer::{write_nrrd, NrrdWriter};

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// DIP boundary executing strict spatial metadata preservation over standard NRRD datasets.
pub struct NrrdDipReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NrrdDipReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
    
    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<B, 3>> {
        read_nrrd(path, &self.device)
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NRRD datasets.
pub struct NrrdDipWriter<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for NrrdDipWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> NrrdDipWriter<B> {
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> anyhow::Result<()> {
        write_nrrd(path, image)
    }
}

#[cfg(test)]
mod tests;
