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
//! - RITK tensors: `[depth, row, col] = [Z, Y, X]`
//! - NRRD storage: `[X, Y, Z]` with X as the fastest-varying raw axis
//! - Raw payload bytes are already in the same flat order as RITK `[Z,Y,X]`
//!   tensors, so read/write use explicit shape conversion without tensor
//!   permutation.
//!
//! # Spatial Metadata
//!
//! NRRD `space directions` list file-axis vectors `[x,y,z]`. RITK image
//! metadata stores columns `[depth,row,col]`, so the authoritative mapping is:
//! ```ignore
//! internal[:, depth] = nrrd[:, z]
//! internal[:, row]   = nrrd[:, y]
//! internal[:, col]   = nrrd[:, x]
//! ```
//!
//! Space origin encodes the physical starting point in [X, Y, Z] space.

pub mod reader;
mod spatial;
pub mod writer;

pub use reader::{read_nrrd, NrrdReader};
pub use writer::{write_nrrd, write_nrrd_with_data, NrrdWriter};

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
    _marker: std::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> Default for NrrdDipWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> NrrdDipWriter<B> {
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> anyhow::Result<()>
    where
        B: ritk_image::HostExtract,
    {
        write_nrrd(path, image)
    }
}

#[cfg(test)]
mod tests;
