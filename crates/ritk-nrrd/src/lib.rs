//! NRRD (Nearly Raw Raster Data) I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for reading and writing
//! NRRD files. It separates NRRD logic from the polymorphic I/O dispatch layer in `ritk-io`.
//!
//! # Key APIs
//!
//! - [`read_nrrd`]: Read a NRRD file as a native image with spatial metadata
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

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::path::Path;

/// DIP boundary executing strict spatial metadata preservation over standard NRRD datasets.
pub struct NrrdDipReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> NrrdDipReader<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<f32, B, 3>> {
        read_nrrd(path, &self.backend)
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NRRD datasets.
pub struct NrrdDipWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> NrrdDipWriter<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> anyhow::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_nrrd(path, image, &self.backend)
    }
}

#[cfg(test)]
mod tests;
