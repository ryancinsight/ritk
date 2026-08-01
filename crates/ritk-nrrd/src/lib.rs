//! NRRD (Nearly Raw Raster Data) I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for reading and writing
//! NRRD files. It separates NRRD logic from the polymorphic I/O dispatch layer in `ritk-io`.
//!
//! # Key APIs
//!
//! - [`read_nrrd`]: Read a NRRD file as a native image with spatial metadata
//! - [`write_nrrd`]: Write an Image to a NRRD file with full space directions and origin encoding
//! - [`read_nrrd_series`]: Read an acquisition series as one image per volume
//! - [`write_nrrd_series`]: Write an acquisition series with a leading acquisition axis
//!
//! # Acquisition Axis
//!
//! A 4-D NRRD carries three spatial axes plus one non-spatial axis — the
//! diffusion gradient index of a DWI file, a functional timepoint. NRRD does
//! not fix that axis's position the way NIfTI does. The NA-MIC convention
//! Slicer and DTIPrep emit places it first:
//!
//! ```text
//! dimension: 4
//! sizes: 33 128 128 60
//! kinds: list domain domain domain
//! space directions: none (1.7,0,0) (0,1.7,0) (0,0,2.2)
//! ```
//!
//! while other tools place it last. The two differ in stride, not meaning: a
//! leading axis varies fastest, so volumes interleave voxel-by-voxel; a
//! trailing axis varies slowest, so volumes are contiguous blocks. Both are
//! read. Writing always emits the leading form, which diffusion tooling
//! expects.
//!
//! The single-volume and series entry points are asymmetric on purpose. The
//! series reader accepts a rank-3 file as a one-volume series, because that is
//! what it is. [`read_nrrd`] rejects a 4-D file rather than returning volume 0,
//! because a series has no correct single-volume decoding and quietly dropping
//! the remaining volumes would report success over lost acquisition data.
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

mod axes;
pub mod reader;
mod spatial;
pub mod writer;

pub use reader::{
    read_nrrd, read_nrrd_gradient_scheme, read_nrrd_header_map, read_nrrd_series, NrrdReader,
};
pub use writer::{write_nrrd, write_nrrd_series, write_nrrd_with_data, NrrdWriter};

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
