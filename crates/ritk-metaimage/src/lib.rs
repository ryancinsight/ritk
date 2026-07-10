//! MetaImage (MHA/MHD) I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for reading and writing
//! MetaImage files (.mha / .mhd format). It separates MetaImage logic from the polymorphic I/O
//! dispatch layer in `ritk-io`.
//!
//! # Key APIs
//!
//! - [`read_metaimage`]: Read a MetaImage file as a native image with spatial metadata
//! - [`write_metaimage`]: Write an Image to a MetaImage file with full affine encoding
//!
//! # Spatial Convention
//!
//! - RITK tensors: `[Z, Y, X]` (ZYX ordering)
//! - MetaImage storage: `[X, Y, Z]` with X-fastest contiguous payload order
//! - All read/write functions shape flat payloads directly as `[Z, Y, X]`
//!
//! # File Formats
//!
//! - `.mha` — single file with header and inline binary data (`ElementDataFile = LOCAL`)
//! - `.mhd` / `.raw` — ASCII header referencing a separate binary raw file
//!
//! # Spatial Metadata
//!
//! TransformMatrix encodes the 3×3 direction matrix (row-major) in MetaImage
//! `[X,Y,Z]` file-axis order. The reader/writer convert spacing and direction
//! columns to and from RITK internal `[Z,Y,X]` image-axis order.

pub mod reader;
mod spatial;
pub mod writer;

pub use reader::{read_metaimage, MetaImageReader};
pub use writer::{write_metaimage, write_metaimage_with_data, MetaImageWriter};

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use std::path::Path;

/// DIP boundary executing strict spatial metadata preservation over standard MetaImage datasets.
pub struct MetaImageDipReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MetaImageDipReader<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<f32, B, 3>> {
        read_metaimage(path, &self.backend)
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard MetaImage datasets.
pub struct MetaImageDipWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MetaImageDipWriter<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> anyhow::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_metaimage(path, image, &self.backend)
    }
}

#[cfg(test)]
mod tests;
