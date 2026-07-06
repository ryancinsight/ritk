//! MetaImage (MHA/MHD) I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for reading and writing
//! MetaImage files (.mha / .mhd format). It separates MetaImage logic from the polymorphic I/O
//! dispatch layer in `ritk-io`.
//!
//! # Key APIs
//!
//! - [`read_metaimage`]: Read a MetaImage file as a Burn tensor-backed Image with spatial metadata
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

/// Atlas-native-substrate I/O (plain end-state names, disambiguated from the
/// Burn functions by module path only; folds away when the Burn path is
/// deleted — ADR 0002 A1).
pub mod native {
    pub use crate::reader::native::*;
    pub use crate::writer::native::*;
}
pub use reader::{read_metaimage, MetaImageReader};
pub use writer::{write_metaimage, write_metaimage_with_data, MetaImageWriter};

use ritk_image::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// DIP boundary executing strict spatial metadata preservation over standard MetaImage datasets.
pub struct MetaImageDipReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MetaImageDipReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<B, 3>> {
        read_metaimage(path, &self.device)
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard MetaImage datasets.
pub struct MetaImageDipWriter<B: Backend> {
    _marker: std::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> Default for MetaImageDipWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> MetaImageDipWriter<B> {
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> anyhow::Result<()>
    where
        B: ritk_image::HostExtract,
    {
        write_metaimage(path, image)
    }
}

#[cfg(test)]
mod tests;
