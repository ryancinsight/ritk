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

mod dip;
mod error;
pub mod reader;
mod spatial;
pub mod writer;

pub use dip::{MetaImageDipReader, MetaImageDipWriter};
pub use error::MetaImageError;
pub use reader::{read_metaimage, MetaImageReader};
pub use writer::{write_metaimage, write_metaimage_with_data, MetaImageWriter};

#[cfg(test)]
mod tests;
