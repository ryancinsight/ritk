//! MetaImage (MHA/MHD) I/O for RITK.
//!
//! This crate reads and writes MetaImage files (`.mha` and `.mhd`). The format
//! implementation is available directly and through `ritk-io` dispatch.
//!
//! # Key APIs
//!
//! - [`read_metaimage`]: Read a MetaImage file as an image of the requested sample type `T`
//!   with spatial metadata
//! - [`write_metaimage`]: Write an image of `T` to a MetaImage file whose `ElementType` names
//!   `T`, with full affine encoding
//!
//! # Sample types
//!
//! A MetaImage stores its voxels as one fixed-width numeric type named by `ElementType`:
//!
//! | `ElementType` | Stored sample |
//! |---|---|
//! | `MET_CHAR` | `i8` |
//! | `MET_UCHAR` | `u8` |
//! | `MET_SHORT` | `i16` |
//! | `MET_USHORT` | `u16` |
//! | `MET_INT` | `i32` |
//! | `MET_UINT` | `u32` |
//! | `MET_LONG` (read only) | `i32` |
//! | `MET_ULONG` (read only) | `u32` |
//! | `MET_LONG_LONG` | `i64` |
//! | `MET_ULONG_LONG` | `u64` |
//! | `MET_FLOAT` | `f32` |
//! | `MET_DOUBLE` | `f64` |
//!
//! The reader decodes the stored type, in the byte order `BinaryDataByteOrderMSB` names and
//! inflating a `CompressedData = True` payload, then converts to the requested `T` under a
//! [`ritk_codecs::sample::Conversion`] (ADR 0053): `Exact` refuses a read
//! that could change a value, `Cast` converts and warns. The writer stores
//! `T` itself.
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

mod element_type;
pub mod reader;
mod spatial;
pub mod writer;

pub use reader::{read_metaimage, MetaImageReader};
pub use writer::{write_metaimage, write_metaimage_with_data, MetaImageWriter};

#[cfg(test)]
mod tests;
