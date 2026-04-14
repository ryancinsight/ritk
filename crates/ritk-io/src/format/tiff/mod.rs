// NOTE: `tiff = "0.9"` must be added to `[dependencies]` in
// `crates/ritk-io/Cargo.toml` for this module to compile.

//! TIFF / BigTIFF reader and writer for 3-D medical images.
//!
//! # Format
//! TIFF (Tagged Image File Format) stores raster images as a sequence of
//! IFD (Image File Directory) pages.  This module interprets each page as
//! one Z-slice of a volumetric image, stacking them into an
//! `Image<B, 3>` with tensor shape `[nz, ny, nx]`.
//!
//! BigTIFF is the 64-bit offset extension and is handled transparently by
//! the underlying `tiff` crate decoder/encoder.
//!
//! # Spatial metadata
//! TIFF has no standardized physical-space metadata fields.  All images
//! returned by [`read_tiff`] use default spatial values:
//! - `origin  = [0, 0, 0]`
//! - `spacing = [1, 1, 1]`
//! - `direction = identity`
//!
//! Users must set these externally when physical coordinates are known.
//!
//! # Pixel types
//! Reading supports u8, u16, u32, u64, i8, i16, i32, i64, f32, and f64
//! sample formats — all converted to f32 in the tensor.
//! Writing emits 32-bit IEEE 754 float samples (Gray32Float).

mod reader;
mod writer;

pub use reader::{read_tiff, TiffReader};
pub use writer::{write_tiff, TiffWriter};
