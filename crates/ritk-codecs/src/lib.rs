//! RITK-native codec implementations.
//!
//! This crate provides typed sample buffers and header-vector parsing shared
//! by volume readers, plus DICOM pixel-layout and transfer-syntax codecs.

pub(crate) mod dimensions;
mod header_text;
pub mod jpeg;
pub mod jpeg_2000;
pub mod jpeg_ls;
pub mod packbits;
pub mod pixel_layout;
pub mod rle;
pub mod sample;

pub use header_text::{parse_f64_vec, parse_floats, parse_usize_vec};
pub use jpeg::decode_jpeg_fragment;
pub use jpeg_2000::decode_jpeg2000_fragment;
pub use jpeg_ls::decode_jpeg_ls_fragment;
pub use packbits::packbits_decode;
pub use pixel_layout::{decode_native_pixel_bytes_checked, PixelLayout, PixelSignedness};
pub use rle::{decode_rle_lossless_fragment, encode_rle_lossless_fragment_u16_grayscale};
