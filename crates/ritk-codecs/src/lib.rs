//! RITK-native codec implementations.
//!
//! This crate is the single source of truth for all DICOM pixel codec
//! primitives: pixel layout arithmetic, native sample decoding, and all
//! encapsulated transfer syntax decoders.
//!
//! # C/C++ dependency migration status
//! | Codec       | C/C++ dep         | Pure Rust implementation             | Status |
//! |-------------|-------------------|--------------------------------------|--------|
//! | JPEG 2000   | (none)            | ISO 15444-1 codec in [`jpeg_2000`]   | done (lossless, 0 DWT levels; multi-level DWT: J2K-DECODE-DWT) |
//! | JPEG        | `jpeg-decoder`    | Pure Rust JPEG decoder (Rust crate)  | done   |
//! | JPEG-LS     | (none â€” RITK-native since Sprint 127) |                  | done   |
//! | PackBits    | (none â€” pure Rust) |                                     | done   |
//! | RLE         | (none â€” pure Rust) |                                     | done   |
//!
//! `openjpeg-sys` / `openjp2` / `jpeg2k` / `charls` are no longer workspace
//! dependencies â€” all DICOM codec paths (decode and the JPEG 2000 / JPEG-LS
//! encoders) are pure Rust with no C/C++ FFI.

pub mod byte_decode;
pub(crate) mod dimensions;
pub mod jpeg;
pub mod jpeg_2000;
pub mod jpeg_ls;
pub mod packbits;
pub mod pixel_layout;
pub mod rle;

pub use byte_decode::{
    decode_bytes_to_f32, parse_f64_vec, parse_floats, parse_usize_vec, require_bytes, ByteOrder,
};
pub use jpeg::decode_jpeg_fragment;
pub use jpeg_2000::decode_jpeg2000_fragment;
pub use jpeg_ls::decode_jpeg_ls_fragment;
pub use packbits::packbits_decode;
pub use pixel_layout::{decode_native_pixel_bytes_checked, PixelLayout, PixelSignedness};
pub use rle::decode_rle_lossless_fragment;
