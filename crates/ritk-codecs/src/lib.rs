//! RITK-native codec implementations.
//!
//! This crate is the single source of truth for all DICOM pixel codec
//! primitives: pixel layout arithmetic, native sample decoding, and all
//! encapsulated transfer syntax decoders.
//!
//! # C/C++ dependency migration plan
//! | Codec       | Current C/C++ dep  | Target pure Rust        | Phase |
//! |-------------|-------------------|-------------------------|-------|
//! | JPEG 2000   | `openjpeg-sys`    | ISO 15444-1 Rust impl   | 2     |
//! | JPEG        | `jpeg-decoder`    | Pure Rust JPEG decoder  | 3     |
//! | JPEG-LS     | (none — RITK-native since Sprint 127) | complete | done |
//! | PackBits    | (none — pure Rust) |                        | done  |
//! | RLE         | (none — pure Rust) |                        | done  |
//!
//! Phase 1 (this sprint): extract all codecs from `ritk-dicom` into this crate.
//! Phase 2: replace `openjpeg-sys` with a pure Rust JPEG 2000 decoder.
//! Phase 3: replace `jpeg-decoder` with a pure Rust JPEG decoder.
//! Phase 4: remove `charls` / `dicom-transfer-syntax-registry` charls+openjpeg
//!          features from the workspace once RITK-native codecs cover all
//!          needed transfer syntaxes.

pub mod jpeg;
pub mod jpeg_2000;
pub mod jpeg_ls;
pub mod packbits;
pub mod pixel_layout;
pub mod rle;

pub use jpeg::decode_jpeg_fragment;
pub use jpeg_2000::decode_jpeg2000_fragment;
pub use jpeg_ls::decode_jpeg_ls_fragment;
pub use packbits::packbits_decode;
#[allow(deprecated)]
pub use pixel_layout::{
    decode_native_pixel_bytes, decode_native_pixel_bytes_checked, PixelLayout,
};
pub use rle::decode_rle_lossless_fragment;
