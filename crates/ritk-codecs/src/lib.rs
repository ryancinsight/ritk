//! RITK-native codec implementations.
//!
//! This crate is the single source of truth for all DICOM pixel codec
//! primitives: pixel layout arithmetic, native sample decoding, and all
//! encapsulated transfer syntax decoders.
//!
//! # C/C++ dependency migration plan
//! | Codec       | Current C/C++ dep  | Target pure Rust        | Phase |
//! |-------------|-------------------|-------------------------|-------|
//! | JPEG 2000   | none              | `jpeg2k` + `openjp2`    | done  |
//! | JPEG        | none (`jpeg-decoder` is pure Rust) | RITK-owned decoder behind `jpeg::backend` | constrained |
//! | JPEG-LS     | (none — RITK-native since Sprint 127) | complete | done |
//! | PackBits    | (none — pure Rust) |                        | done  |
//! | RLE         | (none — pure Rust) |                        | done  |
//!
//! Phase 1 (this sprint): extract all codecs from `ritk-dicom` into this crate.
//! Phase 2: replace `openjpeg-sys` with the Rust `openjp2` backend.
//! Phase 3: replace `JpegDecoderCrate` with a RITK-owned JPEG decoder
//!          implementation behind the sealed `jpeg::backend` boundary.
//! Phase 4: remove `charls` / `dicom-transfer-syntax-registry` external
//!          codec features once RITK-native codecs cover all needed transfer
//!          syntaxes.

pub mod jpeg;
#[cfg(not(target_arch = "wasm32"))]
pub mod jpeg_2000;
#[cfg(target_arch = "wasm32")]
pub mod jpeg_2000 {
    use anyhow::{bail, Result};

    use crate::PixelLayout;

    /// WebAssembly builds do not link OpenJPEG C libraries.
    pub fn decode_jpeg2000_fragment(_fragment: &[u8], _layout: PixelLayout) -> Result<Vec<f32>> {
        bail!("JPEG 2000 decoding is unavailable on wasm32 targets")
    }
}
pub mod jpeg_ls;
pub mod packbits;
pub mod pixel_layout;
pub mod rle;

pub use jpeg::decode_jpeg_fragment;
pub use jpeg_2000::decode_jpeg2000_fragment;
pub use jpeg_ls::decode_jpeg_ls_fragment;
pub use packbits::packbits_decode;
#[allow(deprecated)]
pub use pixel_layout::{decode_native_pixel_bytes, decode_native_pixel_bytes_checked, PixelLayout};
pub use rle::decode_rle_lossless_fragment;
