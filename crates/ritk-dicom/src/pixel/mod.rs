//! Pixel layout and native sample decoding.
//!
//! Re-exports from `ritk-codecs` — the single source of truth for codec
//! domain primitives. All internal `ritk-dicom` modules continue to use
//! `crate::pixel::PixelLayout` without modification.
#[allow(deprecated)]
pub use ritk_codecs::pixel_layout::{
    decode_native_pixel_bytes, decode_native_pixel_bytes_checked, PixelLayout, PixelSignedness,
};
