//! RITK-native DICOM primitives and backend boundaries.
//!
//! This crate is the single source of truth for DICOM transfer-syntax
//! classification and pixel-codec contracts. `dicom-rs` remains available as a
//! backend while RITK-native codecs replace backend-specific paths incrementally.

pub mod backend;
pub mod codec;
pub mod pixel;
pub mod syntax;

pub use backend::{
    DecodeFrameRequest, DecodedFrame, DicomRsBackend, EncapsulatedFrameSource, FrameDecodeBackend,
};
pub use codec::{decode_rle_lossless_fragment, packbits_decode};
pub use pixel::{decode_native_pixel_bytes, PixelLayout};
pub use syntax::TransferSyntaxKind;
