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
    decode_frame_with, parse_file_with, DecodeFrameRequest, DecodedFrame, DicomBackend,
    DicomParseBackend, DicomRsBackend, EncapsulatedFrameSource, NativeCodecBackend,
    PixelDecodeBackend,
};
pub use codec::{decode_jpeg_fragment, decode_rle_lossless_fragment, packbits_decode};
#[allow(deprecated)]
pub use pixel::{decode_native_pixel_bytes, decode_native_pixel_bytes_checked, PixelLayout};
pub use syntax::TransferSyntaxKind;
