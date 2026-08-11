//! RITK-native DICOM primitives and backend boundaries.
//!
//! This crate is the single source of truth for DICOM transfer-syntax
//! classification and pixel-codec contracts. `dicom-rs` remains available as a
//! backend while RITK-native codecs replace backend-specific paths incrementally.

pub mod attribute;
pub mod backend;
pub mod codec;
pub mod diffusion;
pub mod pixel;
pub mod syntax;

pub use attribute::{tags, DicomAttributeRead, DicomTag};
pub use backend::{
    decode_frame_with, parse_bytes_with, parse_file_with, write_bytes_with, write_file_with,
    DecodeFrameRequest, DecodedFrame, DicomBackend, DicomParseBackend, DicomRsBackend,
    DicomWriteBackend, EncapsulatedFrameSource, NativeCodecBackend, PixelDecodeBackend,
};
pub use codec::{decode_jpeg_fragment, decode_rle_lossless_fragment, packbits_decode};
pub use diffusion::{
    extract_diffusion_pair, read_dicom_gradient_scheme_from_file,
    read_dicom_gradient_scheme_from_files,
};
pub use pixel::{decode_native_pixel_bytes_checked, PixelLayout, PixelSignedness};
pub use syntax::TransferSyntaxKind;
