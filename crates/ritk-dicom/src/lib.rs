//! RITK-native DICOM primitives and backend boundaries.
//!
//! This crate is the single source of truth for DICOM transfer-syntax
//! classification, mutable datasets, Part 10 I/O, and pixel-codec dispatch.

pub mod attribute;
pub mod backend;
pub mod codec;
pub mod dataset;
pub use dataset::wire::charset::CharacterSet;
pub mod diffusion;
pub mod pixel;
pub mod syntax;

pub use attribute::{is_private_tag, tags, DicomAttributeRead, DicomTag};
pub use backend::parse_budget::{
    parse_bytes_with_budget, parse_file_with_budget, read_file_with_budget,
    read_file_within_root_with_budget, scan_part10_attributes, validate_part10,
    BoundedDicomParseBackend, DicomParseSummary, Part10Attributes,
};
pub use backend::{
    decode_frame_with, parse_bytes_with, parse_file_with, write_bytes_with, write_file_with,
    DecodeFrameRequest, DicomBackend, DicomParseBackend, DicomWriteBackend,
    EncapsulatedFrameSource, NativeBackend, NativeCodecBackend, PixelDecodeBackend,
};
pub use codec::{decode_jpeg_fragment, decode_rle_lossless_fragment, packbits_decode};
pub use consus_core::ParseBudget;
pub use diffusion::{
    extract_diffusion_pair, read_dicom_gradient_scheme_from_file,
    read_dicom_gradient_scheme_from_files,
};
pub use pixel::{
    decode_native_pixel_bytes_checked, decode_stored_pixel_frame, PixelLayout, PixelSignedness,
    StoredPixelError,
};
pub use syntax::TransferSyntaxKind;
