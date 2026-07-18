//! DICOM backend abstraction.
//!
//! Parsing and frame decode are generic so call sites monomorphize backend
//! dispatch. Concrete libraries remain replaceable implementation details.

pub mod dicom_rs;
pub mod native;

use anyhow::Result;
use std::path::Path;

use crate::pixel::PixelLayout;
use crate::syntax::TransferSyntaxKind;

#[derive(Debug, Clone)]
pub struct DecodeFrameRequest {
    pub frame_index: u32,
    pub transfer_syntax: TransferSyntaxKind,
    pub layout: PixelLayout }

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    pub pixels: Vec<f32> }

pub trait DicomParseBackend {
    type Object;
    fn parse_file(path: &Path) -> Result<Self::Object>;
    fn parse_bytes(data: &[u8]) -> Result<Self::Object>;
}

pub trait PixelDecodeBackend<O> {
    fn decode_frame(object: &O, request: DecodeFrameRequest) -> Result<DecodedFrame>;
}

pub trait DicomBackend:
    DicomParseBackend + PixelDecodeBackend<<Self as DicomParseBackend>::Object>
{
}

impl<T> DicomBackend for T where
    T: DicomParseBackend + PixelDecodeBackend<<T as DicomParseBackend>::Object>
{
}

pub fn parse_file_with<B, P>(path: P) -> Result<<B as DicomParseBackend>::Object>
where
    B: DicomParseBackend,
    P: AsRef<Path>,
{
    B::parse_file(path.as_ref())
}

pub fn parse_bytes_with<B>(data: &[u8]) -> Result<<B as DicomParseBackend>::Object>
where
    B: DicomParseBackend,
{
    B::parse_bytes(data)
}

pub fn decode_frame_with<B>(
    object: &<B as DicomParseBackend>::Object,
    request: DecodeFrameRequest,
) -> Result<DecodedFrame>
where
    B: DicomBackend,
{
    <B as PixelDecodeBackend<<B as DicomParseBackend>::Object>>::decode_frame(object, request)
}

pub trait EncapsulatedFrameSource {
    fn encapsulated_frame(&self, frame_index: u32) -> Result<Vec<u8>>;
}

pub use dicom_rs::DicomRsBackend;
pub use native::NativeCodecBackend;
