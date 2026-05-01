//! Pixel frame backend abstraction.
//!
//! `FrameDecodeBackend` is generic so call sites monomorphize backend dispatch.
//! Concrete backends remain replaceable implementation details.

pub mod dicom_rs;
pub mod native;

use anyhow::Result;

use crate::pixel::PixelLayout;
use crate::syntax::TransferSyntaxKind;

#[derive(Debug, Clone)]
pub struct DecodeFrameRequest {
    pub frame_index: u32,
    pub transfer_syntax: TransferSyntaxKind,
    pub layout: PixelLayout,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    pub pixels: Vec<f32>,
}

pub trait FrameDecodeBackend<O> {
    fn decode_frame(object: &O, request: DecodeFrameRequest) -> Result<DecodedFrame>;
}

pub trait EncapsulatedFrameSource {
    fn encapsulated_frame(&self, frame_index: u32) -> Result<Vec<u8>>;
}

pub use dicom_rs::DicomRsBackend;
pub use native::NativeCodecBackend;
