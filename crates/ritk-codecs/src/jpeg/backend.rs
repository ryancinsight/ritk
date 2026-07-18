//! Static JPEG decoder backend boundary.
//!
//! The sealed `JpegDecodeBackend` trait is the single replacement point for the
//! JPEG decoder. `RitkJpegDecoder` in `ritk_decoder.rs` is the authoritative
//! implementation; this module owns only the contract types and the sealed trait.

use anyhow::Result;

/// Decoded pixel format from a JPEG fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JpegPixelFormat {
    L8,
    L16,
    Rgb24 }

impl JpegPixelFormat {
    pub(crate) const fn pixel_bytes(self) -> usize {
        match self {
            Self::L8 => 1,
            Self::L16 => 2,
            Self::Rgb24 => 3 }
    }

    pub(crate) const fn samples_per_pixel(self) -> usize {
        match self {
            Self::L8 | Self::L16 => 1,
            Self::Rgb24 => 3 }
    }

    pub(crate) const fn bits_allocated(self) -> u16 {
        match self {
            Self::L8 | Self::Rgb24 => 8,
            Self::L16 => 16 }
    }
}

/// Raw decoded JPEG output before DICOM modality-LUT application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JpegDecoded {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixel_format: JpegPixelFormat,
    /// Raster-order pixel bytes: 1 byte/pixel for L8, 2 for L16 (native endian),
    /// 3 interleaved for Rgb24.
    pub(crate) pixels: Vec<u8> }

/// Sealed static-dispatch JPEG decode backend.
pub(crate) trait JpegDecodeBackend: private::Sealed {
    fn decode(fragment: &[u8]) -> Result<JpegDecoded>;
}

pub(super) mod private {
    pub trait Sealed {}
}
