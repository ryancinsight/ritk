//! Static JPEG decoder backend boundary.
//!
//! The current implementation uses the pure Rust `jpeg-decoder` crate. Keeping
//! it behind a sealed ZST backend gives `ritk-codecs::jpeg` one replacement
//! point for an in-crate decoder without changing DICOM frame dispatch.

use std::io::Cursor;

use anyhow::{Context, Result};
use jpeg_decoder::{Decoder, PixelFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JpegPixelFormat {
    L8,
    L16,
    Rgb24,
    Cmyk32,
}

impl JpegPixelFormat {
    pub(crate) const fn pixel_bytes(self) -> usize {
        match self {
            Self::L8 => 1,
            Self::L16 => 2,
            Self::Rgb24 => 3,
            Self::Cmyk32 => 4,
        }
    }
}

impl From<PixelFormat> for JpegPixelFormat {
    fn from(value: PixelFormat) -> Self {
        match value {
            PixelFormat::L8 => Self::L8,
            PixelFormat::L16 => Self::L16,
            PixelFormat::RGB24 => Self::Rgb24,
            PixelFormat::CMYK32 => Self::Cmyk32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JpegDecoded {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixel_format: JpegPixelFormat,
    pub(crate) pixels: Vec<u8>,
}

pub(crate) trait JpegDecodeBackend: private::Sealed {
    fn decode(fragment: &[u8]) -> Result<JpegDecoded>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct JpegDecoderCrate;

impl private::Sealed for JpegDecoderCrate {}

impl JpegDecodeBackend for JpegDecoderCrate {
    fn decode(fragment: &[u8]) -> Result<JpegDecoded> {
        let mut decoder = Decoder::new(Cursor::new(fragment));
        let pixels = decoder
            .decode()
            .context("native JPEG decoder failed to decode fragment")?;
        let info = decoder
            .info()
            .context("native JPEG decoder did not expose image metadata")?;

        Ok(JpegDecoded {
            width: usize::from(info.width),
            height: usize::from(info.height),
            pixel_format: info.pixel_format.into(),
            pixels,
        })
    }
}

mod private {
    pub trait Sealed {}
}
