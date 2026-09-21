//! Native JPEG-LS (ISO 14495-1) codec for DICOM encapsulated frames (lossless and near-lossless).
//!
//! # Architecture
//! - `bitstream`: bit-level reader and writer with JPEG-LS 0xFF stuffing.
//! - `context`: ISO 14495-1 context model and threshold computation.
//! - `scan`: ISO 14495-1 regular-mode and run-mode scan decoder.
//! - [`encoder`]: ISO 14495-1 encoder, lossless and near-lossless.
//! - `decoder`: header-derived decoder state and scan-to-byte conversion.
//! - `image`: DICOM layout validation and modality-domain conversion.
//! - `parser`: marker parsing for SOI, SOF55, SOS, LSE, DRI, DNL, and EOI.

mod bitstream;
mod context;
mod decoder;
pub mod encoder;
mod image;
mod marker;
mod parser;
mod reconstruction;
mod sample_limits;
mod scan;

pub(crate) use decoder::{ComponentInfo, InterleaveMode, JpegLsDecoder};
pub use image::decode_jpeg_ls_fragment;
pub(crate) use marker::{DNL, DRI, EOI, LSE, SOF55, SOI, SOS};

#[cfg(test)]
mod tests;
