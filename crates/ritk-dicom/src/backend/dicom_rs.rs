//! `dicom-rs` frame decode backend.
//!
//! This module is the only place where `dicom-pixeldata::PixelDecoder` is bound
//! to the RITK DICOM domain boundary.

use anyhow::{bail, Context, Result};
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use dicom_pixeldata::PixelDecoder;
use std::path::Path;

use crate::backend::{
    DecodeFrameRequest, DecodedFrame, DicomParseBackend, EncapsulatedFrameSource,
    NativeCodecBackend, PixelDecodeBackend,
};
use crate::pixel::{decode_native_pixel_bytes_checked, PixelLayout};
use crate::syntax::TransferSyntaxKind;

impl EncapsulatedFrameSource for DefaultDicomObject {
    fn encapsulated_frame(&self, frame_index: u32) -> Result<Vec<u8>> {
        let pixel_data = self
            .element(Tag(0x7FE0, 0x0010))
            .context("missing Pixel Data (7FE0,0010)")?;
        match pixel_data.value() {
            Value::PixelSequence(seq) => seq
                .fragments()
                .get(frame_index as usize)
                .map(|f| f.to_vec())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "frame {} out of range for {} pixel fragments",
                        frame_index,
                        seq.fragments().len()
                    )
                }),
            _ => bail!("Pixel Data is not encapsulated"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DicomRsBackend;

impl DicomParseBackend for DicomRsBackend {
    type Object = DefaultDicomObject;

    fn parse_file(path: &Path) -> Result<Self::Object> {
        dicom::object::open_file(path)
            .with_context(|| format!("dicom-rs backend failed to parse {:?}", path))
    }

    fn parse_bytes(data: &[u8]) -> Result<Self::Object> {
        use std::io::Cursor;
        dicom::object::from_reader(Cursor::new(data))
            .with_context(|| "dicom-rs backend failed to parse DICOM bytes")
    }
}

impl PixelDecodeBackend<DefaultDicomObject> for DicomRsBackend {
    fn decode_frame(
        object: &DefaultDicomObject,
        request: DecodeFrameRequest,
    ) -> Result<DecodedFrame> {
        let pixels = match &request.transfer_syntax {
            syntax if syntax.is_native_jpeg_codec() => {
                NativeCodecBackend::decode_frame(object, request.clone())?.pixels
            }
            TransferSyntaxKind::RleLossless => {
                NativeCodecBackend::decode_frame(object, request.clone())?.pixels
            }
            TransferSyntaxKind::JpegLsLossless | TransferSyntaxKind::JpegLsLossy => {
                NativeCodecBackend::decode_frame(object, request.clone())?.pixels
            }
            TransferSyntaxKind::Jpeg2000Lossless | TransferSyntaxKind::Jpeg2000Lossy => {
                NativeCodecBackend::decode_frame(object, request.clone())?.pixels
            }
            syntax if syntax.is_external_backend_codec_candidate() => {
                decode_via_dicom_rs(object, &request)?
            }
            syntax if syntax.is_big_endian() => {
                bail!(
                    "native Pixel Data decode requires little-endian transfer syntax; got {}",
                    syntax.uid()
                )
            }
            _ => {
                let bytes = object
                    .element(Tag(0x7FE0, 0x0010))
                    .context("missing Pixel Data (7FE0,0010)")?
                    .value()
                    .to_bytes()
                    .map_err(|e| anyhow::anyhow!("Pixel Data bytes unreadable: {:?}", e))?;
                let frame_bytes = request.layout.bytes_per_frame()?;
                let frame_index = usize::try_from(request.frame_index)
                    .context("native Pixel Data frame index does not fit usize")?;
                let start = frame_index
                    .checked_mul(frame_bytes)
                    .context("native Pixel Data frame offset overflow")?;
                let end = start
                    .checked_add(frame_bytes)
                    .context("native Pixel Data frame end overflow")?;
                let frame = bytes.get(start..end).ok_or_else(|| {
                    anyhow::anyhow!(
                        "frame {} out of range for native Pixel Data byte length {} and frame byte length {}",
                        request.frame_index,
                        bytes.len(),
                        frame_bytes
                    )
                })?;
                decode_native_pixel_bytes_checked(frame, request.layout)?
            }
        };
        Ok(DecodedFrame { pixels })
    }
}

fn decode_via_dicom_rs(
    object: &DefaultDicomObject,
    request: &DecodeFrameRequest,
) -> Result<Vec<f32>> {
    let decoded = object
        .decode_pixel_data_frame(request.frame_index)
        .with_context(|| {
            format!(
                "dicom-rs backend failed to decode frame {} with transfer syntax {}",
                request.frame_index,
                request.transfer_syntax.uid()
            )
        })?;
    // dicom-pixeldata applies the modality LUT (RescaleSlope × stored +
    // RescaleIntercept) internally per DICOM PS3.3 §C.7.6.3.1.4. Passing the
    // layout with rescale applied again would double-apply the linear
    // transformation: (sample × slope + intercept) × slope + intercept.
    // Use identity rescale (slope=1, intercept=0) so decode_native_pixel_bytes_checked
    // only performs the required integer-to-f32 conversion and byte-length validation.
    let identity_layout = PixelLayout {
        rows: request.layout.rows,
        cols: request.layout.cols,
        samples_per_pixel: request.layout.samples_per_pixel,
        bits_allocated: request.layout.bits_allocated,
        pixel_representation: request.layout.pixel_representation,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    };
    decode_native_pixel_bytes_checked(decoded.data(), identity_layout)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
#[path = "tests_dicom_rs.rs"]
mod tests;
