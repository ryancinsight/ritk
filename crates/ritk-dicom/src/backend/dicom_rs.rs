//! `dicom-rs` frame decode backend.
//!
//! This module is the only place where `dicom-pixeldata::PixelDecoder` is bound
//! to the RITK DICOM domain boundary.

use anyhow::{bail, Context, Result};
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use dicom_pixeldata::PixelDecoder;

use crate::backend::{
    DecodeFrameRequest, DecodedFrame, EncapsulatedFrameSource, FrameDecodeBackend,
    NativeCodecBackend,
};
use crate::pixel::decode_native_pixel_bytes_checked;
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

impl FrameDecodeBackend<DefaultDicomObject> for DicomRsBackend {
    fn decode_frame(
        object: &DefaultDicomObject,
        request: DecodeFrameRequest,
    ) -> Result<DecodedFrame> {
        let pixels = match &request.transfer_syntax {
            syntax if syntax.is_native_jpeg_codec() => {
                decode_jpeg_with_backend_fallback(object, &request)?
            }
            TransferSyntaxKind::RleLossless => {
                NativeCodecBackend::decode_frame(object, request.clone())?.pixels
            }
            syntax if syntax.is_external_backend_codec_candidate() => {
                decode_via_dicom_rs(object, &request)?
            }
            _ => {
                let bytes = object
                    .element(Tag(0x7FE0, 0x0010))
                    .context("missing Pixel Data (7FE0,0010)")?
                    .value()
                    .to_bytes()
                    .map_err(|e| anyhow::anyhow!("Pixel Data bytes unreadable: {:?}", e))?;
                decode_native_pixel_bytes_checked(&bytes, request.layout)?
            }
        };
        Ok(DecodedFrame { pixels })
    }
}

fn decode_jpeg_with_backend_fallback(
    object: &DefaultDicomObject,
    request: &DecodeFrameRequest,
) -> Result<Vec<f32>> {
    match NativeCodecBackend::decode_frame(object, request.clone()) {
        Ok(decoded) => Ok(decoded.pixels),
        Err(native_error) => decode_via_dicom_rs(object, request).with_context(|| {
            format!("native JPEG decode failed ({native_error:#}); dicom-rs fallback failed")
        }),
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
    decode_native_pixel_bytes_checked(decoded.data(), request.layout)
}
