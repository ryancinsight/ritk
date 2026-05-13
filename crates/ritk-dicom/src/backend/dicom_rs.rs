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

impl DicomParseBackend for DicomRsBackend {
    type Object = DefaultDicomObject;

    fn parse_file(path: &Path) -> Result<Self::Object> {
        dicom::object::open_file(path)
            .with_context(|| format!("dicom-rs backend failed to parse {:?}", path))
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
            TransferSyntaxKind::JpegLsLossless => {
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
    decode_native_pixel_bytes_checked(decoded.data(), request.layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{decode_frame_with, parse_file_with};
    use crate::pixel::PixelLayout;
    use dicom::core::smallvec::SmallVec;
    use dicom::core::value::PixelFragmentSequence;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

    #[test]
    fn dicom_rs_backend_parses_file_and_decodes_uncompressed_frame() {
        let dir = tempfile::tempdir().expect("tempdir must be created");
        let path = dir.path().join("slice.dcm");

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.1001"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(vec![10_u16, 20, 30, 40])),
        ));

        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.1001")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("file meta must be valid")
        .write_to_file(&path)
        .expect("DICOM file must be written");

        let parsed = parse_file_with::<DicomRsBackend, _>(&path).expect("parse must succeed");
        let decoded = decode_frame_with::<DicomRsBackend>(
            &parsed,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::ExplicitVrLittleEndian,
                layout: PixelLayout {
                    rows: 2,
                    cols: 2,
                    samples_per_pixel: 1,
                    bits_allocated: 16,
                    pixel_representation: 0,
                    rescale_slope: 2.0,
                    rescale_intercept: -10.0,
                },
            },
        )
        .expect("decode must succeed");

        assert_eq!(decoded.pixels, vec![10.0, 30.0, 50.0, 70.0]);
    }

    #[test]
    fn dicom_rs_backend_decodes_requested_native_multiframe_only() {
        let dir = tempfile::tempdir().expect("tempdir must be created");
        let path = dir.path().join("multiframe.dcm");

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.1002"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0008),
            VR::IS,
            PrimitiveValue::from("2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(vec![1_u16, 2, 100, 200])),
        ));

        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid("2.25.1002")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("file meta must be valid")
        .write_to_file(&path)
        .expect("DICOM file must be written");

        let parsed = parse_file_with::<DicomRsBackend, _>(&path).expect("parse must succeed");
        let decoded = decode_frame_with::<DicomRsBackend>(
            &parsed,
            DecodeFrameRequest {
                frame_index: 1,
                transfer_syntax: TransferSyntaxKind::ExplicitVrLittleEndian,
                layout: PixelLayout {
                    rows: 1,
                    cols: 2,
                    samples_per_pixel: 1,
                    bits_allocated: 16,
                    pixel_representation: 0,
                    rescale_slope: 1.0,
                    rescale_intercept: 0.0,
                },
            },
        )
        .expect("second native frame decode must succeed");

        assert_eq!(decoded.pixels, vec![100.0, 200.0]);
    }

    #[test]
    fn native_owned_jpeg_errors_do_not_fallback_to_dicom_rs() {
        let dir = tempfile::tempdir().expect("tempdir must be created");
        let path = dir.path().join("bad_native_jpeg.dcm");

        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![vec![0xFF, 0xD8, 0xFF]]);
        let pixel_sequence: PixelFragmentSequence<Vec<u8>> =
            PixelFragmentSequence::new_fragments(fragments);

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.1003"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(8_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OB,
            pixel_sequence,
        ));

        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid("2.25.1003")
                .transfer_syntax("1.2.840.10008.1.2.4.50"),
        )
        .expect("file meta must be valid")
        .write_to_file(&path)
        .expect("DICOM file must be written");

        let parsed = parse_file_with::<DicomRsBackend, _>(&path).expect("parse must succeed");
        let err = decode_frame_with::<DicomRsBackend>(
            &parsed,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::JpegBaseline,
                layout: PixelLayout {
                    rows: 1,
                    cols: 1,
                    samples_per_pixel: 1,
                    bits_allocated: 8,
                    pixel_representation: 0,
                    rescale_slope: 1.0,
                    rescale_intercept: 0.0,
                },
            },
        )
        .expect_err("malformed native-owned JPEG fragment must fail");

        let msg = format!("{err:#}");
        assert!(
            msg.contains("native JPEG decoder"),
            "error must come from the RITK-native JPEG decoder, got: {msg}"
        );
        assert!(
            !msg.contains("fallback"),
            "native-owned JPEG syntaxes must not fall back through dicom-rs, got: {msg}"
        );
    }
}
