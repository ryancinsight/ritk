use super::*;
use crate::format::dicom::writer::DicomWriteError;
use ritk_dicom::{EncapsulatedFrameSource, TransferSyntaxKind};

fn unsigned_attribute(object: &dicom::object::DefaultDicomObject, tag: Tag) -> u16 {
    object
        .element(tag)
        .expect("attribute exists")
        .to_int()
        .expect("US value")
}

fn codestream_precision(fragment: &[u8], marker: [u8; 2]) -> u16 {
    // Skip complete header segments, never search entropy bytes for a marker.
    assert_eq!(&fragment[..2], &[0xff, 0xd8]);
    let mut offset = 2;
    loop {
        let segment = &fragment[offset..];
        if segment.starts_with(&marker) {
            return u16::from(segment[4]);
        }
        assert_eq!(segment[0], 0xff);
        let length = usize::from(u16::from_be_bytes([segment[2], segment[3]]));
        offset += 2 + length;
    }
}

#[test]
fn multiframe_pixel_tags_match_each_payload_representation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let image = native_image(
        [vec![0.0; 6], vec![255.0; 6]].concat(),
        [2, 2, 3],
        [0.0; 3],
        [1.0; 3],
    );
    for syntax in [
        TransferSyntaxKind::ExplicitVrLittleEndian,
        TransferSyntaxKind::JpegBaseline,
        TransferSyntaxKind::JpegLsLossless,
        TransferSyntaxKind::JpegLsLossy,
        TransferSyntaxKind::Jpeg2000Lossless,
        TransferSyntaxKind::Jpeg2000Lossy,
        TransferSyntaxKind::JpegLosslessFirstOrderPrediction,
        TransferSyntaxKind::JpegLosslessNonHierarchical,
        TransferSyntaxKind::RleLossless,
    ] {
        let path = temp.path().join("pixels.dcm");
        let config = MultiFrameWriterConfig {
            transfer_syntax: syntax.clone(),
            ..MultiFrameWriterConfig::default()
        };
        write_dicom_multiframe_native_with_config(&path, &image, &config).expect("write");
        let object = dicom::object::open_file(&path).expect("parse emitted tags");
        let bits = unsigned_attribute(&object, Tag(0x0028, 0x0100));
        let expected_bits = if syntax == TransferSyntaxKind::JpegBaseline {
            8
        } else {
            16
        };
        assert_eq!(bits, expected_bits, "{syntax:?}");
        assert_eq!(unsigned_attribute(&object, Tag(0x0028, 0x0101)), bits);
        assert_eq!(unsigned_attribute(&object, Tag(0x0028, 0x0102)), bits - 1);
        assert_eq!(unsigned_attribute(&object, Tag(0x0028, 0x0103)), 0);
        assert_eq!(unsigned_attribute(&object, Tag(0x0028, 0x0002)), 1);

        if syntax == TransferSyntaxKind::ExplicitVrLittleEndian {
            let bytes = object
                .element(Tag(0x7fe0, 0x0010))
                .expect("PixelData")
                .value()
                .to_bytes()
                .expect("native bytes");
            assert_eq!(bytes.len(), 12 * usize::from(bits / 8));
            assert_eq!(bytes.as_ref(), [vec![0; 12], vec![255; 12]].concat());
        } else {
            for frame in 0..2 {
                let fragment = object.encapsulated_frame(frame).expect("frame fragment");
                let payload_bits = match syntax {
                    TransferSyntaxKind::JpegBaseline => {
                        codestream_precision(&fragment, [0xff, 0xc0])
                    }
                    TransferSyntaxKind::JpegLsLossless | TransferSyntaxKind::JpegLsLossy => {
                        codestream_precision(&fragment, [0xff, 0xf7])
                    }
                    TransferSyntaxKind::JpegLosslessFirstOrderPrediction
                    | TransferSyntaxKind::JpegLosslessNonHierarchical => {
                        codestream_precision(&fragment, [0xff, 0xc3])
                    }
                    TransferSyntaxKind::Jpeg2000Lossless | TransferSyntaxKind::Jpeg2000Lossy => {
                        assert_eq!(&fragment[..4], &[0xff, 0x4f, 0xff, 0x51]);
                        // SIZ: first component's Ssiz follows Rsiz, geometry, and Csiz.
                        u16::from(fragment[42] & 0x7f) + 1
                    }
                    TransferSyntaxKind::RleLossless => {
                        let planes =
                            u32::from_le_bytes(fragment[..4].try_into().expect("RLE header"));
                        u16::try_from(planes * 8).expect("scalar byte planes")
                    }
                    _ => panic!("invariant: matrix contains only supported encapsulated syntaxes"),
                };
                assert_eq!(payload_bits, bits, "{syntax:?} frame {frame}");
            }
        }
    }
}

#[test]
fn multiframe_preflight_preserves_existing_output_for_invalid_inputs() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("pixels.dcm");
    let original = b"prior output";
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        std::fs::write(&path, original).expect("prior file");
        let image = native_image(vec![0.0, value], [2, 1, 1], [0.0; 3], [1.0; 3]);
        let error = write_dicom_multiframe_native(&path, &image).expect_err("nonfinite sample");
        assert_eq!(
            error.downcast_ref::<DicomWriteError>(),
            Some(&DicomWriteError::NonFinitePixel { index: 1 })
        );
        assert_eq!(std::fs::read(&path).expect("read prior output"), original);
    }
}
