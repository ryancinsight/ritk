use super::*;
use dicom::core::DataElement;

fn pixel_object(
    rows: u16,
    columns: u16,
    samples_per_pixel: u16,
    bits_allocated: u16,
    photometric: &str,
    planar_configuration: Option<u16>,
    frames: Option<&str>,
    pixel_data: Vec<u8>,
) -> InMemDicomObject {
    let mut object = InMemDicomObject::new_empty();
    for (tag, value) in [
        (ROWS, rows),
        (COLUMNS, columns),
        (SAMPLES_PER_PIXEL, samples_per_pixel),
        (BITS_ALLOCATED, bits_allocated),
        (BITS_STORED, bits_allocated),
        (HIGH_BIT, bits_allocated - 1),
        (PIXEL_REPRESENTATION, 0),
    ] {
        object.put(DataElement::new(tag, VR::US, PrimitiveValue::from(value)));
    }
    object.put(DataElement::new(
        PHOTOMETRIC_INTERPRETATION,
        VR::CS,
        PrimitiveValue::from(photometric),
    ));
    if let Some(planar_configuration) = planar_configuration {
        object.put(DataElement::new(
            PLANAR_CONFIGURATION,
            VR::US,
            PrimitiveValue::from(planar_configuration),
        ));
    }
    if let Some(frames) = frames {
        object.put(DataElement::new(
            NUMBER_OF_FRAMES,
            VR::IS,
            PrimitiveValue::from(frames),
        ));
    }
    object.put(DataElement::new(
        PIXEL_DATA,
        if bits_allocated > 8 { VR::OW } else { VR::OB },
        PrimitiveValue::U8(pixel_data.into()),
    ));
    object
}

fn error(object: &InMemDicomObject) -> DicomWriteError {
    preflight_native_pixel_data(object)
        .expect_err("invalid pixel object must be rejected")
        .downcast::<DicomWriteError>()
        .expect("typed DICOM writer failure")
}

#[test]
fn accepts_single_frame_without_number_of_frames_and_odd_unpadded_payload() {
    let object = pixel_object(1, 3, 1, 8, "MONOCHROME2", None, None, vec![1, 2, 3]);
    preflight_native_pixel_data(&object).expect("valid one-frame pixels");
}

#[test]
fn palette_color_uses_one_sample_per_pixel() {
    let object = pixel_object(1, 2, 1, 8, "PALETTE COLOR", None, None, vec![7, 9]);
    preflight_native_pixel_data(&object).expect("valid palette indices");
}

#[test]
fn rejects_missing_rows_attribute() {
    let mut object = InMemDicomObject::new_empty();
    object.put(DataElement::new(
        PIXEL_DATA,
        VR::OB,
        PrimitiveValue::U8(vec![1].into()),
    ));

    assert!(matches!(
        error(&object),
        DicomWriteError::MissingPixelAttribute { attribute: "Rows" }
    ));
}

#[test]
fn rejects_us_attribute_with_text_primitive() {
    let mut object = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
    object.put(DataElement::new(ROWS, VR::US, PrimitiveValue::from("1")));

    assert!(matches!(
        error(&object),
        DicomWriteError::MalformedPixelAttribute {
            attribute: "Rows",
            ..
        }
    ));
}

#[test]
fn rejects_multi_value_us_attribute() {
    let mut object = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
    object.put(DataElement::new(
        ROWS,
        VR::US,
        PrimitiveValue::U16(vec![1, 2].into()),
    ));

    assert!(matches!(
        error(&object),
        DicomWriteError::MalformedPixelAttribute {
            attribute: "Rows",
            ..
        }
    ));
}

#[test]
fn rejects_multi_value_frame_count() {
    let mut object = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
    object.put(DataElement::new(
        NUMBER_OF_FRAMES,
        VR::IS,
        PrimitiveValue::I32(vec![1, 2].into()),
    ));

    assert!(matches!(
        error(&object),
        DicomWriteError::MalformedPixelAttribute {
            attribute: "NumberOfFrames",
            ..
        }
    ));
}

#[test]
fn rejects_multi_value_photometric_interpretation() {
    let mut object = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
    object.put(DataElement::new(
        PHOTOMETRIC_INTERPRETATION,
        VR::CS,
        PrimitiveValue::Strs(vec!["MONOCHROME2".to_owned(), "RGB".to_owned()].into()),
    ));

    assert!(matches!(
        error(&object),
        DicomWriteError::MalformedPixelAttribute {
            attribute: "PhotometricInterpretation",
            ..
        }
    ));
}

#[test]
fn rejects_empty_or_invalid_photometric_code_string() {
    for photometric in ["", " ", "monochrome2", "MONOCHROME2\t", "0123456789ABCDEFG"] {
        let mut object = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
        object.put(DataElement::new(
            PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            PrimitiveValue::from(photometric),
        ));

        assert!(
            matches!(
                error(&object),
                DicomWriteError::MalformedPixelAttribute {
                    attribute: "PhotometricInterpretation",
                    ..
                }
            ),
            "accepted invalid PhotometricInterpretation {photometric:?}"
        );
    }
}

#[test]
fn rejects_invalid_number_of_frames_integer_string() {
    for frames in ["0000000000001", "\t1", "1 2"] {
        let object = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, Some(frames), vec![1]);

        assert!(
            matches!(
                error(&object),
                DicomWriteError::MalformedPixelAttribute {
                    attribute: "NumberOfFrames",
                    ..
                }
            ),
            "accepted invalid NumberOfFrames {frames:?}"
        );
    }
}

#[test]
fn padding_uses_little_endian_wire_order_for_word_primitives() {
    let valid = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
    let mut valid = valid;
    valid.put(DataElement::new(
        PIXEL_DATA,
        VR::OW,
        PrimitiveValue::U16(vec![1].into()),
    ));
    preflight_native_pixel_data(&valid).expect("zero wire-order padding byte");

    let mut invalid = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
    invalid.put(DataElement::new(
        PIXEL_DATA,
        VR::OW,
        PrimitiveValue::U16(vec![0x0100].into()),
    ));
    assert_eq!(
        error(&invalid),
        DicomWriteError::InvalidPixelDataPadding { value: 1 }
    );
}

#[test]
fn rejects_floating_samples_in_integer_pixel_data() {
    let mut object = pixel_object(1, 1, 1, 8, "MONOCHROME2", None, None, vec![1]);
    object.put(DataElement::new(
        PIXEL_DATA,
        VR::OB,
        PrimitiveValue::from(1.0_f32),
    ));
    assert!(matches!(
        error(&object),
        DicomWriteError::UnsupportedPixelPayloadValue { .. }
    ));
}

#[test]
fn accepts_ybr_full_422_with_even_columns_and_interleaved_chroma() {
    let object = pixel_object(
        1,
        2,
        3,
        8,
        "YBR_FULL_422",
        Some(0),
        None,
        vec![16, 235, 128, 128],
    );
    preflight_native_pixel_data(&object).expect("valid subsampled YBR pixels");
}

#[test]
fn accepts_retired_ybr_partial_422_with_legacy_native_layout() {
    let object = pixel_object(
        1,
        2,
        3,
        8,
        "YBR_PARTIAL_422",
        Some(0),
        None,
        vec![10, 11, 12, 13],
    );
    preflight_native_pixel_data(&object)
        .expect("retired 4:2:2 values retain their historical layout");
}

#[test]
fn native_multiframe_payload_has_no_padding_between_frames() {
    let object = pixel_object(
        1,
        3,
        1,
        8,
        "MONOCHROME2",
        None,
        Some("2"),
        vec![1, 2, 3, 4, 5, 6],
    );
    preflight_native_pixel_data(&object).expect("frames concatenate without per-frame padding");

    let padded_each_frame = pixel_object(
        1,
        3,
        1,
        8,
        "MONOCHROME2",
        None,
        Some("2"),
        vec![1, 2, 3, 0, 4, 5, 6, 0],
    );
    assert_eq!(
        error(&padded_each_frame),
        DicomWriteError::PixelPayloadLengthMismatch {
            expected: 6,
            actual: 8,
        }
    );
}

#[test]
fn accepts_packed_binary_pixels() {
    let object = pixel_object(1, 9, 1, 1, "MONOCHROME2", None, None, vec![0xA5, 0x01]);
    preflight_native_pixel_data(&object).expect("nine one-bit samples occupy two bytes");
}

#[test]
fn requires_even_columns_for_ybr_full_422() {
    let object = pixel_object(
        1,
        3,
        3,
        8,
        "YBR_FULL_422",
        Some(0),
        None,
        vec![10, 11, 12, 13],
    );
    assert!(matches!(
        error(&object),
        DicomWriteError::MalformedPixelAttribute {
            attribute: "PhotometricInterpretation",
            ..
        }
    ));
}

#[test]
fn accepts_zero_value_padding_and_rejects_nonzero_padding() {
    let valid = pixel_object(1, 3, 1, 8, "MONOCHROME2", None, None, vec![1, 2, 3, 0]);
    preflight_native_pixel_data(&valid).expect("zero value padding is valid");

    let invalid = pixel_object(1, 3, 1, 8, "MONOCHROME2", None, None, vec![1, 2, 3, 1]);
    assert_eq!(
        error(&invalid),
        DicomWriteError::InvalidPixelDataPadding { value: 1 }
    );
}

#[test]
fn rejects_mismatched_payload_length() {
    let object = pixel_object(2, 2, 1, 8, "MONOCHROME2", None, None, vec![1, 2, 3]);
    assert_eq!(
        error(&object),
        DicomWriteError::PixelPayloadLengthMismatch {
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn rejects_number_of_frames_products_that_overflow() {
    let object = pixel_object(
        u16::MAX,
        u16::MAX,
        u16::MAX,
        8,
        "VENDOR_DEFINED",
        Some(0),
        Some("2147483647"),
        Vec::new(),
    );
    assert_eq!(error(&object), DicomWriteError::PixelCountOverflow);
}
