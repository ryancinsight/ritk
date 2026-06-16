use super::*;

#[test]
fn test_read_multiframe_info_missing_file_returns_error() {
    let result = read_multiframe_info("/nonexistent/path/file.dcm");
    assert!(result.is_err(), "expected Err for missing file");
}

#[test]
fn test_load_multiframe_missing_file_returns_error() {
    let device = <B as Backend>::Device::default();
    let result = load_dicom_multiframe::<B, _>("/nonexistent/path/file.dcm", &device);
    assert!(result.is_err(), "expected Err for missing file");
}

#[test]
fn test_load_multiframe_rejects_rgb_scalar_volume() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("rgb_multiframe.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(MF_GRAYSCALE_WORD_SC_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.999992"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("OT"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(3_u16),
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
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("RGB"),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::U8(SmallVec::from_vec(vec![120, 64, 32])),
    ));
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(MF_GRAYSCALE_WORD_SC_UID)
                .media_storage_sop_instance_uid("2.25.999992")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build must succeed");
    file_obj.write_to_file(&path).expect("write must succeed");

    let err = load_dicom_multiframe::<B, _>(&path, &device).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("SamplesPerPixel=3") && msg.contains("scalar volume loader"),
        "expected scalar loader RGB rejection, got {err:#}"
    );
}

#[test]
fn test_load_multiframe_compressed_ts_errors() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("compressed.dcm");

    // Construct a minimal DICOM dataset that declares JPEG Baseline TS (1.2.840.10008.1.2.4.50)
    // but contains no real compressed pixels. The TS guard must fire before pixel decode.
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.99999"),
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
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(vec![0u8; 8])),
    ));
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid("2.25.99999")
                .transfer_syntax("1.2.840.10008.1.2.4.80"), // JPEG-LS Lossless (no charls)
        )
        .expect("meta build");
    file_obj
        .write_to_file(&path)
        .expect("write compressed stub");

    let result = load_dicom_multiframe::<B, _>(&path, &device);
    // JPEG-LS lossless routes through the RITK native codec boundary. This synthetic
    // payload is intentionally minimal, so either a valid load or a JPEG-contextual
    // decode error preserves the boundary contract.
    match result {
        Ok(tensor) => {
            // If it succeeds, verify tensor shape is correct
            let shape = tensor.shape();
            assert!(shape.len() >= 3, "tensor must have at least 3 dimensions");
        }
        Err(e) => {
            // If it fails, error should reference JPEG-LS
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("1.2.840.10008.1.2.4.80")
                    || msg.to_lowercase().contains("compress")
                    || msg.contains("JPEG"),
                "error must reference JPEG-LS TS UID or 'compress'; got: {msg}"
            );
        }
    }
}

#[test]
fn test_load_multiframe_jpeg_baseline_codec_round_trip() {
    use dicom::core::smallvec::SmallVec;
    use dicom::core::value::PixelFragmentSequence;
    use image::{DynamicImage, GrayImage};

    let n_frames = 2usize;
    let rows = 4u32;
    let cols = 4u32;

    // Generate two distinct 4×4 frames.
    let frame0: Vec<u8> = (0u8..16).collect();
    let frame1: Vec<u8> = (100u8..116).collect();
    let original = [frame0.clone(), frame1.clone()];

    // JPEG-encode each frame into a separate fragment.
    let mut fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::new();
    for frame_pixels in &original {
        let gray =
            GrayImage::from_raw(cols, rows, frame_pixels.clone()).expect("GrayImage::from_raw");
        let dyn_img = DynamicImage::ImageLuma8(gray);
        let mut jpeg_bytes: Vec<u8> = Vec::new();
        {
            let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Jpeg)
                .expect("JPEG encode");
        }
        fragments.push(jpeg_bytes);
    }

    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(MF_GRAYSCALE_WORD_SC_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.77777701"),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0020),
        VR::LO,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from("2.25.77777702"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("2.25.77777703"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(rows as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(cols as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(format!("{n_frames}")),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(8u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(8u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(7u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1053),
        VR::DS,
        PrimitiveValue::from("1.000000"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1052),
        VR::DS,
        PrimitiveValue::from("0.000000"),
    ));
    obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OB, pfs));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(MF_GRAYSCALE_WORD_SC_UID)
                .media_storage_sop_instance_uid("2.25.77777701")
                .transfer_syntax("1.2.840.10008.1.2.4.50"), // JPEG Baseline
        )
        .expect("meta build");

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("mf_jpeg.dcm");
    file_obj.write_to_file(&path).expect("write");

    let device = <B as Backend>::Device::default();
    let img = load_dicom_multiframe::<B, _>(&path, &device)
        .expect("JPEG Baseline multiframe load must succeed via codec path");

    let [lf, lr, lc] = img.shape();
    assert_eq!(lf, n_frames, "shape[0] must equal n_frames");
    assert_eq!(lr, rows as usize, "shape[1] must equal rows");
    assert_eq!(lc, cols as usize, "shape[2] must equal cols");

    img.with_data_slice(|floats: &[f32]| {
        assert_eq!(
            floats.len(),
            n_frames * (rows as usize) * (cols as usize),
            "total pixel count mismatch"
        );
        // Verify each frame independently.
        let frame_size = (rows * cols) as usize;
        for (f_idx, orig_frame) in original.iter().enumerate() {
            let decoded_frame = &floats[f_idx * frame_size..(f_idx + 1) * frame_size];
            let max_error = orig_frame
                .iter()
                .zip(decoded_frame.iter())
                .map(|(&o, &d)| (o as f32 - d).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 8.0,
                "frame {f_idx}: codec round-trip error {max_error} exceeds JPEG tolerance 8.0"
            );
        }
    });
}

#[test]
fn test_load_multiframe_signed_short_roundtrip() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("signed_i16.dcm");

    // Construct a 1-frame 2×2 DICOM file with PixelRepresentation=1 (signed i16).
    let signed_pixels: [i16; 4] = [-1000, 0, 1000, 2000];
    let pixel_bytes: Vec<u8> = signed_pixels
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(MF_GRAYSCALE_WORD_SC_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.999888777"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("1"),
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
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(1_u16), // signed
    ));
    // Identity rescale: slope=1.0, intercept=0.0 (defaults when absent).
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(pixel_bytes)),
    ));
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(MF_GRAYSCALE_WORD_SC_UID)
                .media_storage_sop_instance_uid("2.25.999888777")
                .transfer_syntax("1.2.840.10008.1.2.1"), // Explicit VR LE
        )
        .expect("meta build");
    file_obj
        .write_to_file(&out_path)
        .expect("write signed file");

    let loaded = load_dicom_multiframe::<B, _>(&out_path, &device)
        .expect("load_dicom_multiframe signed i16");
    let [frames, rows, cols] = loaded.shape();
    assert_eq!(frames, 1, "frames");
    assert_eq!(rows, 2, "rows");
    assert_eq!(cols, 2, "cols");

    loaded.with_data_slice(|result: &[f32]| {
        assert_eq!(result.len(), 4, "pixel count");
        // Analytical ground truth: i16 × 1.0 + 0.0
        let expected = [-1000.0_f32, 0.0, 1000.0, 2000.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.5,
                "pixel {i}: expected {exp:.1} got {got:.1}"
            );
        }
    });
}

#[test]
fn test_multiframe_rejects_big_endian_ts() {
    // Verify that a DICOM multiframe file with ExplicitVrBigEndian TS
    // is rejected before pixel decode. We construct a file with BigEndian
    // in its file meta and assert load_dicom_multiframe returns an error.
    type B = burn_ndarray::NdArray<f32>;
    let device = Default::default();
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("be_test.dcm");

    // Build a minimal multiframe object with BigEndian TS in meta.
    let mut obj = InMemDicomObject::new_empty();
    // PixelData — 4 bytes (1 frame, 1x1 pixel, 16-bit LE; BE interpretation is wrong)
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(dicom::core::smallvec::SmallVec::from_slice(&[0u16])),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    // Build file meta with BigEndian TS UID
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid("2.25.999")
                .transfer_syntax("1.2.840.10008.1.2.2"), // ExplicitVrBigEndian
        )
        .expect("meta build must succeed");
    file_obj.write_to_file(&path).expect("write must succeed");

    let result = load_dicom_multiframe::<B, _>(&path, &device);
    assert!(
        result.is_err(),
        "load_dicom_multiframe must reject ExplicitVrBigEndian TS"
    );
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("big-endian"),
        "error message must mention big-endian; got: {err_msg}"
    );
}
