use super::*;
use dicom::core::value::Value;

// -------------------------------------------------------------------------
// Sprint 58-C: per-frame functional group tests
// -------------------------------------------------------------------------

/// PerFrameInfo::default() must have all fields set to None.
///
/// Invariant: PerFrameInfo::default() == PerFrameInfo { all_fields: None }.
#[test]
fn test_per_frame_info_default_all_none() {
    let pfi = PerFrameInfo::default();
    assert!(
        pfi.image_position.is_none(),
        "image_position must be None by default"
    );
    assert!(
        pfi.image_orientation.is_none(),
        "image_orientation must be None by default"
    );
    assert!(
        pfi.pixel_spacing.is_none(),
        "pixel_spacing must be None by default"
    );
    assert!(
        pfi.slice_thickness.is_none(),
        "slice_thickness must be None by default"
    );
    assert!(
        pfi.rescale_slope.is_none(),
        "rescale_slope must be None by default"
    );
    assert!(
        pfi.rescale_intercept.is_none(),
        "rescale_intercept must be None by default"
    );
}

/// extract_functional_groups must return an empty Vec when neither (5200,9229)
/// nor (5200,9230) is present in the DICOM object.
///
/// Invariant: non-enhanced multiframe objects carry no functional group sequences.
#[test]
fn test_per_frame_empty_when_no_functional_groups() {
    let obj = InMemDicomObject::new_empty();
    let result = extract_functional_groups(&obj, 3);
    assert!(
        result.is_empty(),
        "per_frame must be empty when no functional group sequences are present; got len={}",
        result.len()
    );
}

/// read_multiframe_info on a basic (non-enhanced) multiframe file written by
/// write_dicom_multiframe must return per_frame.is_empty() == true, because no
/// functional group sequences are emitted by the writer.
///
/// Invariant: basic multiframe SOP class has no functional groups.
#[test]
fn test_multiframe_info_per_frame_field_empty_for_basic_sop() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("basic_mf.dcm");
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 2 * 2 * 2], Shape::new([2_usize, 2, 2])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");
    let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
    assert!(
        info.per_frame.is_empty(),
        "per_frame must be empty for basic multiframe SOP; got len={}",
        info.per_frame.len()
    );
    assert_eq!(info.per_frame.len(), 0, "per_frame.len() must be 0");
}

/// extract_functional_groups with a Shared Functional Groups sequence (5200,9229)
/// must populate pixel_spacing and slice_thickness in all n_frames entries.
///
/// Analytical specification:
/// - (5200,9229)[0] → (0028,9110)[0] → (0028,0030) = "0.5\0.5" → pixel_spacing = [0.5, 0.5]
/// - (5200,9229)[0] → (0028,9110)[0] → (0018,0050) = "1.0"     → slice_thickness = 1.0
/// - Both values must appear in all n_frames entries (shared template cloned per frame).
#[test]
fn test_per_frame_shared_functional_groups() {
    use dicom::core::header::Length;
    use dicom::core::value::DataSetSequence;

    // Build PixelMeasuresSequence item with PixelSpacing and SliceThickness.
    let mut px_item = InMemDicomObject::new_empty();
    px_item.put(DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("0.5\\0.5"),
    ));
    px_item.put(DataElement::new(
        Tag(0x0018, 0x0050),
        VR::DS,
        PrimitiveValue::from("1.0"),
    ));
    let px_seq = DataSetSequence::new(vec![px_item], Length::UNDEFINED);
    let px_val = Value::from(px_seq);

    // Build shared functional group item containing PixelMeasuresSequence (0028,9110).
    let mut shared_item = InMemDicomObject::new_empty();
    shared_item.put(DataElement::new(Tag(0x0028, 0x9110), VR::SQ, px_val));

    // Build (5200,9229) Shared Functional Groups Sequence.
    let shared_seq = DataSetSequence::new(vec![shared_item], Length::UNDEFINED);
    let shared_val = Value::from(shared_seq);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(Tag(0x5200, 0x9229), VR::SQ, shared_val));

    let result = extract_functional_groups(&obj, 2);
    assert_eq!(result.len(), 2, "must produce one entry per frame");

    // Both frames share the same template from (5200,9229).
    for (k, pfi) in result.iter().enumerate() {
        assert_eq!(
            pfi.pixel_spacing,
            Some([0.5, 0.5]),
            "frame {k}: pixel_spacing must be [0.5, 0.5] from shared groups"
        );
        assert_eq!(
            pfi.slice_thickness,
            Some(1.0),
            "frame {k}: slice_thickness must be 1.0 from shared groups"
        );
    }
}

/// load_dicom_multiframe on an Enhanced CT–like object with per-frame functional
/// groups (5200,9230) must apply per-frame rescale slope/intercept independently
/// for each frame, overriding the global tags.
///
/// # Analytical specification
///
/// Two frames, 2×2 pixels each, native 16-bit unsigned.
/// Frame 0: raw u16 = [100, 200, 300, 400], slope=1.0, intercept=0.0
///          → decoded f32 = [100.0, 200.0, 300.0, 400.0]
/// Frame 1: raw u16 = [10, 20, 30, 40], slope=2.0, intercept=10.0
///          → decoded f32 = [30.0, 50.0, 70.0, 90.0]
///
/// Global slope/intercept set to 99.0 to verify per-frame values override them.
#[test]
fn test_load_dicom_multiframe_enhanced_per_frame_rescale() {
    use dicom::core::header::Length;
    use dicom::core::value::DataSetSequence;

    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("enhanced_pf_rescale.dcm");

    // Pixel data: frame0=[100,200,300,400] u16 LE, frame1=[10,20,30,40] u16 LE
    let frame0: [u16; 4] = [100, 200, 300, 400];
    let frame1: [u16; 4] = [10, 20, 30, 40];
    let pixel_bytes: Vec<u8> = frame0
        .iter()
        .chain(frame1.iter())
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    // Helper: build a PixelValueTransformation item with given slope/intercept DS strings.
    let make_pvt_item = |slope: &str, intercept: &str| {
        let mut item = InMemDicomObject::new_empty();
        item.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from(slope),
        ));
        item.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from(intercept),
        ));
        item
    };

    // Build per-frame item wrapping a PixelValueTransformationSequence (0028,9145).
    let make_pf_item = |slope: &str, intercept: &str| {
        let pvt_item = make_pvt_item(slope, intercept);
        let pvt_seq = DataSetSequence::new(vec![pvt_item], Length::UNDEFINED);
        let pvt_val = Value::from(pvt_seq);
        let mut pf = InMemDicomObject::new_empty();
        pf.put(DataElement::new(Tag(0x0028, 0x9145), VR::SQ, pvt_val));
        pf
    };

    let pf0 = make_pf_item("1.0", "0.0");
    let pf1 = make_pf_item("2.0", "10.0");

    // Build (5200,9230) Per-Frame Functional Groups Sequence with 2 items.
    let pf_seq = DataSetSequence::new(vec![pf0, pf1], Length::UNDEFINED);
    let pf_val = Value::from(pf_seq);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(MF_GRAYSCALE_WORD_SC_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.58001"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("2"),
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
        PrimitiveValue::from(0_u16),
    ));
    // Global slope/intercept = 99.0: must NOT be used when per-frame groups present.
    obj.put(DataElement::new(
        Tag(0x0028, 0x1053),
        VR::DS,
        PrimitiveValue::from("99.0"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1052),
        VR::DS,
        PrimitiveValue::from("99.0"),
    ));
    // Per-frame functional groups.
    obj.put(DataElement::new(Tag(0x5200, 0x9230), VR::SQ, pf_val));
    // Pixel data.
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(pixel_bytes)),
    ));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(MF_GRAYSCALE_WORD_SC_UID)
                .media_storage_sop_instance_uid("2.25.58001")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build");
    file_obj.write_to_file(&path).expect("write enhanced file");

    // Load and verify per-frame rescale via load_dicom_multiframe.
    let img = load_dicom_multiframe::<B, _>(&path, &device).expect("load enhanced per-frame");
    let [lf, lr, lc] = img.shape();
    assert_eq!(lf, 2, "frames");
    assert_eq!(lr, 2, "rows");
    assert_eq!(lc, 2, "cols");

    img.with_data_slice(|floats: &[f32]| {
        assert_eq!(floats.len(), 8, "total pixel count = 2 frames × 4 pixels");
        // Frame 0: raw=[100,200,300,400], slope=1.0, intercept=0.0
        // Decoded: [100.0, 200.0, 300.0, 400.0]
        assert!(
            (floats[0] - 100.0).abs() < 0.5,
            "frame0 pixel0: expected 100.0, got {}",
            floats[0]
        );
        assert!(
            (floats[3] - 400.0).abs() < 0.5,
            "frame0 pixel3: expected 400.0, got {}",
            floats[3]
        );
        // Frame 1: raw=[10,20,30,40], slope=2.0, intercept=10.0
        // Decoded: [30.0, 50.0, 70.0, 90.0]
        assert!(
            (floats[4] - 30.0).abs() < 0.5,
            "frame1 pixel0: expected 30.0 (10*2+10), got {}",
            floats[4]
        );
        assert!(
            (floats[7] - 90.0).abs() < 0.5,
            "frame1 pixel3: expected 90.0 (40*2+10), got {}",
            floats[7]
        );
    });

    // Verify per_frame metadata via read_multiframe_info.
    let info = read_multiframe_info(&path).expect("read_multiframe_info");
    assert_eq!(
        info.per_frame.len(),
        2,
        "per_frame.len() must equal n_frames"
    );
    assert_eq!(
        info.per_frame[0].rescale_slope,
        Some(1.0),
        "per_frame[0].rescale_slope must be Some(1.0)"
    );
    assert_eq!(
        info.per_frame[0].rescale_intercept,
        Some(0.0),
        "per_frame[0].rescale_intercept must be Some(0.0)"
    );
    assert_eq!(
        info.per_frame[1].rescale_slope,
        Some(2.0),
        "per_frame[1].rescale_slope must be Some(2.0)"
    );
    assert_eq!(
        info.per_frame[1].rescale_intercept,
        Some(10.0),
        "per_frame[1].rescale_intercept must be Some(10.0)"
    );
}
