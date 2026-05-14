use super::helpers::{make_image_with_spatial, make_test_metadata};
use super::super::write_dicom_series_with_metadata;
use dicom::core::Tag;
use dicom::object::open_file;

#[test]
fn test_metadata_writer_spatial_tags_first_slice() {
    let meta = make_test_metadata();
    let image = make_image_with_spatial(3, 4, 4, 50.0, meta.origin, meta.spacing);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("meta_series");
    write_dicom_series_with_metadata(&path, &image, Some(&meta))
        .expect("metadata write must succeed");

    let dcm_path = path.join("slice_0000.dcm");
    let obj = open_file(&dcm_path).expect("must open written DICOM");

    let ipp = obj.element(Tag(0x0020, 0x0032)).expect("IPP tag must exist");
    let ipp_str = ipp.to_str().unwrap();
    let ipp_vals: Vec<f64> = ipp_str
        .split('\\')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(ipp_vals.len(), 3);
    assert!((ipp_vals[0] - 10.0).abs() < 1e-3, "IPP x={}", ipp_vals[0]);
    assert!((ipp_vals[1] - 20.0).abs() < 1e-3, "IPP y={}", ipp_vals[1]);
    assert!((ipp_vals[2] - 30.0).abs() < 1e-3, "IPP z={}", ipp_vals[2]);

    let iop = obj.element(Tag(0x0020, 0x0037)).expect("IOP tag must exist");
    let iop_str = iop.to_str().unwrap();
    let iop_vals: Vec<f64> = iop_str
        .split('\\')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(iop_vals.len(), 6);
    assert!((iop_vals[0] - 1.0).abs() < 1e-6, "IOP[0]");
    assert!((iop_vals[4] - 1.0).abs() < 1e-6, "IOP[4]");

    let ps = obj
        .element(Tag(0x0028, 0x0030))
        .expect("PixelSpacing must exist");
    let ps_str = ps.to_str().unwrap();
    let ps_vals: Vec<f64> = ps_str
        .split('\\')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(ps_vals.len(), 2);
    assert!((ps_vals[0] - 0.5).abs() < 1e-6, "PS row={}", ps_vals[0]);
    assert!((ps_vals[1] - 0.5).abs() < 1e-6, "PS col={}", ps_vals[1]);

    let st = obj
        .element(Tag(0x0018, 0x0050))
        .expect("SliceThickness must exist");
    let st_val: f64 = st.to_str().unwrap().trim().parse().unwrap();
    assert!((st_val - 2.5).abs() < 1e-6, "SliceThickness={}", st_val);

    let mod_elem = obj
        .element(Tag(0x0008, 0x0060))
        .expect("Modality must exist");
    assert_eq!(mod_elem.to_str().unwrap().trim(), "CT");

    let pid = obj
        .element(Tag(0x0010, 0x0020))
        .expect("PatientID must exist");
    assert_eq!(pid.to_str().unwrap().trim(), "PAT001");

    let sd = obj
        .element(Tag(0x0008, 0x0021))
        .expect("SeriesDate must exist");
    assert_eq!(sd.to_str().unwrap().trim(), "20240102");

    let st = obj
        .element(Tag(0x0008, 0x0031))
        .expect("SeriesTime must exist");
    assert_eq!(st.to_str().unwrap().trim(), "123456");

    let for_uid = obj
        .element(Tag(0x0020, 0x0052))
        .expect("FrameOfReferenceUID must exist");
    assert_eq!(for_uid.to_str().unwrap().trim(), "1.2.3.4.5.6.200");

    let private = obj
        .element(Tag(0x0019, 0x10AA))
        .expect("private tag must exist");
    assert_eq!(private.to_str().unwrap().trim(), "PRIVATE_SERIES_VALUE");
}

#[test]
fn test_metadata_writer_multislice_ipp_increment() {
    let meta = make_test_metadata();
    let image = make_image_with_spatial(3, 4, 4, 75.0, meta.origin, meta.spacing);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("multi_series");
    write_dicom_series_with_metadata(&path, &image, Some(&meta))
        .expect("metadata write must succeed");

    let expected_z = [30.0, 32.5, 35.0];
    for (z_idx, &ez) in expected_z.iter().enumerate() {
        let dcm_path = path.join(format!("slice_{z_idx:04}.dcm"));
        let obj = open_file(&dcm_path).unwrap_or_else(|_| panic!("must open slice {z_idx}"));
        let ipp = obj.element(Tag(0x0020, 0x0032)).expect("IPP must exist");
        let ipp_str = ipp.to_str().unwrap();
        let ipp_vals: Vec<f64> = ipp_str
            .split('\\')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert!(
            (ipp_vals[2] - ez).abs() < 1e-3,
            "slice {z_idx}: expected z={ez}, got z={}",
            ipp_vals[2]
        );
    }
}

#[test]
fn test_metadata_writer_none_metadata_fallback() {
    use super::helpers::make_image;
    let image = make_image(2, 4, 4, 25.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("no_meta_series");
    write_dicom_series_with_metadata(&path, &image, None)
        .expect("write with None metadata must succeed");

    let dcm_path = path.join("slice_0000.dcm");
    let obj = open_file(&dcm_path).expect("must open written DICOM");

    assert!(
        obj.element(Tag(0x0020, 0x0032)).is_err(),
        "IPP should not exist when metadata is None"
    );
    assert!(
        obj.element(Tag(0x0020, 0x0037)).is_err(),
        "IOP should not exist when metadata is None"
    );

    let mod_elem = obj
        .element(Tag(0x0008, 0x0060))
        .expect("Modality must exist");
    assert_eq!(mod_elem.to_str().unwrap().trim(), "OT");

    assert!(
        obj.element(Tag(0x0019, 0x10AA)).is_err(),
        "private tag should not exist when metadata is None"
    );
}

#[test]
fn test_metadata_writer_rejects_zero_dimension() {
    use super::helpers::make_image;
    let meta = make_test_metadata();
    let image = make_image(0, 4, 4, 0.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("zero_series");
    let result = write_dicom_series_with_metadata(&path, &image, Some(&meta));
    assert!(result.is_err(), "zero depth must be rejected");
}

#[test]
fn test_metadata_writer_pixel_tags_precede_pixel_data_and_are_unique() {
    let meta = make_test_metadata();
    let image = make_image_with_spatial(1, 4, 4, 42.0, meta.origin, meta.spacing);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("pixel_tag_order_series");
    write_dicom_series_with_metadata(&path, &image, Some(&meta))
        .expect("metadata write must succeed");

    let dcm_path = path.join("slice_0000.dcm");
    let bytes = std::fs::read(&dcm_path).expect("slice file must exist");

    let bits_allocated = [0x28_u8, 0x00, 0x00, 0x01];
    let bits_stored = [0x28_u8, 0x00, 0x01, 0x01];
    let high_bit = [0x28_u8, 0x00, 0x02, 0x01];
    let pixel_data = [0xE0_u8, 0x7F, 0x10, 0x00];

    fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        haystack
            .windows(needle.len())
            .enumerate()
            .filter_map(|(idx, window)| (window == needle).then_some(idx))
            .collect()
    }

    let bits_allocated_pos = find_all(&bytes, &bits_allocated[..]);
    let bits_stored_pos = find_all(&bytes, &bits_stored[..]);
    let high_bit_pos = find_all(&bytes, &high_bit[..]);
    let pixel_data_pos = find_all(&bytes, &pixel_data[..]);

    assert_eq!(
        bits_allocated_pos.len(),
        1,
        "BitsAllocated tag must appear exactly once, got {:?}",
        bits_allocated_pos
    );
    assert_eq!(
        bits_stored_pos.len(),
        1,
        "BitsStored tag must appear exactly once, got {:?}",
        bits_stored_pos
    );
    assert_eq!(
        high_bit_pos.len(),
        1,
        "HighBit tag must appear exactly once, got {:?}",
        high_bit_pos
    );
    assert_eq!(
        pixel_data_pos.len(),
        1,
        "PixelData tag must appear exactly once, got {:?}",
        pixel_data_pos
    );

    let pixel_data_offset = pixel_data_pos[0];
    assert!(
        bits_allocated_pos[0] < pixel_data_offset,
        "BitsAllocated must precede PixelData: {:?} vs {}",
        bits_allocated_pos,
        pixel_data_offset
    );
    assert!(
        bits_stored_pos[0] < pixel_data_offset,
        "BitsStored must precede PixelData: {:?} vs {}",
        bits_stored_pos,
        pixel_data_offset
    );
    assert!(
        high_bit_pos[0] < pixel_data_offset,
        "HighBit must precede PixelData: {:?} vs {}",
        high_bit_pos,
        pixel_data_offset
    );
}

/// When `write_dicom_series_with_metadata` is called with `None` metadata,
/// the five Type 2 mandatory DICOM tags must be present in the output slice.
///
/// Invariants: PatientName (0010,0010), PatientID (0010,0020),
/// ReferringPhysicianName (0008,0090), StudyDate (0008,0020),
/// SeriesNumber (0020,0011) — all PS3.3 Type 2.
#[test]
fn test_metadata_writer_none_metadata_type2_tags() {
    use super::helpers::make_image;
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("meta_none_type2_series");
    let image = make_image(2, 4, 4, 1.0);
    write_dicom_series_with_metadata(&out_path, &image, None)
        .expect("write_dicom_series_with_metadata(None)");

    let first_slice = out_path.join("slice_0000.dcm");
    let obj = open_file(&first_slice).expect("open_file");

    assert!(
        obj.element(Tag(0x0010, 0x0010)).is_ok(),
        "PatientName (0010,0010) must be present for None metadata"
    );
    assert!(
        obj.element(Tag(0x0010, 0x0020)).is_ok(),
        "PatientID (0010,0020) must be present for None metadata"
    );
    assert!(
        obj.element(Tag(0x0008, 0x0090)).is_ok(),
        "ReferringPhysicianName (0008,0090) must be present for None metadata"
    );
    assert!(
        obj.element(Tag(0x0008, 0x0020)).is_ok(),
        "StudyDate (0008,0020) must be present for None metadata"
    );
    assert!(
        obj.element(Tag(0x0020, 0x0011)).is_ok(),
        "SeriesNumber (0020,0011) must be present for None metadata"
    );
}
