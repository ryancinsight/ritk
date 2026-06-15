use super::super::{dicom_seg_to_label_map, read_dicom_seg};
use std::path::PathBuf;

#[test]
fn test_read_external_dcmqi_liver_seg_real_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("dcmqi")
        .join("liver.dcm");

    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );

    let seg = read_dicom_seg(&path).expect("read external dcmqi liver SEG");
    assert_eq!(seg.rows, 512);
    assert_eq!(seg.cols, 512);
    assert_eq!(seg.n_frames, 3);
    assert_eq!(seg.bits_allocated, 1);
    assert_eq!(seg.segmentation_type.as_dicom_str(), "BINARY");
    assert_eq!(seg.segments.len(), 1);
    assert_eq!(seg.segments[0].segment_number, 1);
    assert_eq!(seg.segments[0].segment_label, "Liver");
    assert_eq!(
        seg.segments[0]
            .algorithm_type
            .as_ref()
            .map(|t| t.as_dicom_str()),
        Some("SEMIAUTOMATIC")
    );
    assert_eq!(seg.frame_segment_numbers, vec![1, 1, 1]);

    let pixel_spacing = seg.pixel_spacing.expect("pixel spacing from shared FG");
    assert!((pixel_spacing[0] - 0.810547).abs() < 1e-6);
    assert!((pixel_spacing[1] - 0.810547).abs() < 1e-6);
    let slice_thickness = seg.slice_thickness.expect("slice thickness from shared FG");
    assert!((slice_thickness - 1.0).abs() < 1e-9);

    let positions: Vec<[f64; 3]> = seg
        .image_position_per_frame
        .iter()
        .map(|p| p.expect("all frame positions must be present"))
        .collect();
    assert_eq!(positions.len(), 3);
    assert!((positions[0][2] + 128.69).abs() < 1e-6);
    assert!((positions[1][2] + 127.69).abs() < 1e-6);
    assert!((positions[2][2] + 126.69).abs() < 1e-6);

    let rebuilt = dicom_seg_to_label_map(&seg).expect("rebuild label map from external SEG");
    assert_eq!(rebuilt.shape.0, [3, 512, 512]);
    assert!(rebuilt
        .present_labels()
        .contains(&ritk_annotation::LabelId(1)));
    assert!(rebuilt.as_slice().contains(&1));
}

#[test]
fn test_read_external_dcmqi_partial_overlaps_seg_real_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("dcmqi")
        .join("partial_overlaps.dcm");

    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );

    let seg = read_dicom_seg(&path).expect("read external dcmqi partial-overlap SEG");
    assert_eq!(seg.rows, 512);
    assert_eq!(seg.cols, 512);
    assert_eq!(seg.n_frames, 7);
    assert_eq!(seg.bits_allocated, 1);
    assert_eq!(seg.segmentation_type.as_dicom_str(), "BINARY");
    assert_eq!(seg.segments.len(), 5);
    assert_eq!(seg.frame_segment_numbers.len(), 7);
    assert!(seg
        .segments
        .iter()
        .all(|s| s.algorithm_type.as_ref().map(|t| t.as_dicom_str()) == Some("MANUAL")));

    let rebuilt = dicom_seg_to_label_map(&seg)
        .expect("rebuild label map from external dcmqi partial-overlap SEG");
    assert_eq!(rebuilt.shape.0, [3, 512, 512]);
    let present = rebuilt.present_labels();
    for label in [1u32, 2, 3, 4, 5] {
        assert!(
            present.contains(&ritk_annotation::LabelId(label)),
            "label {label} must be present"
        );
        assert!(
            rebuilt.count_label(label) > 0,
            "label {label} voxels must survive reconstruction"
        );
    }
}

#[test]
fn test_read_external_highdicom_overlap_seg_real_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("highdicom")
        .join("seg_image_ct_binary_overlap.dcm");

    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );

    let seg = read_dicom_seg(&path).expect("read external highdicom overlap SEG");
    assert_eq!(seg.rows, 16);
    assert_eq!(seg.cols, 16);
    assert_eq!(seg.n_frames, 8);
    assert_eq!(seg.bits_allocated, 1);
    assert_eq!(seg.segmentation_type.as_dicom_str(), "BINARY");
    assert_eq!(seg.segments.len(), 2);
    assert_eq!(seg.segments[0].segment_label, "first segment");
    assert_eq!(seg.segments[1].segment_label, "second segment");
    assert_eq!(
        seg.segments[0]
            .algorithm_type
            .as_ref()
            .map(|t| t.as_dicom_str()),
        Some("AUTOMATIC")
    );
    assert_eq!(
        seg.segments[1]
            .algorithm_type
            .as_ref()
            .map(|t| t.as_dicom_str()),
        Some("AUTOMATIC")
    );
    assert_eq!(seg.frame_segment_numbers, vec![1, 1, 1, 1, 2, 2, 2, 2]);

    let pixel_spacing = seg.pixel_spacing.expect("pixel spacing from shared FG");
    assert!((pixel_spacing[0] - 0.488281).abs() < 1e-6);
    assert!((pixel_spacing[1] - 0.488281).abs() < 1e-6);
    let slice_thickness = seg.slice_thickness.expect("slice thickness from shared FG");
    assert!((slice_thickness - 1.25).abs() < 1e-9);

    let rebuilt =
        dicom_seg_to_label_map(&seg).expect("rebuild label map from external highdicom SEG");
    assert_eq!(rebuilt.shape.0, [4, 16, 16]);
    let present = rebuilt.present_labels();
    assert!(
        present.contains(&ritk_annotation::LabelId(1)),
        "segment 1 must be present"
    );
    assert!(
        present.contains(&ritk_annotation::LabelId(2)),
        "segment 2 must be present"
    );
    assert!(
        rebuilt.count_label(1) > 0,
        "segment 1 voxels must survive reconstruction"
    );
    assert!(
        rebuilt.count_label(2) > 0,
        "segment 2 voxels must survive reconstruction"
    );
}

#[test]
fn test_read_external_highdicom_binary_seg_real_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("highdicom")
        .join("seg_image_ct_binary.dcm");

    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );

    let seg = read_dicom_seg(&path).expect("read external highdicom binary SEG");
    assert_eq!(seg.rows, 16);
    assert_eq!(seg.cols, 16);
    assert_eq!(seg.n_frames, 3);
    assert_eq!(seg.bits_allocated, 1);
    assert_eq!(seg.segmentation_type.as_dicom_str(), "BINARY");
    assert_eq!(seg.segments.len(), 1);
    assert_eq!(seg.segments[0].segment_label, "first segment");
    assert_eq!(
        seg.segments[0]
            .algorithm_type
            .as_ref()
            .map(|t| t.as_dicom_str()),
        Some("AUTOMATIC")
    );
    assert_eq!(seg.frame_segment_numbers, vec![1, 1, 1]);

    let rebuilt =
        dicom_seg_to_label_map(&seg).expect("rebuild label map from external highdicom binary SEG");
    assert_eq!(rebuilt.shape.0, [3, 16, 16]);
    assert!(rebuilt
        .present_labels()
        .contains(&ritk_annotation::LabelId(1)));
    assert!(
        rebuilt.count_label(1) > 0,
        "segment voxels must survive reconstruction"
    );
}

#[test]
fn test_read_external_rsna_dido_liver_seg_real_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("rsna_dido")
        .join("xTtzBC6F6p_rpexuszCnb_01_liver.dcm");

    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );

    let seg = read_dicom_seg(&path).expect("read external rsna dido liver SEG");
    assert_eq!(seg.rows, 512);
    assert_eq!(seg.cols, 512);
    assert_eq!(seg.n_frames, 34);
    assert_eq!(seg.bits_allocated, 1);
    assert_eq!(seg.segmentation_type.as_dicom_str(), "BINARY");
    assert_eq!(seg.segments.len(), 1);
    assert_eq!(seg.segments[0].segment_number, 1);
    assert_eq!(seg.segments[0].segment_label, "liver");
    assert_eq!(
        seg.segments[0]
            .algorithm_type
            .as_ref()
            .map(|t| t.as_dicom_str()),
        Some("MANUAL")
    );
    assert_eq!(seg.frame_segment_numbers, vec![1; 34]);

    let pixel_spacing = seg.pixel_spacing.expect("pixel spacing from shared FG");
    assert!((pixel_spacing[0] - 0.742188).abs() < 1e-6);
    assert!((pixel_spacing[1] - 0.742188).abs() < 1e-6);
    let slice_thickness = seg.slice_thickness.expect("slice thickness from shared FG");
    assert!((slice_thickness - 5.0).abs() < 1e-9);
    assert!(
        seg.image_position_per_frame.iter().all(|p| p.is_some()),
        "all frame positions must be present"
    );

    let rebuilt = dicom_seg_to_label_map(&seg).expect("rebuild label map from rsna dido SEG");
    assert_eq!(rebuilt.shape.0, [34, 512, 512]);
    assert!(rebuilt
        .present_labels()
        .contains(&ritk_annotation::LabelId(1)));
    assert!(
        rebuilt.count_label(1) > 0,
        "segment voxels must survive reconstruction"
    );
    assert_eq!(
        rebuilt.table.get_label(1).map(|e| e.name.as_str()),
        Some("liver")
    );
}
