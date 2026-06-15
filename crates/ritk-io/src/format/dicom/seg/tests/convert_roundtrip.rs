use super::super::{
    dicom_seg_to_label_map, label_map_to_dicom_seg, read_dicom_seg, write_dicom_seg,
    DicomSegmentInfo, DicomSegmentation, SegEncoding, SegmentationType,
};
use ritk_annotation::RgbaBytes;

#[test]
fn test_dicom_seg_to_label_map_roundtrip_single_label() {
    use ritk_annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(1, "Label 1", [255, 0, 0, 255].into())
        .unwrap();
    let original = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        SegEncoding::Binary,
    )
    .expect("label_map_to_dicom_seg");

    let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
    assert_eq!(rebuilt.shape.0, [2, 2, 2]);
    assert_eq!(rebuilt.as_slice(), original.as_slice());
    assert_eq!(
        rebuilt.table.get_label(1).map(|e| e.name.as_str()),
        Some("Label 1")
    );
}

#[test]
fn test_dicom_seg_to_label_map_roundtrip_multi_label() {
    use ritk_annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(1, "A", RgbaBytes::new(255, 0, 0, 255))
        .unwrap();
    table
        .add_label(2, "B", RgbaBytes::new(0, 255, 0, 255))
        .unwrap();
    let original = LabelMap::from_data([2, 2, 2], vec![1, 1, 0, 2, 2, 0, 1, 2], table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        SegEncoding::Binary,
    )
    .expect("label_map_to_dicom_seg");

    let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
    assert_eq!(rebuilt.shape.0, [2, 2, 2]);
    assert_eq!(rebuilt.as_slice(), original.as_slice());
    assert!(rebuilt.table.get_label(1).is_some());
    assert!(rebuilt.table.get_label(2).is_some());
}

#[test]
fn test_dicom_seg_to_label_map_error_bad_frame_lengths() {
    let seg = DicomSegmentation {
        rows: 2,
        cols: 2,
        n_frames: 1,
        bits_allocated: 1,
        segmentation_type: SegmentationType::Binary,
        segments: vec![DicomSegmentInfo {
            segment_number: 1,
            segment_label: "A".to_owned(),
            segment_description: None,
            algorithm_type: None,
        }],
        frame_segment_numbers: vec![1],
        pixel_data: vec![vec![1, 0, 1]],
        image_position_per_frame: vec![None],
        image_orientation: None,
        pixel_spacing: None,
        slice_thickness: None,
    };

    let result = dicom_seg_to_label_map(&seg);
    assert!(result.is_err(), "invalid frame length must fail");
}

#[test]
fn test_dicom_seg_to_label_map_sparse_uneven_frames_supported() {
    let seg = DicomSegmentation {
        rows: 2,
        cols: 2,
        n_frames: 3,
        bits_allocated: 1,
        segmentation_type: SegmentationType::Binary,
        segments: vec![
            DicomSegmentInfo {
                segment_number: 1,
                segment_label: "A".to_owned(),
                segment_description: None,
                algorithm_type: None,
            },
            DicomSegmentInfo {
                segment_number: 2,
                segment_label: "B".to_owned(),
                segment_description: None,
                algorithm_type: None,
            },
        ],
        frame_segment_numbers: vec![1, 1, 2],
        pixel_data: vec![vec![1, 0, 0, 0], vec![0, 1, 0, 0], vec![0, 0, 1, 0]],
        image_position_per_frame: vec![None, None, None],
        image_orientation: None,
        pixel_spacing: None,
        slice_thickness: None,
    };

    let result =
        dicom_seg_to_label_map(&seg).expect("sparse uneven frame layout must be supported");
    assert_eq!(
        result.shape.0,
        [2, 2, 2],
        "nz inferred from max per-segment frame count"
    );
    let present = result.present_labels();
    assert!(
        present.contains(&ritk_annotation::LabelId(1)),
        "label 1 must be reconstructed"
    );
    assert!(
        present.contains(&ritk_annotation::LabelId(2)),
        "label 2 must be reconstructed"
    );
}

#[test]
fn test_dicom_seg_to_label_map_sorts_frames_by_physical_position() {
    let seg = DicomSegmentation {
        rows: 2,
        cols: 2,
        n_frames: 2,
        bits_allocated: 1,
        segmentation_type: SegmentationType::Binary,
        segments: vec![DicomSegmentInfo {
            segment_number: 1,
            segment_label: "A".to_owned(),
            segment_description: None,
            algorithm_type: None,
        }],
        frame_segment_numbers: vec![1, 1],
        pixel_data: vec![vec![1, 0, 0, 0], vec![0, 0, 0, 1]],
        // Intentionally out-of-order in z: first frame is z=5, second frame is z=0.
        image_position_per_frame: vec![Some([0.0, 0.0, 5.0]), Some([0.0, 0.0, 0.0])],
        image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        pixel_spacing: None,
        slice_thickness: None,
    };

    let rebuilt =
        dicom_seg_to_label_map(&seg).expect("position-sorted frame reconstruction must succeed");
    assert_eq!(rebuilt.shape.0, [2, 2, 2]);

    // z=0 slice must come from second frame: pixel (1,1) = label 1.
    assert_eq!(rebuilt.label_at([0, 1, 1]), 1);
    assert_eq!(rebuilt.label_at([0, 0, 0]), 0);

    // z=1 slice must come from first frame: pixel (0,0) = label 1.
    assert_eq!(rebuilt.label_at([1, 0, 0]), 1);
    assert_eq!(rebuilt.label_at([1, 1, 1]), 0);
}

#[test]
fn test_label_map_dicom_seg_file_roundtrip_identity() {
    use ritk_annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(1, "A", RgbaBytes::new(255, 0, 0, 255))
        .unwrap();
    table
        .add_label(2, "B", RgbaBytes::new(0, 255, 0, 255))
        .unwrap();
    let original =
        LabelMap::from_data([2, 3, 2], vec![1, 0, 2, 0, 1, 2, 2, 1, 0, 1, 0, 2], table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [2.0, 3.0, 4.0],
        [1.5, 0.75, 0.5],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        SegEncoding::Binary,
    )
    .expect("label_map_to_dicom_seg");

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("roundtrip_seg.dcm");

    write_dicom_seg(&path, &seg).expect("write_dicom_seg");
    let read_back = read_dicom_seg(&path).expect("read_dicom_seg");
    let rebuilt = dicom_seg_to_label_map(&read_back).expect("dicom_seg_to_label_map");

    assert_eq!(rebuilt.shape, original.shape, "shape must round-trip");
    assert_eq!(
        rebuilt.as_slice(),
        original.as_slice(),
        "voxel labels must round-trip"
    );
    assert_eq!(
        rebuilt.table.get_label(1).map(|e| e.name.as_str()),
        Some("A")
    );
    assert_eq!(
        rebuilt.table.get_label(2).map(|e| e.name.as_str()),
        Some("B")
    );
}
