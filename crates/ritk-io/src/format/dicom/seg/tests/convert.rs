use super::super::{
    dicom_seg_to_label_map, label_map_to_dicom_seg, read_dicom_seg, write_dicom_seg,
    DicomSegmentInfo, DicomSegmentation,
};
use arrayvec::ArrayString;

#[test]
fn test_label_map_to_dicom_seg_identity_single_label() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table.add_label(1, "Label 1", [255, 0, 0, 255]).unwrap();

    let map = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();
    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
        .expect("label_map_to_dicom_seg");

    assert_eq!(seg.rows, 2, "rows must equal ny");
    assert_eq!(seg.cols, 2, "cols must equal nx");
    assert_eq!(seg.n_frames, 2, "nz=2 frames, one per Z-slice per segment");
    assert_eq!(seg.bits_allocated, 1, "use_binary=true → bits_allocated=1");
    assert_eq!(seg.segmentation_type.as_str(), "BINARY");
    assert_eq!(seg.segments.len(), 1, "one segment");
    assert_eq!(seg.segments[0].segment_label, "Label 1");
    assert_eq!(seg.segments[0].segment_number, 1);
    assert_eq!(seg.pixel_data[0], vec![1u8; 4], "frame 0 all ones");
    assert_eq!(seg.pixel_data[1], vec![1u8; 4], "frame 1 all ones");
}

#[test]
fn test_label_map_to_dicom_seg_multi_label() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table.add_label(1, "Foreground", [255, 0, 0, 255]).unwrap();
    table.add_label(2, "Other", [0, 255, 0, 255]).unwrap();

    let data = vec![1, 1, 1, 1, 2, 2, 2, 2];
    let map = LabelMap::from_data([2, 2, 2], data, table).unwrap();
    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
        .expect("label_map_to_dicom_seg");

    assert_eq!(seg.n_frames, 4, "2 segments × 2 z-slices = 4 frames");
    assert_eq!(seg.segments.len(), 2, "two segments");
    assert_eq!(
        seg.frame_segment_numbers,
        vec![1, 1, 2, 2],
        "segment assignment per frame"
    );
    assert_eq!(
        seg.pixel_data[0],
        vec![1u8; 4],
        "frame 0 (seg 1, z=0) all ones"
    );
    assert_eq!(
        seg.pixel_data[1],
        vec![0u8; 4],
        "frame 1 (seg 1, z=1) all zeros"
    );
    assert_eq!(
        seg.pixel_data[2],
        vec![0u8; 4],
        "frame 2 (seg 2, z=0) all zeros"
    );
    assert_eq!(
        seg.pixel_data[3],
        vec![1u8; 4],
        "frame 3 (seg 2, z=1) all ones"
    );
}

#[test]
fn test_label_map_to_dicom_seg_background_excluded() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table.add_label(1, "Lesion", [255, 0, 0, 255]).unwrap();

    let data = vec![0, 0, 1, 1];
    let map = LabelMap::from_data([1, 2, 2], data, table).unwrap();
    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
        .expect("label_map_to_dicom_seg");

    assert_eq!(seg.n_frames, 1, "1 foreground label × 1 z-slice = 1 frame");
    assert_eq!(
        seg.pixel_data[0],
        vec![0, 0, 1, 1],
        "correct mask for label 1"
    );
}

#[test]
fn test_label_map_to_dicom_seg_spatial_metadata() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table.add_label(1, "A", [255, 0, 0, 255]).unwrap();

    let map = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();
    let origin = [10.0, 20.0, 30.0];
    let spacing = [2.0, 0.5, 1.5];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
        .expect("label_map_to_dicom_seg");

    assert_eq!(
        seg.image_orientation,
        Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        "image_orientation from direction"
    );
    assert_eq!(
        seg.pixel_spacing,
        Some([0.5, 1.5]),
        "pixel_spacing [ny_spacing, nx_spacing]"
    );
    assert_eq!(
        seg.slice_thickness,
        Some(2.0),
        "slice_thickness from spacing[0]"
    );
    assert_eq!(
        seg.image_position_per_frame[0],
        Some([10.0, 20.0, 30.0]),
        "frame 0 position at z=0"
    );
    assert_eq!(
        seg.image_position_per_frame[1],
        Some([10.0, 20.0, 32.0]),
        "frame 1 position at z=1 (z_offset = 1*spacing[0]=2)"
    );
}

#[test]
fn test_label_map_to_dicom_seg_error_empty_geometry() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let table = LabelTable::new();
    let map = LabelMap::new([1, 2, 2], table);

    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let result = label_map_to_dicom_seg(&map, origin, spacing, direction, true);
    assert!(result.is_err(), "all-background map should return Err");
}

#[test]
fn test_label_map_to_dicom_seg_error_no_foreground() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let table = LabelTable::new();
    let map = LabelMap::from_data([2, 2, 2], vec![0u32; 8], table).unwrap();

    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let result = label_map_to_dicom_seg(&map, origin, spacing, direction, true);
    assert!(result.is_err(), "all-background map should return Err");
}

#[test]
fn test_dicom_seg_to_label_map_roundtrip_single_label() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table.add_label(1, "Label 1", [255, 0, 0, 255]).unwrap();
    let original = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        true,
    )
    .expect("label_map_to_dicom_seg");

    let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
    assert_eq!(rebuilt.shape, [2, 2, 2]);
    assert_eq!(rebuilt.as_slice(), original.as_slice());
    assert_eq!(
        rebuilt.table.get_label(1).map(|e| e.name.as_str()),
        Some("Label 1")
    );
}

#[test]
fn test_dicom_seg_to_label_map_roundtrip_multi_label() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table.add_label(1, "A", [255, 0, 0, 255]).unwrap();
    table.add_label(2, "B", [0, 255, 0, 255]).unwrap();
    let original = LabelMap::from_data([2, 2, 2], vec![1, 1, 0, 2, 2, 0, 1, 2], table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        true,
    )
    .expect("label_map_to_dicom_seg");

    let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
    assert_eq!(rebuilt.shape, [2, 2, 2]);
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
        segmentation_type: ArrayString::from("BINARY").unwrap(),
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
        segmentation_type: ArrayString::from("BINARY").unwrap(),
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
        result.shape,
        [2, 2, 2],
        "nz inferred from max per-segment frame count"
    );
    let present = result.present_labels();
    assert!(present.contains(&1), "label 1 must be reconstructed");
    assert!(present.contains(&2), "label 2 must be reconstructed");
}

#[test]
fn test_dicom_seg_to_label_map_sorts_frames_by_physical_position() {
    let seg = DicomSegmentation {
        rows: 2,
        cols: 2,
        n_frames: 2,
        bits_allocated: 1,
        segmentation_type: ArrayString::from("BINARY").unwrap(),
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
    assert_eq!(rebuilt.shape, [2, 2, 2]);

    // z=0 slice must come from second frame: pixel (1,1) = label 1.
    assert_eq!(rebuilt.label_at([0, 1, 1]), 1);
    assert_eq!(rebuilt.label_at([0, 0, 0]), 0);

    // z=1 slice must come from first frame: pixel (0,0) = label 1.
    assert_eq!(rebuilt.label_at([1, 0, 0]), 1);
    assert_eq!(rebuilt.label_at([1, 1, 1]), 0);
}

#[test]
fn test_label_map_dicom_seg_file_roundtrip_identity() {
    use ritk_core::annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table.add_label(1, "A", [255, 0, 0, 255]).unwrap();
    table.add_label(2, "B", [0, 255, 0, 255]).unwrap();
    let original =
        LabelMap::from_data([2, 3, 2], vec![1, 0, 2, 0, 1, 2, 2, 1, 0, 1, 0, 2], table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [2.0, 3.0, 4.0],
        [1.5, 0.75, 0.5],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        true,
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
