use super::super::{label_map_to_dicom_seg, SegEncoding};
use ritk_annotation::RgbaBytes;

#[test]
fn test_label_map_to_dicom_seg_identity_single_label() {
    use ritk_annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(1, "Label 1", RgbaBytes::new(255, 0, 0, 255))
        .unwrap();

    let map = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();
    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, SegEncoding::Binary)
        .expect("label_map_to_dicom_seg");

    assert_eq!(seg.rows, 2, "rows must equal ny");
    assert_eq!(seg.cols, 2, "cols must equal nx");
    assert_eq!(seg.n_frames, 2, "nz=2 frames, one per Z-slice per segment");
    assert_eq!(
        seg.bits_allocated, 1,
        "SegEncoding::Binary → bits_allocated=1"
    );
    assert_eq!(seg.segmentation_type.as_dicom_str(), "BINARY");
    assert_eq!(seg.segments.len(), 1, "one segment");
    assert_eq!(seg.segments[0].segment_label, "Label 1");
    assert_eq!(seg.segments[0].segment_number, 1);
    assert_eq!(seg.pixel_data[0], vec![1u8; 4], "frame 0 all ones");
    assert_eq!(seg.pixel_data[1], vec![1u8; 4], "frame 1 all ones");
}

#[test]
fn test_label_map_to_dicom_seg_background_excluded() {
    use ritk_annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(1, "Lesion", RgbaBytes::new(255, 0, 0, 255))
        .unwrap();

    let data = vec![0, 0, 1, 1];
    let map = LabelMap::from_data([1, 2, 2], data, table).unwrap();
    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, SegEncoding::Binary)
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
    use ritk_annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(1, "A", RgbaBytes::new(255, 0, 0, 255))
        .unwrap();

    let map = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();
    let origin = [10.0, 20.0, 30.0];
    let spacing = [2.0, 0.5, 1.5];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, SegEncoding::Binary)
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
    use ritk_annotation::{LabelMap, LabelTable};

    let table = LabelTable::new();
    let map = LabelMap::new([1, 2, 2], table);

    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let result = label_map_to_dicom_seg(&map, origin, spacing, direction, SegEncoding::Binary);
    assert!(result.is_err(), "all-background map should return Err");
}

#[test]
fn test_label_map_to_dicom_seg_error_no_foreground() {
    use ritk_annotation::{LabelMap, LabelTable};

    let table = LabelTable::new();
    let map = LabelMap::from_data([2, 2, 2], vec![0u32; 8], table).unwrap();

    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let result = label_map_to_dicom_seg(&map, origin, spacing, direction, SegEncoding::Binary);
    assert!(result.is_err(), "all-background map should return Err");
}
