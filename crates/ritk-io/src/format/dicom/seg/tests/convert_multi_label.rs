use super::super::{label_map_to_dicom_seg, SegEncoding};
use ritk_annotation::RgbaBytes;

#[test]
fn test_label_map_to_dicom_seg_multi_label() {
    use ritk_annotation::{LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(1, "Foreground", RgbaBytes::new(255, 0, 0, 255))
        .unwrap();
    table
        .add_label(2, "Other", RgbaBytes::new(0, 255, 0, 255))
        .unwrap();

    let data = vec![1, 1, 1, 1, 2, 2, 2, 2];
    let map = LabelMap::from_data([2, 2, 2], data, table).unwrap();
    let origin = [0.0, 0.0, 0.0];
    let spacing = [1.0, 1.0, 1.0];
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, SegEncoding::Binary)
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
