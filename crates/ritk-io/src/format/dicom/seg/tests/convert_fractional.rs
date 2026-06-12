use super::super::{dicom_seg_to_label_map, label_map_to_dicom_seg, SegEncoding};
use ritk_annotation::RgbaBytes;

// ── LabelId cross-crate regression tests ──────────────────────────────────

/// DICOM-SEG remaps arbitrary LabelIds to sequential 1-based segment_numbers.
/// LabelId(42) and LabelId(7) become segment_number 1 and 2, and are
/// reconstructed as LabelId(1) and LabelId(2). The numeric LabelId is NOT
/// preserved through the round-trip — only voxel data (pixel masks) and
/// label names survive. This is by design: DICOM-SEG segment numbers are
/// sequential identifiers, not general-purpose label storage.
///
/// This test verifies that (a) voxel data round-trips correctly through
/// the 1-based remapping, and (b) label names are preserved.
#[test]
fn test_voxel_data_survives_dicom_seg_segment_number_remapping() {
    use ritk_annotation::{LabelId, LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(LabelId(42), "Lesion 42", RgbaBytes::new(255, 0, 0, 255))
        .unwrap();
    table
        .add_label(LabelId(7), "Region 7", RgbaBytes::new(0, 255, 0, 255))
        .unwrap();

    let data = vec![42, 42, 0, 7, 7, 42, 0, 0];
    let original = LabelMap::from_data([2, 2, 2], data, table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        SegEncoding::Binary,
    )
    .expect("label_map_to_dicom_seg");

    // DICOM-SEG assigns sequential segment_numbers starting from 1,
    // ordered by sorted foreground LabelIds (7 < 42 → seg 1 = LabelId(7)).
    assert_eq!(seg.segments.len(), 2);
    assert_eq!(seg.segments[0].segment_number, 1);
    assert_eq!(seg.segments[1].segment_number, 2);
    // Sorted order: LabelId(7) → segment 1, LabelId(42) → segment 2.
    assert_eq!(seg.segments[0].segment_label, "Region 7");
    assert_eq!(seg.segments[1].segment_label, "Lesion 42");

    let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
    assert_eq!(rebuilt.shape.0, [2, 2, 2]);

    // Voxel data survives as spatial pattern: original LabelId(7) → seg 1 →
    // LabelId(1), original LabelId(42) → seg 2 → LabelId(2). The numeric
    // LabelId values differ (7→1, 42→2), but the spatial assignment is
    // preserved: foreground voxels stay foreground, background stays background.
    let orig_slice = original.as_slice();
    let reb_slice = rebuilt.as_slice();
    for (i, &ov) in orig_slice.iter().enumerate() {
        if ov == 0 {
            assert_eq!(reb_slice[i], 0, "background voxel {i} must stay background");
        } else {
            assert_ne!(reb_slice[i], 0, "foreground voxel {i} must stay foreground");
        }
    }

    // Label names survive the round-trip.
    assert_eq!(
        rebuilt.table.get_label(LabelId(1)).map(|e| e.name.as_str()),
        Some("Region 7")
    );
    assert_eq!(
        rebuilt.table.get_label(LabelId(2)).map(|e| e.name.as_str()),
        Some("Lesion 42")
    );
}

/// DICOM-SEG remaps LabelIds to sequential 1-based segment_numbers, so
/// a single label always gets segment_number=1 regardless of its LabelId
/// value. Label names survive, and voxel data is consistently remapped.
#[test]
fn test_label_id_remapped_to_sequential_segment_numbers() {
    use ritk_annotation::{LabelId, LabelMap, LabelTable};

    let mut table = LabelTable::new();
    table
        .add_label(
            LabelId(u32::from(u16::MAX)),
            "Max",
            RgbaBytes::new(0, 0, 255, 255),
        )
        .unwrap();

    let original = LabelMap::from_data([1, 2, 2], vec![u32::from(u16::MAX); 4], table).unwrap();

    let seg = label_map_to_dicom_seg(
        &original,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        SegEncoding::Binary,
    )
    .expect("label_map_to_dicom_seg");

    // segment_number is always sequential from 1, regardless of LabelId value.
    assert_eq!(seg.segments.len(), 1);
    assert_eq!(seg.segments[0].segment_number, 1);
    assert_eq!(seg.segments[0].segment_label, "Max");

    let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
    // Reconstructed label is LabelId(1), not LabelId(u16::MAX).
    assert_eq!(rebuilt.shape.0, [1, 2, 2]);
    let present = rebuilt.present_labels();
    assert!(
        present.contains(&LabelId(1)),
        "reconstructed label must be 1"
    );
    assert_eq!(
        rebuilt.table.get_label(LabelId(1)).map(|e| e.name.as_str()),
        Some("Max")
    );
    // Voxel spatial pattern survives: all 4 voxels were foreground
    // (u16::MAX) and remain foreground (1) after remapping.
    let reb_slice = rebuilt.as_slice();
    for (i, &v) in reb_slice.iter().enumerate() {
        assert_ne!(v, 0, "voxel {i} must be foreground after remapping");
    }
}

/// segment_color must produce identical output for the same LabelId,
/// ensuring that the color lookup table is deterministic across
/// encode/decode cycles.
#[test]
fn test_segment_color_deterministic_for_label_id() {
    use super::super::converters::segment_color;
    use ritk_annotation::LabelId;

    let a = segment_color(LabelId(1));
    let b = segment_color(LabelId(1));
    assert_eq!(a, b, "segment_color must be deterministic");

    let c = segment_color(LabelId(2));
    assert_ne!(
        a, c,
        "different LabelIds should usually map to different colors"
    );
}

/// segment_color must accept LabelId::BACKGROUND (0) without panicking.
#[test]
fn test_segment_color_accepts_background_label_id() {
    use super::super::converters::segment_color;
    use ritk_annotation::LabelId;

    let color = segment_color(LabelId::BACKGROUND);
    assert_eq!(color.a(), 180, "background alpha must be 180");
}
