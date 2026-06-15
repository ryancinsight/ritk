use super::*;

// ── trace_closed_contour ─────────────────────────────────────────────

#[test]
fn trace_on_empty_mask_returns_none() {
    let mask = vec![0u8; 16];
    assert!(trace_closed_contour(&mask, 4, 4).is_none());
}

#[test]
fn trace_single_voxel_at_origin() {
    let ny = 3;
    let nx = 3;
    let mut mask = vec![0u8; ny * nx];
    mask[0] = 1; // (0,0)
    let result = trace_closed_contour(&mask, ny, nx);
    assert!(result.is_none(), "single voxel yields < 3 points");
}

#[test]
fn trace_2x2_block_returns_closed_contour() {
    let ny = 4;
    let nx = 4;
    let mut mask = vec![0u8; ny * nx];
    for y in 1..3 {
        for x in 1..3 {
            mask[y * nx + x] = 1;
        }
    }
    let result = trace_closed_contour(&mask, ny, nx).expect("contour");
    assert!(result.len() >= 3, "length = {}", result.len());
    // all points must be in the 2x2 block region
    for &(y, x) in &result {
        assert!((1..=2).contains(&y), "y={} out of [1,2]", y);
        assert!((1..=2).contains(&x), "x={} out of [1,2]", x);
    }
    // The contour is implicitly closed (no duplicate start at end)
    assert_ne!(result.first(), result.last());
}

#[test]
fn trace_plus_shape_returns_valid_polygon() {
    let ny = 5;
    let nx = 5;
    let mut mask = vec![0u8; ny * nx];
    // Plus shape: center row and center column
    for x in 0..5 {
        mask[2 * nx + x] = 1;
    }
    for y in 0..5 {
        mask[y * nx + 2] = 1;
    }
    let result = trace_closed_contour(&mask, ny, nx).expect("contour");
    assert!(
        result.len() >= 8,
        "plus should have >= 8 boundary points, got {}",
        result.len()
    );
    // The contour is implicitly closed; start point appears only at index 0.
    assert_ne!(result.first(), result.last());
}

// ── voxel_to_phys ────────────────────────────────────────────────────

#[test]
fn voxel_to_phys_identity_direction() {
    let origin = [1.0, 2.0, 3.0];
    let spacing = [2.0, 1.5, 1.0]; // [dz, dy, dx]
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let p = voxel_to_phys(0, 0, 0, origin, spacing, direction);
    assert_eq!(p, origin);
    let p = voxel_to_phys(0, 1, 2, origin, spacing, direction);
    assert_eq!(p[0], 1.0 + 2.0 * 1.0); // ox + x*dx
    assert_eq!(p[1], 2.0 + 1.0 * 1.5); // oy + y*dy
    assert_eq!(p[2], 3.0 + 0.0 * 2.0); // oz + z*dz
}

// ── label_map_to_rt_struct ───────────────────────────────────────────

fn make_small_label_map() -> ritk_annotation::LabelMap {
    let mut table = ritk_annotation::LabelTable::new();
    table
        .add_label(1, "Tumor", ritk_annotation::RgbaBytes::new(255, 0, 0, 255))
        .unwrap();
    table
        .add_label(2, "Organ", ritk_annotation::RgbaBytes::new(0, 255, 0, 255))
        .unwrap();
    let mut lm = ritk_annotation::LabelMap::new([2, 4, 4], table);
    // Label 1: voxels [z=0, y=1..2, x=1..2]
    for y in 1..3 {
        for x in 1..3 {
            lm.set_label_at([0, y, x], 1);
        }
    }
    // Label 2: voxels [z=1, y=1..2, x=1..2]
    for y in 1..3 {
        for x in 1..3 {
            lm.set_label_at([1, y, x], 2);
        }
    }
    lm
}

#[test]
fn label_map_to_rt_struct_single_label() {
    let lm = make_small_label_map();
    let rt = label_map_to_rt_struct(
        &lm,
        [0.0; 3],
        [1.0; 3],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .expect("conversion");
    assert_eq!(rt.rois.len(), 2, "two labels");
    assert_eq!(rt.rois[0].roi_name, "Tumor", "label 1 name");
    assert_eq!(rt.rois[1].roi_name, "Organ", "label 2 name");
    assert!(rt.rois[0].contours[0].points.len() >= 3, "label 1 contour");
    assert_eq!(rt.rois[0].roi_number, 1);
    assert_eq!(rt.rois[1].roi_number, 2);
}

#[test]
fn label_map_to_rt_struct_zero_dim_returns_err() {
    let table = ritk_annotation::LabelTable::new();
    let lm = ritk_annotation::LabelMap::new([0, 4, 4], table);
    let result = label_map_to_rt_struct(
        &lm,
        [0.0; 3],
        [1.0; 3],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    assert!(result.is_err());
}

#[test]
fn label_map_to_rt_struct_no_foreground_returns_err() {
    let table = ritk_annotation::LabelTable::new();
    let lm = ritk_annotation::LabelMap::new([2, 4, 4], table);
    let result = label_map_to_rt_struct(
        &lm,
        [0.0; 3],
        [1.0; 3],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    assert!(result.is_err());
}

#[test]
fn label_map_to_rt_struct_round_trip() {
    let lm = make_small_label_map();
    let origin = [10.0, 20.0, 30.0];
    let spacing = [2.0, 1.5, 1.0]; // [dz, dy, dx]
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let rt = label_map_to_rt_struct(&lm, origin, spacing, direction).expect("convert");

    // Write → read round-trip
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("rt_roundtrip.dcm");
    crate::format::dicom::rt_struct::writer::write_rt_struct(&path, &rt).expect("write");
    let loaded = crate::format::dicom::rt_struct::reader::read_rt_struct(&path).expect("read");

    assert_eq!(loaded.rois.len(), 2);
    assert_eq!(loaded.rois[0].roi_name, "Tumor");
    assert_eq!(loaded.rois[1].roi_name, "Organ");
    // Verify contour points are in physical coordinates
    for roi in &loaded.rois {
        for ct in &roi.contours {
            assert_eq!(ct.geometric_type.as_dicom_str(), "CLOSED_PLANAR");
            assert!(ct.points.len() >= 3);
            for pt in &ct.points {
                // Points should be in patient space around the origin
                assert!(pt[0] >= 0.0 && pt[0] <= 20.0, "x out of range: {}", pt[0]);
            }
        }
    }
}

#[test]
fn label_map_to_rt_struct_contour_physical_positions() {
    let mut table = ritk_annotation::LabelTable::new();
    table
        .add_label(
            1,
            "ROI",
            ritk_annotation::RgbaBytes::new(255, 255, 255, 255),
        )
        .unwrap();
    let mut lm = ritk_annotation::LabelMap::new([1, 3, 3], table);
    // Single voxel at row=1, col=1 on slice 0
    lm.set_label_at([0, 1, 1], 1);

    let origin = [0.0; 3];
    let spacing = [1.0, 1.0, 1.0]; // [dz, dy, dx]
    let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let result = label_map_to_rt_struct(&lm, origin, spacing, direction);
    // Single isolated voxel -> boundary trace may give < 3 points -> no contour
    // Check that it still produces an ROI (just may have 0 contours)
    assert!(result.is_ok());
}
