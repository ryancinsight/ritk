#![allow(clippy::needless_range_loop)]

use arrayvec::ArrayString;
use super::*;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn write_rt_dose_file(obj: InMemDicomObject, path: &std::path::Path) {
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(RT_DOSE_SOP_CLASS_UID)
            .media_storage_sop_instance_uid("2.25.1")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("meta build")
    .write_to_file(path)
    .expect("write RT Dose file");
}

fn write_wrong_sop_file(sop: &str, path: &std::path::Path) {
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(sop),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.99"),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(dicom::core::smallvec::SmallVec::new()),
    ));
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(sop)
            .media_storage_sop_instance_uid("2.25.99")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("meta")
    .write_to_file(path)
    .expect("write wrong-SOP file");
}

fn build_rt_dose_obj(
    rows: u16,
    cols: u16,
    n_frames: u32,
    dose_grid_scaling: f64,
    pixel_val: u32,
) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(rows),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(cols),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(n_frames.to_string().as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(32u16),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x000E),
        VR::DS,
        PrimitiveValue::from(format!("{}", dose_grid_scaling).as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x0002),
        VR::CS,
        PrimitiveValue::from("PLAN"),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x0004),
        VR::CS,
        PrimitiveValue::from("PHYSICAL"),
    ));

    let offset_str: String = (0..n_frames as usize)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join("\\");
    obj.put(DataElement::new(
        Tag(0x3004, 0x000C),
        VR::DS,
        PrimitiveValue::from(offset_str.as_str()),
    ));

    let n_voxels = n_frames as usize * rows as usize * cols as usize;
    let mut pixel_bytes: Vec<u8> = Vec::with_capacity(n_voxels * 4);
    for _ in 0..n_voxels {
        pixel_bytes.extend_from_slice(&pixel_val.to_le_bytes());
    }
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(pixel_bytes)),
    ));

    obj
}

// ── Test A: missing file ──────────────────────────────────────────────────────

/// Invariant: a nonexistent path must produce Err mentioning the path or open failure.
#[test]
fn test_read_rt_dose_missing_file_returns_error() {
    let result = read_rt_dose("/nonexistent/path.dcm");
    assert!(result.is_err(), "nonexistent path must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("nonexistent") || msg.contains("open"),
        "error must mention path or open failure; got: {msg}"
    );
}

// ── Test B: wrong SOP class ───────────────────────────────────────────────────

/// Invariant: a file whose SOP Class UID ≠ RT Dose must produce Err
/// containing the rejected UID in the message.
#[test]
fn test_read_rt_dose_wrong_sop_class_returns_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("ct.dcm");
    write_wrong_sop_file("1.2.840.10008.5.1.4.1.1.2", &path);

    let result = read_rt_dose(&path);
    assert!(result.is_err(), "wrong SOP class must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("1.2.840.10008.5.1.4.1.1.2"),
        "error must contain the rejected SOP UID; got: {msg}"
    );
}

// ── Test C: synthetic 4×4×2 grid ─────────────────────────────────────────────

/// Invariant: for pixel_val = 1000 and dose_grid_scaling = 0.001,
/// every dose_gy[i] = 1000 × 0.001 = 1.0 Gy (exact in f64 representation).
#[test]
fn test_read_rt_dose_synthetic_grid() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("rt_dose.dcm");

    let obj = build_rt_dose_obj(4, 4, 2, 0.001, 1000);
    write_rt_dose_file(obj, &path);

    let grid = read_rt_dose(&path).expect("read_rt_dose synthetic");

    assert_eq!(grid.rows, 4, "rows");
    assert_eq!(grid.cols, 4, "cols");
    assert_eq!(grid.n_frames, 2, "n_frames");
    assert_eq!(grid.dose_gy.len(), 4 * 4 * 2, "dose_gy length");

    for (i, &v) in grid.dose_gy.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-12, "dose_gy[{i}] = {v}, expected 1.0");
    }

    assert_eq!(grid.dose_grid_scaling, 0.001, "dose_grid_scaling");
    assert_eq!(grid.dose_summation_type.as_str(), "PLAN", "dose_summation_type");
    assert_eq!(grid.dose_type.as_str(), "PHYSICAL", "dose_type");

    assert_eq!(grid.frame_offsets.len(), 2, "frame_offsets length");
    assert!(
        (grid.frame_offsets[0] - 0.0).abs() < 1e-12,
        "frame_offsets[0] = {}, expected 0.0",
        grid.frame_offsets[0]
    );
    assert!(
        (grid.frame_offsets[1] - 1.0).abs() < 1e-12,
        "frame_offsets[1] = {}, expected 1.0",
        grid.frame_offsets[1]
    );
}

// ── Test D: validation rejects mismatched voxel count ────────────────────────

/// Invariant: write_rt_dose must return Err when dose_gy.len() ≠ n_frames * rows * cols.
#[test]
fn test_write_rt_dose_rejects_mismatched_voxel_count() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("dose_bad.dcm");
    let grid = RtDoseGrid {
        rows: 2,
        cols: 2,
        n_frames: 1,
        dose_type: ArrayString::from("PHYSICAL").unwrap(),
        dose_summation_type: ArrayString::from("PLAN").unwrap(),
        dose_grid_scaling: 0.001,
        frame_offsets: vec![0.0],
        dose_gy: vec![0.0, 0.001, 0.002, 0.003, 0.004],
        image_position: None,
        image_orientation: None,
        pixel_spacing: None,
        referenced_rt_plan_sop_instance_uid: None,
    };
    let result = write_rt_dose(&path, &grid);
    assert!(result.is_err(), "mismatched voxel count must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("dose_gy") || msg.contains("voxel") || msg.contains('5') || msg.contains('4'),
        "error message must reference the count mismatch; got: {msg}"
    );
}

// ── Test E: round-trip write/read preserves all fields ───────────────────────

/// Invariant: write_rt_dose followed by read_rt_dose reconstructs all fields
/// within the quantization tolerance of dose_grid_scaling.
#[test]
fn test_write_rt_dose_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("dose_rt.dcm");

    let dose_gy: Vec<f64> = (0..32).map(|i| i as f64 * 0.001).collect();
    let grid = RtDoseGrid {
        rows: 4,
        cols: 4,
        n_frames: 2,
        dose_type: ArrayString::from("PHYSICAL").unwrap(),
        dose_summation_type: ArrayString::from("BEAM").unwrap(),
        dose_grid_scaling: 0.001,
        frame_offsets: vec![0.0, 5.0],
        dose_gy: dose_gy.clone(),
        image_position: Some([10.0, 20.0, 30.0]),
        image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        pixel_spacing: Some([2.5, 2.5]),
        referenced_rt_plan_sop_instance_uid: Some(ArrayString::from("2.25.12345").unwrap()),
    };

    write_rt_dose(&path, &grid).expect("write_rt_dose round-trip");
    let back = read_rt_dose(&path).expect("read_rt_dose round-trip");

    assert_eq!(back.rows, 4, "rows");
    assert_eq!(back.cols, 4, "cols");
    assert_eq!(back.n_frames, 2, "n_frames");
    assert_eq!(back.dose_gy.len(), 32, "dose_gy.len");

    for i in 0..32 {
        assert!(
            (back.dose_gy[i] - dose_gy[i]).abs() < 1e-12,
            "dose_gy[{i}]: got {}, expected {}",
            back.dose_gy[i],
            dose_gy[i]
        );
    }

    assert_eq!(back.dose_grid_scaling, 0.001, "dose_grid_scaling");
    assert_eq!(back.dose_summation_type.as_str(), "BEAM", "dose_summation_type");

    assert!(
        (back.frame_offsets[0] - 0.0).abs() < 1e-12,
        "frame_offsets[0] = {}, expected 0.0",
        back.frame_offsets[0]
    );
    assert!(
        (back.frame_offsets[1] - 5.0).abs() < 1e-12,
        "frame_offsets[1] = {}, expected 5.0",
        back.frame_offsets[1]
    );

    let pos = back.image_position.expect("image_position must be Some");
    assert!(
        (pos[0] - 10.0).abs() < 1e-6,
        "image_position[0] = {}",
        pos[0]
    );
    assert!(
        (pos[1] - 20.0).abs() < 1e-6,
        "image_position[1] = {}",
        pos[1]
    );
    assert!(
        (pos[2] - 30.0).abs() < 1e-6,
        "image_position[2] = {}",
        pos[2]
    );

    let ps = back.pixel_spacing.expect("pixel_spacing must be Some");
    assert!((ps[0] - 2.5).abs() < 1e-6, "pixel_spacing[0] = {}", ps[0]);
    assert!((ps[1] - 2.5).abs() < 1e-6, "pixel_spacing[1] = {}", ps[1]);

    assert_eq!(
        back.referenced_rt_plan_sop_instance_uid.as_deref(),
        Some("2.25.12345"),
        "referenced RT plan SOP Instance UID"
    );
}
