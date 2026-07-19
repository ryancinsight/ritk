use super::*;
use ritk_io::{RtDoseGrid, RtDoseSummationType, RtDoseType};

// Helper to build a minimal RtDoseGrid for testing.
fn make_dose_grid(
    rows: usize,
    cols: usize,
    n_frames: usize,
    dose_values: Vec<f64>,
    origin: [f64; 3],
    orient: [f64; 6],
    spacing: [f64; 2],
    frame_offsets: Vec<f64>,
) -> RtDoseGrid {
    RtDoseGrid {
        rows,
        cols,
        n_frames,
        dose_type: RtDoseType::Physical,
        dose_summation_type: RtDoseSummationType::Plan,
        dose_grid_scaling: 1.0,
        frame_offsets,
        dose_gy: dose_values,
        image_position: Some(origin),
        image_orientation: Some(orient),
        pixel_spacing: Some(spacing),
        referenced_rt_plan_sop_instance_uid: None,
    }
}

#[test]
fn invert3x3_identity_returns_identity() {
    // Analytical: inv(I) = I.
    let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let inv = invert3x3(id).expect("identity is invertible");
    for i in 0..9 {
        assert!(
            (inv[i] - id[i]).abs() < 1e-10,
            "inv[{i}] = {} â‰  {}",
            inv[i],
            id[i]
        );
    }
}

#[test]
fn invert3x3_singular_returns_none() {
    // Rows are linearly dependent â†’ det = 0.
    let m = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 0.0];
    assert!(invert3x3(m).is_none());
}

#[test]
fn invert3x3_round_trip() {
    // A Â· Aâ»Â¹ = I (Frobenius error < 1e-10).
    let m = [2.0, 1.0, 0.0, 1.5, 3.0, 0.5, 0.0, 0.5, 2.0];
    let inv = invert3x3(m).expect("matrix must be invertible");
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0f64;
            for k in 0..3 {
                sum += m[i * 3 + k] * inv[k * 3 + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-10,
                "mÂ·inv[{i},{j}] = {sum:.2e} â‰  {expected}"
            );
        }
    }
}

#[test]
fn dose_to_rgba_nan_returns_transparent() {
    let rgba = dose_to_rgba(f32::NAN, 0.0, 60.0, 0.5);
    assert_eq!(rgba[3], 0, "NaN dose should have alpha=0");
}

#[test]
fn dose_to_rgba_zero_dose_returns_transparent() {
    let rgba = dose_to_rgba(0.0, 0.0, 60.0, 0.5);
    assert_eq!(rgba[3], 0, "zero dose should have alpha=0");
}

#[test]
fn dose_to_rgba_max_dose_is_red() {
    // At max dose (t=1.0), the colormap is in the red segment.
    let rgba = dose_to_rgba(60.0, 0.0, 60.0, 1.0);
    assert!(rgba[0] > 200, "red channel should be high at max dose");
    assert!(rgba[3] > 0, "alpha should be non-zero at max dose");
}

#[test]
fn dose_to_rgba_midpoint_is_greenish() {
    // At t=0.5 (midpoint), the colormap yields green.
    let rgba = dose_to_rgba(30.0, 0.0, 60.0, 1.0);
    assert!(rgba[1] > 150, "green channel high at midpoint");
    assert!(rgba[3] > 0, "alpha should be non-zero at midpoint");
}

#[test]
fn cross3_unit_vectors() {
    // Analytical: x Ã— y = z, y Ã— z = x, z Ã— x = y.
    let x = [1.0_f64, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let z = [0.0, 0.0, 1.0];
    let xy = cross3(x, y);
    assert!(
        (xy[0] - z[0]).abs() < 1e-12
            && (xy[1] - z[1]).abs() < 1e-12
            && (xy[2] - z[2]).abs() < 1e-12,
        "xÃ—y != z: {:?}",
        xy
    );
    let yz = cross3(y, z);
    assert!(
        (yz[0] - x[0]).abs() < 1e-12
            && (yz[1] - x[1]).abs() < 1e-12
            && (yz[2] - x[2]).abs() < 1e-12,
        "yÃ—z != x: {:?}",
        yz
    );
}

#[test]
fn extract_dose_axial_identity_grid() {
    let n = 4;
    let dose = make_dose_grid(
        n,
        n,
        n,
        vec![2.0_f64; n * n * n],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0],
        vec![0.0, 1.0, 2.0, 3.0],
    );
    let vol_dir = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let result = extract_dose_slice_for_volume(
        &dose,
        0,
        0,
        [n, n, n],
        [0.0, 0.0, 0.0],
        vol_dir,
        [1.0, 1.0, 1.0],
    );
    let map = result.expect("should produce a dose map");
    assert_eq!(map.len(), n * n);
    let valid: Vec<f32> = map.iter().cloned().filter(|v| v.is_finite()).collect();
    assert!(!valid.is_empty(), "expected some finite dose values");
    for &d in &valid {
        assert!((d - 2.0).abs() < 1e-4, "expected 2.0 Gy, got {d}");
    }
}

#[test]
fn extract_dose_no_spatial_metadata_returns_none() {
    let dose = RtDoseGrid {
        rows: 4,
        cols: 4,
        n_frames: 1,
        dose_type: RtDoseType::Physical,
        dose_summation_type: RtDoseSummationType::Plan,
        dose_grid_scaling: 1.0,
        frame_offsets: vec![0.0],
        dose_gy: vec![1.0; 16],
        image_position: None,
        image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        pixel_spacing: Some([1.0, 1.0]),
        referenced_rt_plan_sop_instance_uid: None,
    };
    let result = extract_dose_slice_for_volume(
        &dose,
        0,
        0,
        [4, 4, 4],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    );
    assert!(result.is_none(), "missing origin must return None");
}
