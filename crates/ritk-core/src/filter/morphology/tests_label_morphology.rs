//! Tests for label_morphology
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
type B = NdArray<f32>;
fn img(v: Vec<f32>, d: [usize; 3]) -> Image<B, 3> {
    let t = Tensor::<B, 3>::from_data(TensorData::new(v, Shape::new(d)), &Default::default());
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}
fn vv(i: &Image<B, 3>) -> Vec<f32> {
    i.data().clone().into_data().into_vec::<f32>().unwrap()
}

#[test]
fn test_all_background_unchanged() {
    let d = [6, 6, 6];
    let n = d[0] * d[1] * d[2];
    let v = vec![0.0_f32; n];
    let out = vv(&LabelDilation::new(1).apply(&img(v.clone(), d)).unwrap());
    for &x in &out {
        assert!(x.abs() < 1e-6, "all-bg: {x}");
    }
}
#[test]
fn test_label_expands_into_background() {
    let d = [7, 7, 7];
    let [nz, ny, nx] = d;
    let n = nz * ny * nx;
    let mut v = vec![0.0_f32; n];
    let c = 3 * ny * nx + 3 * nx + 3;
    v[c] = 1.0;
    let out = vv(&LabelDilation::new(1).apply(&img(v, d)).unwrap());
    assert!((out[c] - 1.0).abs() < 1e-6, "centre preserved");
    let neighbour = 3 * ny * nx + 3 * nx + 4;
    assert!(
        (out[neighbour] - 1.0).abs() < 1e-6,
        "neighbour expanded to label 1"
    );
}
#[test]
fn test_conflict_min_label_wins() {
    let d = [5, 5, 5];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mut v = vec![0.0_f32; n];
    let a = 2 * ny * nx + 1 * nx + 1;
    v[a] = 1.0;
    let b = 2 * ny * nx + 1 * nx + 3;
    v[b] = 2.0;
    let out = vv(&LabelDilation::new(1).apply(&img(v, d)).unwrap());
    let middle = 2 * ny * nx + 1 * nx + 2;
    assert!(
        (out[middle] - 1.0).abs() < 1e-6,
        "conflict: min label wins (got {})",
        out[middle]
    );
}
#[test]
fn test_radius_zero_identity() {
    let d = [6, 6, 6];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mut v = vec![0.0_f32; n];
    v[2 * ny * nx + 2 * nx + 2] = 1.0;
    v[3 * ny * nx + 3 * nx + 3] = 2.0;
    let out = vv(&LabelDilation::new(0).apply(&img(v.clone(), d)).unwrap());
    for (i, (&e, &a)) in v.iter().zip(out.iter()).enumerate() {
        assert!((a - e).abs() < 1e-6, "r=0 identity voxel {i}: {a} != {e}");
    }
}
#[test]
fn test_metadata_preserved() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.0_f32; n], Shape::new(d)),
        &Default::default(),
    );
    let o = Point::new([1.0, 2.0, 3.0]);
    let s = Spacing::new([0.5, 0.5, 0.5]);
    let r = LabelDilation::new(1)
        .apply(&Image::new(t, o, s, Direction::identity()))
        .unwrap();
    assert_eq!(*r.origin(), o);
    assert_eq!(*r.spacing(), s);
}

// ── LabelErosion tests ─────────────────────────────────────────────────

#[test]
fn test_label_erosion_all_background_unchanged() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let out = vv(&LabelErosion::new(1)
        .apply(&img(vec![0.0_f32; n], d))
        .unwrap());
    for &x in &out {
        assert!(x.abs() < 1e-6, "all-bg erosion: {x}");
    }
}

#[test]
fn test_label_erosion_single_voxel_erodes_to_zero() {
    let d = [5, 5, 5];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mut v = vec![0.0_f32; n];
    let c = 2 * ny * nx + 2 * nx + 2;
    v[c] = 1.0;
    let out = vv(&LabelErosion::new(1).apply(&img(v, d)).unwrap());
    assert!(
        out[c].abs() < 1e-6,
        "single voxel should erode to 0, got {}",
        out[c]
    );
}

#[test]
fn test_label_erosion_radius_zero_is_identity() {
    let d = [7, 7, 7];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mut v = vec![0.0_f32; n];
    v[3 * ny * nx + 3 * nx + 3] = 1.0;
    let out = vv(&LabelErosion::new(0).apply(&img(v.clone(), d)).unwrap());
    for (i, (&e, &a)) in v.iter().zip(out.iter()).enumerate() {
        assert!((a - e).abs() < 1e-6, "r=0 identity voxel {i}: {a} != {e}");
    }
}

#[test]
fn test_label_erosion_interior_preserved() {
    let d = [9, 9, 9];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mut v = vec![0.0_f32; n];
    for iz in 1..8 {
        for iy in 1..8 {
            for ix in 1..8 {
                v[iz * ny * nx + iy * nx + ix] = 1.0;
            }
        }
    }
    let out = vv(&LabelErosion::new(1).apply(&img(v, d)).unwrap());
    let centre = 4 * ny * nx + 4 * nx + 4;
    assert!(
        (out[centre] - 1.0).abs() < 1e-6,
        "interior preserved, got {}",
        out[centre]
    );
}

#[test]
fn test_label_erosion_metadata_preserved() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.0_f32; n], Shape::new(d)),
        &Default::default(),
    );
    let o = Point::new([1.0, 2.0, 3.0]);
    let s = Spacing::new([0.5, 0.5, 0.5]);
    let r = LabelErosion::new(1)
        .apply(&Image::new(t, o, s, Direction::identity()))
        .unwrap();
    assert_eq!(*r.origin(), o);
    assert_eq!(*r.spacing(), s);
}

// ── LabelOpening tests ─────────────────────────────────────────────────

#[test]
fn test_label_opening_removes_isolated_voxel() {
    let d = [7, 7, 7];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mut v = vec![0.0_f32; n];
    for iz in 2..6 {
        for iy in 2..6 {
            for ix in 2..6 {
                v[iz * ny * nx + iy * nx + ix] = 1.0;
            }
        }
    }
    v[0] = 1.0;
    let out = vv(&LabelOpening::new(1).apply(&img(v, d)).unwrap());
    assert!(
        out[0].abs() < 1e-6,
        "isolated voxel removed by opening, got {}",
        out[0]
    );
}

#[test]
fn test_label_opening_empty_is_identity() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let out = vv(&LabelOpening::new(1)
        .apply(&img(vec![0.0_f32; n], d))
        .unwrap());
    for &x in &out {
        assert!(x.abs() < 1e-6, "empty opening: {x}");
    }
}

#[test]
fn test_label_opening_metadata_preserved() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.0_f32; n], Shape::new(d)),
        &Default::default(),
    );
    let o = Point::new([1.0, 0.0, 0.0]);
    let s = Spacing::new([2.0, 2.0, 2.0]);
    let r = LabelOpening::new(1)
        .apply(&Image::new(t, o, s, Direction::identity()))
        .unwrap();
    assert_eq!(*r.origin(), o);
    assert_eq!(*r.spacing(), s);
}

// ── LabelClosing tests ─────────────────────────────────────────────────

#[test]
fn test_label_closing_fills_background_hole() {
    let d = [7, 7, 7];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mut v = vec![1.0_f32; n];
    let hole = 3 * ny * nx + 3 * nx + 3;
    v[hole] = 0.0;
    let out = vv(&LabelClosing::new(1).apply(&img(v, d)).unwrap());
    assert!(
        (out[hole] - 1.0).abs() < 1e-6,
        "hole should be filled, got {}",
        out[hole]
    );
}

#[test]
fn test_label_closing_empty_is_identity() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let out = vv(&LabelClosing::new(1)
        .apply(&img(vec![0.0_f32; n], d))
        .unwrap());
    for &x in &out {
        assert!(x.abs() < 1e-6, "empty closing: {x}");
    }
}

#[test]
fn test_label_closing_metadata_preserved() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.0_f32; n], Shape::new(d)),
        &Default::default(),
    );
    let o = Point::new([0.0, 1.0, 2.0]);
    let s = Spacing::new([1.5, 1.5, 1.5]);
    let r = LabelClosing::new(1)
        .apply(&Image::new(t, o, s, Direction::identity()))
        .unwrap();
    assert_eq!(*r.origin(), o);
    assert_eq!(*r.spacing(), s);
}

// ── MorphologicalReconstruction tests ─────────────────────────────────

#[test]
fn test_recon_dilation_expands_to_mask() {
    let d = [5, 5, 5];
    let [_, ny, nx] = d;
    let n = d[0] * d[1] * d[2];
    let mask = img(vec![1.0_f32; n], d);
    let mut mv = vec![0.0_f32; n];
    mv[2 * ny * nx + 2 * nx + 2] = 1.0;
    let marker = img(mv, d);
    let out = vv(
        &MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .with_max_iter(500)
            .apply(&marker, &mask)
            .unwrap(),
    );
    let mean: f32 = out.iter().sum::<f32>() / n as f32;
    assert!(
        (mean - 1.0).abs() < 1e-3,
        "dilation should fill mask, mean={mean}"
    );
}

#[test]
fn test_recon_erosion_contracts_to_mask() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let mask = img(vec![0.0_f32; n], d);
    let marker = img(vec![1.0_f32; n], d);
    let out = vv(
        &MorphologicalReconstruction::new(ReconstructionMode::Erosion)
            .apply(&marker, &mask)
            .unwrap(),
    );
    let max_val = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_val < 1e-4,
        "erosion should contract to 0, max={max_val}"
    );
}

#[test]
fn test_recon_marker_equals_mask_converges_immediately() {
    let d = [4, 4, 4];
    let n = d[0] * d[1] * d[2];
    let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let marker = img(vals.clone(), d);
    let mask = img(vals.clone(), d);
    let out = vv(
        &MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .apply(&marker, &mask)
            .unwrap(),
    );
    for (i, (&e, &a)) in vals.iter().zip(out.iter()).enumerate() {
        assert!((a - e).abs() < 1e-4, "identity voxel {i}: {a} != {e}");
    }
}

#[test]
fn test_recon_shape_mismatch_returns_error() {
    let marker = img(vec![0.0_f32; 27], [3, 3, 3]);
    let mask = img(vec![1.0_f32; 64], [4, 4, 4]);
    let result =
        MorphologicalReconstruction::new(ReconstructionMode::Dilation).apply(&marker, &mask);
    assert!(result.is_err(), "shape mismatch must return Err");
}

#[test]
fn test_recon_metadata_preserved() {
    let d = [3, 3, 3];
    let n = d[0] * d[1] * d[2];
    let o = Point::new([1.0, 2.0, 3.0]);
    let s = Spacing::new([0.5, 0.5, 0.5]);
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.5_f32; n], Shape::new(d)),
        &Default::default(),
    );
    let marker = Image::new(t.clone(), o, s, Direction::identity());
    let mask = Image::new(t, o, s, Direction::identity());
    let out = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
        .apply(&marker, &mask)
        .unwrap();
    assert_eq!(*out.origin(), o);
    assert_eq!(*out.spacing(), s);
}
