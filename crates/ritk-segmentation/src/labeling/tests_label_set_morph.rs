//! Differential tests for [`label_set_morph`] against SimpleITK reference output.
//!
//! Every expected vector is the exact `sitk.LabelSetDilate` / `sitk.LabelSetErode`
//! result (uint8 labels), captured verbatim — a genuine external oracle, not a
//! ritk self-comparison.

use super::{label_set_morph, LabelSetMorphOp};
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec_infallible;

type B = burn_ndarray::NdArray<f32>;

/// z=1 label image from a flat `rows×cols` slice.
fn z1(flat: Vec<f32>, rows: usize, cols: usize) -> ritk_image::Image<B, 3> {
    assert_eq!(flat.len(), rows * cols);
    ts::burn_compat::make_image::<B, 3>(flat, [1, rows, cols])
}

fn run(img: &ritk_image::Image<B, 3>, r: f64, use_spacing: bool, op: LabelSetMorphOp) -> Vec<f32> {
    let out = label_set_morph(img, [r, r, 0.0], use_spacing, op);
    extract_vec_infallible(&out).0
}

fn f(v: &[i32]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// 7×7 multi-label scene (labels 1, 2, 3).
fn scene() -> ritk_image::Image<B, 3> {
    let mut a = vec![0i32; 49];
    a[8] = 1;
    a[9] = 1;
    a[32] = 2;
    a[40] = 2;
    a[22] = 3;
    z1(f(&a), 7, 7)
}

#[test]
fn dilate_spacing_r2_matches_sitk() {
    let expect = f(&[
        1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 3, 3,
        3, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2,
    ]);
    assert_eq!(run(&scene(), 2.0, true, LabelSetMorphOp::Dilate), expect);
}

#[test]
fn dilate_voxel_r1_matches_sitk() {
    let expect = f(&[
        1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 3, 3,
        3, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2,
    ]);
    assert_eq!(run(&scene(), 1.0, false, LabelSetMorphOp::Dilate), expect);
}

#[test]
fn dilate_voxel_r2_matches_sitk() {
    let expect = f(&[
        1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 0, 3, 3, 3, 2, 2, 2, 2, 3, 3,
        3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2,
    ]);
    assert_eq!(run(&scene(), 2.0, false, LabelSetMorphOp::Dilate), expect);
}

#[test]
fn dilate_spacing_r1_is_near_identity() {
    // with spacing, r=1 reach is k<1: single-pixel labels are unchanged.
    let expect = f(&[
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    assert_eq!(run(&scene(), 1.0, true, LabelSetMorphOp::Dilate), expect);
}

/// 9×9 solid 5×5 block of label 4 (rows/cols 2..7).
fn block() -> ritk_image::Image<B, 3> {
    let mut b = vec![0i32; 81];
    for r in 2..7 {
        for c in 2..7 {
            b[r * 9 + c] = 4;
        }
    }
    z1(f(&b), 9, 9)
}

#[test]
fn erode_spacing_r1_keeps_full_block() {
    // spacing r=1: boundary not eroded (k<1), block unchanged.
    let mut e = vec![0i32; 81];
    for r in 2..7 {
        for c in 2..7 {
            e[r * 9 + c] = 4;
        }
    }
    assert_eq!(run(&block(), 1.0, true, LabelSetMorphOp::Erode), f(&e));
}

#[test]
fn erode_voxel_r1_matches_sitk() {
    // 3×3 interior survives.
    let mut e = vec![0i32; 81];
    for r in 3..6 {
        for c in 3..6 {
            e[r * 9 + c] = 4;
        }
    }
    assert_eq!(run(&block(), 1.0, false, LabelSetMorphOp::Erode), f(&e));
}

#[test]
fn erode_voxel_r2_matches_sitk() {
    // single center voxel survives.
    let mut e = vec![0i32; 81];
    e[4 * 9 + 4] = 4;
    assert_eq!(run(&block(), 2.0, false, LabelSetMorphOp::Erode), f(&e));
}

#[test]
fn erode_spacing_r2_matches_sitk() {
    // 3×3 cross-ish region survives (Euclidean ball, spacing): rows 3..6, col 3..6.
    let expect = f(&[
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        4, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    assert_eq!(run(&block(), 2.0, true, LabelSetMorphOp::Erode), expect);
}
