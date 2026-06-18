use super::*;
use crate::image_comparison::{dice_coefficient, similarity_index};

#[test]
fn test_dice_identical_masks_is_one() {
    let mask: Image<TestBackend, 3> = make_image(vec![1.0f32; 27], [3, 3, 3]);
    let dice = dice_coefficient(&mask, &mask);
    assert!(
        (dice - 1.0).abs() < F32_TOL,
        "identical masks -> Dice = 1.0, got {}",
        dice
    );
}

#[test]
fn test_dice_disjoint_masks_is_zero() {
    let mut pred = vec![0.0f32; 27];
    for v in pred.iter_mut().take(13) {
        *v = 1.0;
    }
    let mut gt = vec![0.0f32; 27];
    for v in gt.iter_mut().take(27).skip(14) {
        *v = 1.0;
    }
    let pred_img: Image<TestBackend, 3> = make_image(pred, [3, 3, 3]);
    let gt_img: Image<TestBackend, 3> = make_image(gt, [3, 3, 3]);
    let dice = dice_coefficient(&pred_img, &gt_img);
    assert!(
        dice.abs() < F32_TOL,
        "disjoint masks -> Dice = 0.0, got {}",
        dice
    );
}

#[test]
fn test_dice_known_overlap_half() {
    let pred = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let gt = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let pred_img: Image<TestBackend, 1> = make_image(pred, [8]);
    let gt_img: Image<TestBackend, 1> = make_image(gt, [8]);
    let dice = dice_coefficient(&pred_img, &gt_img);
    assert!(
        (dice - 0.5).abs() < F32_TOL,
        "Dice = 2*2/(4+4) = 0.5, got {}",
        dice
    );
}

#[test]
fn test_dice_both_empty_returns_one() {
    let pred: Image<TestBackend, 3> = make_image(vec![0.0; 27], [3, 3, 3]);
    let gt: Image<TestBackend, 3> = make_image(vec![0.0; 27], [3, 3, 3]);
    let dice = dice_coefficient(&pred, &gt);
    assert!(
        (dice - 1.0).abs() < F32_TOL,
        "both empty -> Dice = 1.0, got {}",
        dice
    );
}

#[test]
fn test_dice_2d_known_overlap() {
    let mut pred = vec![0.0f32; 16];
    let mut gt = vec![0.0f32; 16];
    pred[0] = 1.0;
    pred[1] = 1.0;
    gt[1] = 1.0;
    gt[2] = 1.0;
    pred[4] = 1.0;
    pred[5] = 1.0;
    gt[5] = 1.0;
    gt[6] = 1.0;

    let pred_img: Image<TestBackend, 2> = make_image(pred, [4, 4]);
    let gt_img: Image<TestBackend, 2> = make_image(gt, [4, 4]);
    let dice = dice_coefficient(&pred_img, &gt_img);
    assert!((dice - 0.5).abs() < F32_TOL, "2D Dice = 0.5, got {}", dice);
}

#[test]
fn test_dice_symmetry() {
    let pred: Image<TestBackend, 1> = make_image(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [8]);
    let gt: Image<TestBackend, 1> = make_image(vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [8]);
    let d_pg = dice_coefficient(&pred, &gt);
    let d_gp = dice_coefficient(&gt, &pred);
    assert!(
        (d_pg - d_gp).abs() < 1e-6,
        "Dice is not symmetric: {} vs {}",
        d_pg,
        d_gp
    );
}

#[test]
fn test_dice_one_empty_one_nonempty_is_zero() {
    let pred: Image<TestBackend, 1> = make_image(vec![0.0; 8], [8]);
    let gt: Image<TestBackend, 1> = make_image(vec![1.0; 8], [8]);
    let dice = dice_coefficient(&pred, &gt);
    assert!(
        dice.abs() < F32_TOL,
        "one empty -> Dice = 0.0, got {}",
        dice
    );
}

#[test]
fn test_similarity_index_binarizes_multivalued_labels() {
    // a nonzero = {0,1,2,4,5}? indices: a=[0,1,1,0,2,2], b=[0,1,0,1,2,0].
    // A=4 (idx1,2,4,5), B=3 (idx1,3,4), A∩B={idx1,idx4}=2 -> 2*2/(4+3)=4/7.
    let a: Image<TestBackend, 1> = make_image(vec![0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [6]);
    let b: Image<TestBackend, 1> = make_image(vec![0.0, 1.0, 0.0, 1.0, 2.0, 0.0], [6]);
    let si = similarity_index(&a, &b);
    let expected = 4.0f32 / 7.0;
    assert!(
        (si - expected).abs() < F32_TOL,
        "similarity index = 4/7, got {}",
        si
    );
}

#[test]
fn test_similarity_index_both_empty_is_zero() {
    // Distinct from dice_coefficient (which returns 1.0) — ITK convention is 0.0.
    let z: Image<TestBackend, 3> = make_image(vec![0.0; 27], [3, 3, 3]);
    assert_eq!(similarity_index(&z, &z), 0.0);
}
