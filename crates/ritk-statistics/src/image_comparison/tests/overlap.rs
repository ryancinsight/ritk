use super::{make_mask_1d, make_mask_2d, make_mask_3d};
use crate::image_comparison::dice_coefficient;

#[test]
fn test_dice_identical_masks_is_one() {
    let mask = make_mask_3d(vec![1.0f32; 27], [3, 3, 3]);
    let dice = dice_coefficient(&mask, &mask);
    assert!(
        (dice - 1.0).abs() < super::F32_TOL,
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
    let pred_img = make_mask_3d(pred, [3, 3, 3]);
    let gt_img = make_mask_3d(gt, [3, 3, 3]);
    let dice = dice_coefficient(&pred_img, &gt_img);
    assert!(
        dice.abs() < super::F32_TOL,
        "disjoint masks -> Dice = 0.0, got {}",
        dice
    );
}

#[test]
fn test_dice_known_overlap_half() {
    let pred = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let gt = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let pred_img = make_mask_1d(pred);
    let gt_img = make_mask_1d(gt);
    let dice = dice_coefficient(&pred_img, &gt_img);
    assert!(
        (dice - 0.5).abs() < super::F32_TOL,
        "Dice = 2*2/(4+4) = 0.5, got {}",
        dice
    );
}

#[test]
fn test_dice_both_empty_returns_one() {
    let pred = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
    let gt = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
    let dice = dice_coefficient(&pred, &gt);
    assert!(
        (dice - 1.0).abs() < super::F32_TOL,
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

    let pred_img = make_mask_2d(pred, [4, 4]);
    let gt_img = make_mask_2d(gt, [4, 4]);
    let dice = dice_coefficient(&pred_img, &gt_img);
    assert!(
        (dice - 0.5).abs() < super::F32_TOL,
        "2D Dice = 0.5, got {}",
        dice
    );
}

#[test]
fn test_dice_symmetry() {
    let pred = make_mask_1d(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let gt = make_mask_1d(vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
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
    let pred = make_mask_1d(vec![0.0; 8]);
    let gt = make_mask_1d(vec![1.0; 8]);
    let dice = dice_coefficient(&pred, &gt);
    assert!(
        dice.abs() < super::F32_TOL,
        "one empty -> Dice = 0.0, got {}",
        dice
    );
}
