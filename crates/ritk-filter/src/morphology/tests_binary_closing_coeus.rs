//! Differential coverage: `binary_closing_coeus` must be value-identical to
//! the Burn-generic `BinaryMorphologicalClosing::apply` it mirrors — both
//! compose the identical `erode ∘ dilate` cores (shared harness in
//! `coeus_support`).

use super::binary_closing_coeus;
use crate::coeus_support::assert_coeus_matches_burn;
use crate::morphology::BinaryMorphologicalClosing;

fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
    assert_coeus_matches_burn(
        vals,
        dims,
        |img| {
            BinaryMorphologicalClosing::new(radius)
                .apply(img)
                .expect("burn closing")
        },
        |img, backend| binary_closing_coeus(img, radius, Default::default(), backend),
    );
}

#[test]
fn matches_burn_fills_small_gap_radius_one() {
    // Two foreground blocks separated by a one-voxel gap along x; closing at
    // r=1 bridges it. 1x1x5 line: fg at x=0,1 and x=3,4, gap at x=2.
    check(vec![1.0, 1.0, 0.0, 1.0, 1.0], [1, 1, 5], 1);
}

#[test]
fn matches_burn_all_foreground_radius_one() {
    check(vec![1.0f32; 27], [3, 3, 3], 1);
}

#[test]
fn matches_burn_scattered_foreground_radius_one() {
    let dims = [6usize, 5, 4];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect();
    check(vals, dims, 1);
}

#[test]
fn matches_burn_all_background() {
    check(vec![0.0f32; 8], [2, 2, 2], 1);
}
