//! Differential coverage: `binary_erode_coeus` must be value-identical to
//! the Burn-generic `BinaryErodeFilter::apply` it mirrors — both call the
//! identical `erode_binary_3d` core (shared harness in `coeus_support`).

use super::binary_erode_coeus;
use crate::coeus_support::assert_coeus_matches_burn;
use crate::morphology::BinaryErodeFilter;

fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
    assert_coeus_matches_burn(
        vals,
        dims,
        |img| BinaryErodeFilter::new(radius).apply(img).expect("burn erode"),
        |img, backend| binary_erode_coeus(img, radius, Default::default(), backend),
    );
}

#[test]
fn matches_burn_radius_zero_identity() {
    check(vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2, 2, 2], 0);
}

#[test]
fn matches_burn_all_foreground_radius_one() {
    check(vec![1.0f32; 27], [3, 3, 3], 1);
}

#[test]
fn matches_burn_scattered_foreground_radius_one() {
    let dims = [6usize, 5, 4];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    check(vals, dims, 1);
}

#[test]
fn matches_burn_all_background() {
    check(vec![0.0f32; 8], [2, 2, 2], 1);
}
