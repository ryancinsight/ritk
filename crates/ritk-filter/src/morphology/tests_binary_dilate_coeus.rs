//! Differential coverage: `binary_dilate_coeus` must be value-identical to
//! the Burn-generic `BinaryDilateFilter::apply` it mirrors — both call the
//! identical `dilate_binary_3d` core (shared harness in `coeus_support`).

use super::binary_dilate_coeus;
use crate::coeus_support::assert_coeus_matches_burn;
use crate::morphology::BinaryDilateFilter;

fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
    assert_coeus_matches_burn(
        vals,
        dims,
        |img| {
            BinaryDilateFilter::new(radius)
                .apply(img)
                .expect("burn dilate")
        },
        |img, backend| binary_dilate_coeus(img, radius, Default::default(), backend),
    );
}

#[test]
fn matches_burn_radius_zero_identity() {
    check(vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2, 2, 2], 0);
}

#[test]
fn matches_burn_single_seed_radius_one() {
    let dims = [3usize, 3, 3];
    let mut vals = vec![0.0f32; 27];
    vals[13] = 1.0; // centre
    check(vals, dims, 1);
}

#[test]
fn matches_burn_scattered_foreground_radius_one() {
    let dims = [6usize, 5, 4];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
    check(vals, dims, 1);
}

#[test]
fn matches_burn_all_background() {
    check(vec![0.0f32; 8], [2, 2, 2], 1);
}
