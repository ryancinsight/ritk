//! Differential coverage: `distance_transform_coeus` must be value-identical
//! to the Burn-generic `DistanceTransformImageFilter::apply` it mirrors —
//! both call the same `euclidean_dt` core (shared harness in `coeus_support`).

use super::distance_transform_coeus;
use crate::coeus_support::assert_coeus_matches_burn;
use crate::distance::euclidean::DistanceTransformImageFilter;
use crate::distance::types::BinarizationThreshold;

fn check(vals: Vec<f32>, dims: [usize; 3]) {
    assert_coeus_matches_burn(
        vals,
        dims,
        |img| {
            DistanceTransformImageFilter::new()
                .apply(img)
                .expect("burn distance transform")
        },
        |img, backend| distance_transform_coeus(img, BinarizationThreshold::DEFAULT, backend),
    );
}

#[test]
fn matches_burn_single_foreground_voxel() {
    let dims = [5usize, 5, 5];
    let mut fg = vec![0.0f32; 5 * 5 * 5];
    fg[0] = 1.0;
    check(fg, dims);
}

#[test]
fn matches_burn_all_foreground() {
    check(vec![1.0f32; 4 * 4 * 4], [4, 4, 4]);
}

#[test]
fn matches_burn_all_background() {
    check(vec![0.0f32; 3 * 3 * 3], [3, 3, 3]);
}

#[test]
fn matches_burn_scattered_foreground() {
    let dims = [6usize, 5, 4];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    check(vals, dims);
}
