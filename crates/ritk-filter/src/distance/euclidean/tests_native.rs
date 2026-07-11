//! Differential coverage: `distance_transform` must be value-identical
//! to the Burn-generic `DistanceTransformImageFilter::apply` it mirrors —
//! both call the same `euclidean_dt` core (shared harness in `coeus_support`).

use super::{distance_transform, signed_distance_transform};
use crate::distance::euclidean::{
    DistanceTransformImageFilter, SignedDistanceTransformImageFilter,
};
use crate::distance::types::BinarizationThreshold;
use crate::native_support::assert_native_matches_burn;

fn check(vals: Vec<f32>, dims: [usize; 3]) {
    assert_native_matches_burn(
        vals,
        dims,
        |img| {
            DistanceTransformImageFilter::new()
                .apply(img)
                .expect("burn distance transform")
        },
        |img, backend| distance_transform(img, BinarizationThreshold::DEFAULT, backend),
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

#[test]
fn signed_matches_burn_voxel_centre_convention() {
    assert_native_matches_burn(
        vec![0.0, 1.0, 0.0],
        [1, 1, 3],
        |image| {
            SignedDistanceTransformImageFilter::new()
                .apply(image)
                .expect("burn signed distance transform")
        },
        |image, backend| signed_distance_transform(image, BinarizationThreshold::DEFAULT, backend),
    );
}
