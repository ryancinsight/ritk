//! Differential coverage: `distance_transform` must be value-identical
//! to the Burn-generic `DistanceTransformImageFilter::apply` it mirrors —
//! both call the same `euclidean_dt` core (shared harness in `coeus_support`).

use crate::distance::euclidean::{
    DistanceTransformImageFilter, SignedDistanceTransformImageFilter,
};
use crate::distance::DistanceMeasure;
use crate::native_support::assert_coeus_matches_coeus;

fn check(vals: Vec<f32>, dims: [usize; 3]) {
    check_measure(vals, dims, DistanceMeasure::Euclidean);
}

fn check_measure(vals: Vec<f32>, dims: [usize; 3], measure: DistanceMeasure) {
    assert_coeus_matches_coeus(
        vals,
        dims,
        |img| {
            DistanceTransformImageFilter::new()
                .with_measure(measure)
                .apply(img)
                .expect("burn distance transform")
        },
        |img, backend| {
            DistanceTransformImageFilter::new()
                .with_measure(measure)
                .apply_native(img, backend)
        },
    );
}

#[test]
fn matches_coeus_single_foreground_voxel() {
    let dims = [5usize, 5, 5];
    let mut fg = vec![0.0f32; 5 * 5 * 5];
    fg[0] = 1.0;
    check(fg, dims);
}

#[test]
fn matches_coeus_all_foreground() {
    check(vec![1.0f32; 4 * 4 * 4], [4, 4, 4]);
}

#[test]
fn matches_coeus_all_background() {
    check(vec![0.0f32; 3 * 3 * 3], [3, 3, 3]);
}

#[test]
fn matches_coeus_scattered_foreground() {
    let dims = [6usize, 5, 4];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    check(vals, dims);
}

#[test]
fn squared_measure_matches_coeus() {
    let mut values = vec![0.0; 27];
    values[0] = 1.0;
    check_measure(values, [3, 3, 3], DistanceMeasure::Squared);
}

#[test]
fn signed_matches_coeus() {
    assert_coeus_matches_coeus(
        vec![0.0, 1.0, 0.0],
        [1, 1, 3],
        |image| {
            SignedDistanceTransformImageFilter::new()
                .apply(image)
                .expect("burn signed distance transform")
        },
        |image, backend| SignedDistanceTransformImageFilter::new().apply_native(image, backend),
    );
}
