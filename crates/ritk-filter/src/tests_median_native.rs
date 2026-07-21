//! Differential + analytical coverage for the Coeus-native median-filter path.
//!
//! The native wrapper shares the exact `median_3d` host core the Coeus path
//! calls, so the differential assertion is bitwise-exact (shared harness).
//! Analytical oracles pin the constant-preservation and impulse-removal
//! contracts directly on the native path.

use crate::median::MedianFilter;
use crate::native_support::{assert_coeus_matches_coeus, make_native_image, native_vals};

#[test]
fn matches_coeus_radius_one() {
    let dims = [6, 5, 7];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| ((i * 17) % 43) as f32 * 0.3 - 5.0).collect();
    assert_coeus_matches_coeus(
        vals,
        dims,
        |img| MedianFilter::new(1).apply(img).expect("coeus median"),
        |img, _b| MedianFilter::new(1).apply_native(img),
    );
}

/// A constant field is invariant under median filtering (median of identical
/// values is that value).
#[test]
fn constant_field_preserved() {
    let dims = [6, 6, 6];
    let c = 42.0_f32;
    let img = make_native_image(vec![c; dims[0] * dims[1] * dims[2]], dims);
    let out = MedianFilter::new(2)
        .apply_native(&img)
        .expect("infallible: validated precondition");
    for v in native_vals(&out) {
        assert_eq!(v, c, "median of a constant must equal that constant");
    }
}

/// A single salt spike in a constant field is removed by a radius-1 median
/// (1 outlier out of 27 samples cannot be the median).
#[test]
fn impulse_removed() {
    let dims = [8, 8, 8];
    let bg = 10.0_f32;
    let n = dims[0] * dims[1] * dims[2];
    let mut vals = vec![bg; n];
    let spike_idx = 4 * dims[1] * dims[2] + 4 * dims[2] + 4;
    vals[spike_idx] = 1000.0;
    let img = make_native_image(vals, dims);
    let out = MedianFilter::new(1)
        .apply_native(&img)
        .expect("infallible: validated precondition");
    assert_eq!(
        native_vals(&out)[spike_idx],
        bg,
        "median must remove an isolated impulse"
    );
}
