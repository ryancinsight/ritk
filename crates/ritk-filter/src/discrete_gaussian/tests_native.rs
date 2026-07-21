//! Differential + analytical coverage for the Coeus-native discrete-Gaussian path.
//!
//! The native wrapper shares the exact kernel builder (`kernels_for_spacing`)
//! and substrate-agnostic `convolve_separable` host core the Coeus path calls,
//! so the differential assertion is bitwise-exact (shared harness). The
//! analytical oracle pins the uniform-image identity invariant directly on the
//! native path: replicate-padded normalized convolution of a constant field is
//! that field, at every voxel including boundaries.

use super::DiscreteGaussianFilter;
use crate::native_support::{assert_coeus_matches_coeus, make_native_image, native_vals};

type CoeusB = coeus_core::SequentialBackend;

fn filter() -> DiscreteGaussianFilter<CoeusB> {
    DiscreteGaussianFilter::<CoeusB>::new_isotropic(2.0)
}

#[test]
fn matches_coeus() {
    let dims = [6, 7, 5];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - (i % 5) as f32).collect();
    assert_coeus_matches_coeus(
        vals,
        dims,
        |img| filter().apply(img),
        |img, _b| filter().apply_native(img),
    );
}

/// conv(constant, normalized_kernel) = constant for all voxels, including
/// boundaries (replicate padding preserves the uniform-image identity).
#[test]
fn constant_field_preserved() {
    let dims = [8, 8, 8];
    let c = 12.25_f32;
    let img = make_native_image(vec![c; dims[0] * dims[1] * dims[2]], dims);
    let out = filter()
        .apply_native(&img)
        .expect("infallible: validated precondition");
    for v in native_vals(&out) {
        assert!(
            (v - c).abs() < 1e-4,
            "discrete Gaussian must preserve a constant field, got {v}"
        );
    }
}
