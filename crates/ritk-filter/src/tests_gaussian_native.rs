//! Differential + analytical coverage for the Coeus-native Gaussian path.
//!
//! `GaussianFilter::apply_native` and the Burn `apply` evaluate the *same*
//! separable zero-padded convolution with the *same* per-axis kernels
//! ([`super::axis_kernel`]), but Burn `conv1d` and the host core
//! [`super::convolve_zero_pad_3d`] sum the taps in different orders, so the
//! results agree only to floating-point accumulation rounding — not bitwise.
//!
//! # Tolerance derivation
//!
//! Each output voxel is a dot product of at most `width ≤ max_kernel_width`
//! kernel taps (weights in `[0, 1]`, summing to 1) with input values bounded by
//! `M = ‖I‖∞`. Reassociating a length-`w` sum of well-conditioned (here
//! non-negative) terms perturbs the result by at most `≈ w · ε · M`
//! (ε = `f32::EPSILON`). The separable pass convolves along all three axes;
//! each pass's inputs stay bounded by `M` (convex combination), so the errors
//! add to `≈ 3 · w · ε · M`. With `w ≤ 32` that is `≈ 1.1 × 10⁻⁵ · M`; the
//! assertion uses `10⁻⁴ · (1 + M)`, roughly a 9× safety margin.

use crate::edge::GaussianSigma;
use crate::gaussian::GaussianFilter;
use crate::native_support::{assert_native_matches_burn_approx, make_native_image, native_vals};
use crate::native_support::LegacyBurnBackend;
use coeus_core::SequentialBackend;

/// Backend type parameter for the `GaussianFilter` struct. The native path does
/// not use it (its compute backend is the separate `SequentialBackend`), but the
/// struct is generic over a Burn `Backend`, so a concrete one must be named.
type BurnB = LegacyBurnBackend;

fn sigmas(s: f64) -> Vec<GaussianSigma> {
    vec![GaussianSigma::new(s).expect("positive sigma")]
}

#[test]
fn matches_burn_within_derived_tolerance() {
    let dims = [5usize, 6, 7];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| ((i * 13) % 37) as f32).collect();
    let m_max = vals.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
    let tol = 1e-4 * (1.0 + m_max);

    assert_native_matches_burn_approx(
        vals,
        dims,
        tol,
        |img| GaussianFilter::new(sigmas(1.5)).apply(img),
        |img, backend| GaussianFilter::<BurnB>::new(sigmas(1.5)).apply_native(img, backend),
    );
}

#[test]
fn oracle_constant_field_interior_preserved() {
    // A normalized Gaussian convolved with a constant leaves interior voxels
    // (all taps in-bounds) equal to the constant; only zero-padded boundary
    // voxels darken. Half-width for σ_px = 1 is ⌈3⌉ = 3, so a 9³ volume's
    // centre (4,4,4) has margin 4 ≥ 3 and must be preserved.
    let dims = [9usize, 9, 9];
    let n = dims[0] * dims[1] * dims[2];
    let img = make_native_image(vec![4.0f32; n], dims);
    let out = GaussianFilter::<BurnB>::new(sigmas(1.0))
        .apply_native(&img, &SequentialBackend)
        .expect("native gaussian");
    let v = native_vals(&out);
    let center = 4 * 81 + 4 * 9 + 4;
    assert!(
        (v[center] - 4.0).abs() < 1e-5,
        "interior of a constant field must be preserved, got {}",
        v[center]
    );
    // A boundary voxel loses mass under zero padding → strictly darker.
    assert!(v[0] < 4.0, "zero-padded corner must darken, got {}", v[0]);
}
