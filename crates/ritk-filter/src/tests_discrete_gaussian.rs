use super::*;
use crate::edge::GaussianSigma;
use ritk_core::image::Image;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};
type B = burn_ndarray::NdArray<f32>;

fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, shape)
}
fn make_image_with_spacing(vals: Vec<f32>, shape: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    ts::make_image_with_spacing::<B, 3>(vals, shape, spacing)
}
fn vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data().clone().into_data().into_vec::<f32>().unwrap()
}

// Helper: near-zero sigma that lies below the 1e-9 skip threshold.
// sigma = 1e-100 → variance ≈ 0 → pixel_sigma << 1e-9 → no kernel built (identity).
const NEAR_ZERO_SIGMA: f64 = 1e-100;

#[test]
fn test_uniform_image_is_unchanged_by_gaussian() {
    let img = make_image(vec![7.0_f32; 125], [5, 5, 5]);
    let out = DiscreteGaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1.0)]).apply(&img);
    for &x in &vals(&out) {
        assert!((x - 7.0).abs() < 1e-4);
    }
}

#[test]
fn test_output_shape_matches_input() {
    let img = make_image(vec![1.0_f32; 216], [6, 6, 6]);
    let out = DiscreteGaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(2.0)]).apply(&img);
    assert_eq!(out.shape(), img.shape());
}

#[test]
fn test_larger_variance_produces_more_smoothing_on_step_edge() {
    let mut v: Vec<f32> = vec![0.0; 8];
    v.extend(vec![100.0; 8]);
    let img = make_image(v, [1, 1, 16]);
    let sv = vals(
        &DiscreteGaussianFilter::<B>::new(vec![
            GaussianSigma::new_unchecked(0.5),
            GaussianSigma::new_unchecked(0.5),
            GaussianSigma::new_unchecked(0.5),
        ])
        .with_spacing_mode(SpacingMode::Voxel)
        .apply(&img),
    );
    let lv = vals(
        &DiscreteGaussianFilter::<B>::new(vec![
            GaussianSigma::new_unchecked(0.5),
            GaussianSigma::new_unchecked(0.5),
            GaussianSigma::new_unchecked(4.0),
        ])
        .with_spacing_mode(SpacingMode::Voxel)
        .apply(&img),
    );
    assert!((50.0 - lv[8]).abs() < (50.0 - sv[8]).abs());
}

#[test]
fn test_use_image_spacing_accounts_for_spacing() {
    let mut v: Vec<f32> = vec![0.0; 8];
    v.extend(vec![100.0; 8]);
    let img_a = make_image_with_spacing(v.clone(), [1, 1, 16], [1.0, 1.0, 1.0]);
    let img_b = make_image_with_spacing(v.clone(), [1, 1, 16], [1.0, 1.0, 2.0]);
    let f = DiscreteGaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(4.0)]);
    let a8 = vals(&f.apply(&img_a))[8];
    let b8 = vals(&f.apply(&img_b))[8];
    assert!((100.0 - a8).abs() > (100.0 - b8).abs());
}

#[test]
fn test_zero_variance_produces_identity() {
    let v: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_image(v.clone(), [3, 3, 3]);
    // zero variance = sigma = 0 → use new_isotropic(0.0) which accepts variance directly
    let out = DiscreteGaussianFilter::<B>::new_isotropic(0.0)
        .with_spacing_mode(SpacingMode::Voxel)
        .apply(&img);
    for (&e, &a) in v.iter().zip(vals(&out).iter()) {
        assert!((e - a).abs() < 1e-4);
    }
}

#[test]
fn test_spatial_metadata_preserved() {
    let dev: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3])),
        &dev,
    );
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let dir = Direction::identity();
    let img = Image::new(t, origin, spacing, dir);
    let out = DiscreteGaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1.0)]).apply(&img);
    assert_eq!(out.origin(), &origin);
    assert_eq!(out.spacing(), &spacing);
    assert_eq!(out.direction(), &dir);
}

#[test]
fn test_maximum_error_smaller_produces_larger_kernel() {
    let mut v: Vec<f32> = vec![0.0; 8];
    v.extend(vec![100.0; 8]);
    let img = make_image(v, [1, 1, 16]);
    // sigma=2.0 → variance=4.0 (same as original var=4.0); near-zero sigmas for unused dims
    let loose = vals(
        &DiscreteGaussianFilter::<B>::new(vec![
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(2.0),
        ])
        .with_maximum_error(0.1)
        .with_spacing_mode(SpacingMode::Voxel)
        .apply(&img),
    );
    let strict = vals(
        &DiscreteGaussianFilter::<B>::new(vec![
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(2.0),
        ])
        .with_maximum_error(0.001)
        .with_spacing_mode(SpacingMode::Voxel)
        .apply(&img),
    );
    assert!((50.0 - strict[8]).abs() <= (50.0 - loose[8]).abs() + 1.0);
}

#[test]
fn test_per_dimension_variance_applied_independently() {
    let mut v = vec![0.0_f32; 64];
    v[4 * 8 + 4] = 100.0;
    let img = make_image(v, [1, 8, 8]);
    // sigma=2.0 in dim2 (x) only; near-zero sigma in dim1 (y) → no y-smoothing
    let ov = vals(
        &DiscreteGaussianFilter::<B>::new(vec![
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(2.0),
        ])
        .with_spacing_mode(SpacingMode::Voxel)
        .apply(&img),
    );
    assert!(ov[4 * 8 + 3] > 1.0);
    assert!(ov[4 * 8 + 5] > 1.0);
    assert!(ov[3 * 8 + 4] < 1.0);
    assert!(ov[5 * 8 + 4] < 1.0);
}

#[test]
fn test_impulse_response_matches_discrete_gaussian() {
    // The impulse response of the separable filter IS its kernel. ITK's discrete
    // Gaussian kernel is g[d] = e^{-t}·I_|d|(t), t = pixel variance (Lindeberg's
    // discrete analog of the Gaussian) — NOT a sampled continuous Gaussian. The
    // untruncated form sums to 1 exactly (Σ_n I_n(t) = e^t); the filter truncates
    // at maximum_error = 0.01 and renormalises, so each tap exceeds the untruncated
    // value by the redistributed tail mass (< 1.2 %).
    let n = 31usize;
    let c = 15usize;
    let t = 4.0f64; // pixel variance (Voxel mode, sigma = 2 ⇒ t = sigma² = 4)
    let sigma = t.sqrt();
    let mut imp = vec![0.0_f32; n];
    imp[c] = 1.0;
    let img = make_image(imp, [1, 1, n]);
    let ov = vals(
        &DiscreteGaussianFilter::<B>::new(vec![
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(NEAR_ZERO_SIGMA),
            GaussianSigma::new_unchecked(sigma),
        ])
        .with_spacing_mode(SpacingMode::Voxel)
        .apply(&img),
    );
    // Reference kernel from the ITK spec (independent of build_kernel's code):
    // accumulate g[i] = e^{-t}·I_i(t) until the mass reaches 1 - max_error, then
    // normalise by that truncated sum.
    let et = (-t).exp();
    let mut coeff = vec![et * super::modified_bessel_i(0, t)];
    let mut s = coeff[0];
    let mut i = 1usize;
    loop {
        let cval = et * super::modified_bessel_i(i, t);
        coeff.push(cval);
        s += 2.0 * cval;
        if s >= 1.0 - 0.01 || i >= 32 {
            break;
        }
        i += 1;
    }
    let radius = coeff.len() - 1;

    let mut total = 0.0f64;
    for (k, &ovk) in ov.iter().enumerate().take(n) {
        let d = (k as i64 - c as i64).unsigned_abs() as usize;
        let expected = if d <= radius { coeff[d] / s } else { 0.0 };
        total += ovk as f64;
        assert!(
            (ovk as f64 - expected).abs() < 1e-6,
            "tap {k}: filter {} vs ITK discrete Gaussian {expected}",
            ovk
        );
    }
    assert!(
        (total - 1.0).abs() < 1e-5,
        "impulse response must sum to 1: {total}"
    );
}

#[test]
#[should_panic]
fn test_empty_variance_panics() {
    let _ = DiscreteGaussianFilter::<B>::new(vec![]);
}
#[test]
#[should_panic]
fn test_maximum_error_zero_panics() {
    let _ = DiscreteGaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1.0)])
        .with_maximum_error(0.0);
}
#[test]
#[should_panic]
fn test_maximum_error_one_panics() {
    let _ = DiscreteGaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1.0)])
        .with_maximum_error(1.0);
}
