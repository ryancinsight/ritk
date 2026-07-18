use super::*;
use crate::image_comparison::{pearson_correlation, psnr, psnr_native, ssim, ssim_native};
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;

#[test]
fn pearson_correlation_parallel_affine_contracts() {
    let image: Vec<f32> = (0..16_384).map(|index| index as f32).collect();
    let correlated: Vec<f32> = image.iter().map(|value| value.mul_add(3.0, 7.0)).collect();
    let anticorrelated: Vec<f32> = image.iter().map(|value| 7.0 - 3.0 * value).collect();

    let positive = pearson_correlation(&image, &correlated).expect("valid equal-length inputs");
    let negative = pearson_correlation(&image, &anticorrelated).expect("valid equal-length inputs");

    // Affine transforms with positive/negative slope have Pearson r = +/-1.
    // f64 accumulation over 2^14 exactly represented integer-valued f32 inputs
    // bounds the observed rounding error below 32 * f64::EPSILON.
    let tolerance = 32.0 * f64::EPSILON;
    assert!((positive - 1.0).abs() <= tolerance, "r = {positive}");
    assert!((negative + 1.0).abs() <= tolerance, "r = {negative}");
}

#[test]
fn pearson_correlation_rejects_invalid_shapes() {
    let mismatch = pearson_correlation(&[1.0, 2.0], &[1.0]);
    assert_eq!(
        mismatch
            .expect_err("mismatched lengths must fail")
            .to_string(),
        "pearson correlation requires equal element counts: 2 != 1"
    );
    let empty = pearson_correlation(&[], &[]);
    assert_eq!(
        empty.expect_err("empty inputs must fail").to_string(),
        "pearson correlation requires at least one element"
    );
}

#[test]
fn pearson_correlation_constant_inputs_are_zero() {
    let value = pearson_correlation(&[4.0; 32], &[4.0; 32]).expect("valid constant inputs");
    assert_eq!(value, 0.0);
}

#[test]
fn test_psnr_identical_images_is_infinity() {
    let img: Image<f32, TestBackend, 1> = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    let result = psnr(&img, &img, 255.0);
    assert!(
        result.is_infinite() && result > 0.0,
        "identical images -> PSNR = +inf, got {}",
        result
    );
}

#[test]
fn native_psnr_known_value_matches_formula() {
    let image = NativeImage::from_flat_on(
        vec![0.0, 0.0],
        [1, 1, 2],
        ritk_spatial::Point::new([0.0; 3]),
        ritk_spatial::Spacing::new([1.0; 3]),
        ritk_spatial::Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let reference = NativeImage::from_flat_on(
        vec![0.1, 0.1],
        [1, 1, 2],
        ritk_spatial::Point::new([0.0; 3]),
        ritk_spatial::Spacing::new([1.0; 3]),
        ritk_spatial::Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native reference");
    let value = psnr_native(&image, &reference, 1.0).expect("native psnr succeeds");
    assert!((value - 20.0).abs() < 1e-3);
}

#[test]
fn test_psnr_known_value() {
    let img: Image<f32, TestBackend, 1> = make_image(vec![0.0, 0.0], [2]);
    let reference: Image<f32, TestBackend, 1> = make_image(vec![0.1, 0.1], [2]);
    let result = psnr(&img, &reference, 1.0);
    assert!(
        (result - 20.0).abs() < 1e-3,
        "expected PSNR ~= 20.0 dB, got {}",
        result
    );
}

#[test]
fn test_psnr_symmetry() {
    let a: Image<f32, TestBackend, 1> = make_image(vec![0.0, 1.0, 2.0, 3.0], [4]);
    let b: Image<f32, TestBackend, 1> = make_image(vec![0.5, 1.5, 2.5, 3.5], [4]);
    let psnr_ab = psnr(&a, &b, 10.0);
    let psnr_ba = psnr(&b, &a, 10.0);
    assert!(
        (psnr_ab - psnr_ba).abs() < F32_TOL,
        "PSNR must be symmetric: {} vs {}",
        psnr_ab,
        psnr_ba
    );
}

#[test]
fn test_psnr_larger_error_lower_value() {
    let img: Image<f32, TestBackend, 1> = make_image(vec![0.0; 4], [4]);
    let small_err: Image<f32, TestBackend, 1> = make_image(vec![0.1; 4], [4]);
    let large_err: Image<f32, TestBackend, 1> = make_image(vec![1.0; 4], [4]);
    let psnr_small = psnr(&img, &small_err, 1.0);
    let psnr_large = psnr(&img, &large_err, 1.0);
    assert!(
        psnr_small > psnr_large,
        "smaller error -> higher PSNR: {} should be > {}",
        psnr_small,
        psnr_large
    );
}

#[test]
fn test_ssim_identical_images_is_one() {
    let img: Image<f32, TestBackend, 1> = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    let result = ssim(&img, &img, 5.0);
    assert!(
        (result - 1.0).abs() < F32_TOL,
        "identical images -> SSIM = 1.0, got {}",
        result
    );
}

#[test]
fn native_ssim_identical_images_is_one() {
    let image = NativeImage::from_flat_on(
        vec![1.0, 2.0, 3.0],
        [1, 1, 3],
        ritk_spatial::Point::new([0.0; 3]),
        ritk_spatial::Spacing::new([1.0; 3]),
        ritk_spatial::Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let value = ssim_native(&image, &image, 3.0).expect("native ssim succeeds");
    assert!((value - 1.0).abs() < F32_TOL);
}

#[test]
fn test_ssim_negated_image_is_low() {
    let img: Image<f32, TestBackend, 1> = make_image(vec![-1.0, -0.5, 0.0, 0.5, 1.0], [5]);
    let neg: Image<f32, TestBackend, 1> = make_image(vec![1.0, 0.5, 0.0, -0.5, -1.0], [5]);
    let result = ssim(&img, &neg, 1.0);
    assert!(result < 0.0, "negated image -> SSIM < 0, got {}", result);
}

#[test]
fn test_ssim_symmetry() {
    let a: Image<f32, TestBackend, 1> = make_image(vec![1.0, 3.0, 5.0, 7.0], [4]);
    let b: Image<f32, TestBackend, 1> = make_image(vec![2.0, 4.0, 6.0, 8.0], [4]);
    let ssim_ab = ssim(&a, &b, 10.0);
    let ssim_ba = ssim(&b, &a, 10.0);
    assert!(
        (ssim_ab - ssim_ba).abs() < F32_TOL,
        "SSIM must be symmetric: {} vs {}",
        ssim_ab,
        ssim_ba
    );
}

#[test]
fn test_ssim_constant_identical_is_one() {
    let img: Image<f32, TestBackend, 1> = make_image(vec![42.0; 10], [10]);
    let result = ssim(&img, &img, 255.0);
    assert!(
        (result - 1.0).abs() < F32_TOL,
        "identical constant images -> SSIM = 1.0, got {}",
        result
    );
}
