use super::make_mask_1d;
use crate::image_comparison::{psnr, ssim};

#[test]
fn test_psnr_identical_images_is_infinity() {
    let img = make_mask_1d(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = psnr(&img, &img, 255.0);
    assert!(
        result.is_infinite() && result > 0.0,
        "identical images -> PSNR = +inf, got {}",
        result
    );
}

#[test]
fn test_psnr_known_value() {
    let img = make_mask_1d(vec![0.0, 0.0]);
    let reference = make_mask_1d(vec![0.1, 0.1]);
    let result = psnr(&img, &reference, 1.0);
    assert!(
        (result - 20.0).abs() < 1e-3,
        "expected PSNR ~= 20.0 dB, got {}",
        result
    );
}

#[test]
fn test_psnr_symmetry() {
    let a = make_mask_1d(vec![0.0, 1.0, 2.0, 3.0]);
    let b = make_mask_1d(vec![0.5, 1.5, 2.5, 3.5]);
    let psnr_ab = psnr(&a, &b, 10.0);
    let psnr_ba = psnr(&b, &a, 10.0);
    assert!(
        (psnr_ab - psnr_ba).abs() < super::F32_TOL,
        "PSNR must be symmetric: {} vs {}",
        psnr_ab,
        psnr_ba
    );
}

#[test]
fn test_psnr_larger_error_lower_value() {
    let img = make_mask_1d(vec![0.0; 4]);
    let small_err = make_mask_1d(vec![0.1; 4]);
    let large_err = make_mask_1d(vec![1.0; 4]);
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
    let img = make_mask_1d(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = ssim(&img, &img, 5.0);
    assert!(
        (result - 1.0).abs() < super::F32_TOL,
        "identical images -> SSIM = 1.0, got {}",
        result
    );
}

#[test]
fn test_ssim_negated_image_is_low() {
    let img = make_mask_1d(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    let neg = make_mask_1d(vec![1.0, 0.5, 0.0, -0.5, -1.0]);
    let result = ssim(&img, &neg, 1.0);
    assert!(result < 0.0, "negated image -> SSIM < 0, got {}", result);
}

#[test]
fn test_ssim_symmetry() {
    let a = make_mask_1d(vec![1.0, 3.0, 5.0, 7.0]);
    let b = make_mask_1d(vec![2.0, 4.0, 6.0, 8.0]);
    let ssim_ab = ssim(&a, &b, 10.0);
    let ssim_ba = ssim(&b, &a, 10.0);
    assert!(
        (ssim_ab - ssim_ba).abs() < super::F32_TOL,
        "SSIM must be symmetric: {} vs {}",
        ssim_ab,
        ssim_ba
    );
}

#[test]
fn test_ssim_constant_identical_is_one() {
    let img = make_mask_1d(vec![42.0; 10]);
    let result = ssim(&img, &img, 255.0);
    assert!(
        (result - 1.0).abs() < super::F32_TOL,
        "identical constant images -> SSIM = 1.0, got {}",
        result
    );
}
