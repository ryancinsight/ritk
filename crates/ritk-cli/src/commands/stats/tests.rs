use super::*;
use crate::commands::{read_image_native, write_image_native, NativeBackend};
use ritk_image::native::Image as NativeImage;
use ritk_io::ImageFormat;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_statistics::{
    dice_coefficient_native, hausdorff_distance_native,
    image_statistics::native::compute_statistics as compute_native_statistics, psnr_native,
    ssim_native };
use std::path::PathBuf;
use tempfile::tempdir;

/// Build a 4Ã—4Ã—4 image filled with the given constant value.
fn native_image(values: Vec<f32>) -> NativeImage<f32, NativeBackend, 3> {
    NativeImage::from_flat_on(
        values,
        [4, 4, 4],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &NativeBackend::default(),
    )
    .expect("invariant: valid native stats fixture")
}

fn make_constant_image(value: f32) -> NativeImage<f32, NativeBackend, 3> {
    native_image(vec![value; 64])
}

/// Build a 4Ã—4Ã—4 ramp image whose voxel at flat index i has value `i as f32`.
fn make_ramp_image() -> NativeImage<f32, NativeBackend, 3> {
    native_image((0..64).map(|i| i as f32).collect())
}

/// Build a 4Ã—4Ã—4 binary mask with the first `n_foreground` voxels set to
/// 1.0 and the remainder set to 0.0.
fn make_binary_mask(n_foreground: usize) -> NativeImage<f32, NativeBackend, 3> {
    native_image(
        (0..64)
            .map(|i| if i < n_foreground { 1.0 } else { 0.0 })
            .collect(),
    )
}

/// Helper: write a NIfTI image and return the path.
fn write_nifti_tmp(
    dir: &std::path::Path,
    name: &str,
    image: &NativeImage<f32, NativeBackend, 3>,
) -> PathBuf {
    let path = dir.join(name);
    write_image_native(&path, image, ImageFormat::NIfTI).expect("write native NIfTI fixture");
    path
}

// â”€â”€ Positive: summary computes correct statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// For a constant image, min == max == mean == value, std == 0.
#[test]
fn test_stats_summary_constant_image() {
    let dir = tempdir().unwrap();
    let image = make_constant_image(42.0);
    let input = write_nifti_tmp(dir.path(), "const.nii", &image);

    let result = run(StatsArgs {
        input,
        reference: None,
        metric: StatMetric::Summary,
        max_val: 255.0 });
    assert!(result.is_ok(), "summary must succeed: {:?}", result.err());
}

/// Summary on a ramp image must report correct min and max.
#[test]
fn test_stats_summary_ramp_image_values() {
    let image = make_ramp_image();
    let s = compute_native_statistics(&image).expect("native statistics succeeds");

    assert!((s.min - 0.0).abs() < 1e-4, "min must be 0.0, got {}", s.min);
    assert!(
        (s.max - 63.0).abs() < 1e-4,
        "max must be 63.0, got {}",
        s.max
    );

    let expected_mean = (0..64).map(|i| i as f32).sum::<f32>() / 64.0;
    assert!(
        (s.mean - expected_mean).abs() < 1e-3,
        "mean must be {expected_mean}, got {}",
        s.mean
    );
}

// â”€â”€ Positive: dice on identical masks returns 1.0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stats_dice_identical_masks_returns_one() {
    let dir = tempdir().unwrap();
    let mask = make_binary_mask(32);
    let input = write_nifti_tmp(dir.path(), "mask_a.nii", &mask);
    let reference = write_nifti_tmp(dir.path(), "mask_b.nii", &mask);

    let result = run(StatsArgs {
        input: input.clone(),
        reference: Some(reference),
        metric: StatMetric::Dice,
        max_val: 255.0 });
    assert!(result.is_ok(), "dice must succeed: {:?}", result.err());

    // Verify the value directly via the library function.
    let img = read_image_native(&input).expect("read native Dice fixture");
    let value = dice_coefficient_native(&img, &img).expect("native Dice succeeds");
    assert!(
        (value - 1.0).abs() < 1e-5,
        "Dice of identical masks must be 1.0, got {value}"
    );
}

// â”€â”€ Positive: dice on disjoint masks returns 0.0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stats_dice_disjoint_masks_returns_zero() {
    // Mask A: first 32 voxels foreground.
    let a = make_binary_mask(32);

    // Mask B: last 32 voxels foreground.
    let b = native_image((0..64).map(|i| if i >= 32 { 1.0 } else { 0.0 }).collect());

    let value = dice_coefficient_native(&a, &b).expect("native Dice succeeds");
    assert!(
        value.abs() < 1e-5,
        "Dice of disjoint masks must be 0.0, got {value}"
    );
}

// â”€â”€ Positive: psnr on identical images returns infinity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stats_psnr_identical_images_returns_inf() {
    let dir = tempdir().unwrap();
    let image = make_ramp_image();
    let input = write_nifti_tmp(dir.path(), "img_a.nii", &image);
    let reference = write_nifti_tmp(dir.path(), "img_b.nii", &image);

    let result = run(StatsArgs {
        input: input.clone(),
        reference: Some(reference),
        metric: StatMetric::Psnr,
        max_val: 63.0 });
    assert!(result.is_ok(), "psnr must succeed: {:?}", result.err());

    let img = read_image_native(&input).expect("read native PSNR fixture");
    let value = psnr_native(&img, &img, 63.0).expect("native PSNR succeeds");
    assert!(
        value.is_infinite() || value > 100.0,
        "PSNR of identical images must be very large or infinite, got {value}"
    );
}

// â”€â”€ Positive: ssim on identical images returns 1.0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stats_ssim_identical_images_returns_one() {
    let dir = tempdir().unwrap();
    let image = make_ramp_image();
    let input = write_nifti_tmp(dir.path(), "img_a.nii", &image);
    let reference = write_nifti_tmp(dir.path(), "img_b.nii", &image);

    let result = run(StatsArgs {
        input: input.clone(),
        reference: Some(reference),
        metric: StatMetric::Ssim,
        max_val: 63.0 });
    assert!(result.is_ok(), "ssim must succeed: {:?}", result.err());

    let img = read_image_native(&input).expect("read native SSIM fixture");
    let value = ssim_native(&img, &img, 63.0).expect("native SSIM succeeds");
    assert!(
        (value - 1.0).abs() < 1e-4,
        "SSIM of identical images must be 1.0, got {value}"
    );
}

// â”€â”€ Positive: hausdorff on identical masks returns 0.0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stats_hausdorff_identical_masks_returns_zero() {
    let dir = tempdir().unwrap();
    let mask = make_binary_mask(32);
    let input = write_nifti_tmp(dir.path(), "mask_a.nii", &mask);
    let reference = write_nifti_tmp(dir.path(), "mask_b.nii", &mask);

    let result = run(StatsArgs {
        input: input.clone(),
        reference: Some(reference),
        metric: StatMetric::Hausdorff,
        max_val: 255.0 });
    assert!(result.is_ok(), "hausdorff must succeed: {:?}", result.err());

    let img = read_image_native(&input).expect("read native Hausdorff fixture");
    let sp = img.spacing();
    let spacing = [sp[0], sp[1], sp[2]];
    let value = hausdorff_distance_native(&img, &img, &spacing).expect("native Hausdorff succeeds");
    assert!(
        value.abs() < 1e-5,
        "Hausdorff distance of identical masks must be 0.0, got {value}"
    );
}

// â”€â”€ Negative: invalid metric names are rejected by clap at parse time;
//    the `run()` function is exhaustive over `StatMetric` and cannot receive
//    an unknown variant. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// â”€â”€ Negative: comparison metric without --reference returns error â”€â”€â”€â”€â”€

#[test]
fn test_stats_dice_without_reference_returns_error() {
    let dir = tempdir().unwrap();
    let image = make_binary_mask(32);
    let input = write_nifti_tmp(dir.path(), "mask.nii", &image);

    let result = run(StatsArgs {
        input,
        reference: None,
        metric: StatMetric::Dice,
        max_val: 255.0 });
    assert!(result.is_err(), "dice without --reference must return Err");

    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--reference is required"),
        "error must explain the missing argument, got: {msg}"
    );
}

#[test]
fn test_stats_psnr_without_reference_returns_error() {
    let dir = tempdir().unwrap();
    let image = make_ramp_image();
    let input = write_nifti_tmp(dir.path(), "img.nii", &image);

    let result = run(StatsArgs {
        input,
        reference: None,
        metric: StatMetric::Psnr,
        max_val: 255.0 });
    assert!(result.is_err(), "psnr without --reference must return Err");

    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--reference is required"),
        "error must explain the missing argument, got: {msg}"
    );
}

#[test]
fn test_stats_ssim_without_reference_returns_error() {
    let dir = tempdir().unwrap();
    let image = make_ramp_image();
    let input = write_nifti_tmp(dir.path(), "img.nii", &image);

    let result = run(StatsArgs {
        input,
        reference: None,
        metric: StatMetric::Ssim,
        max_val: 255.0 });
    assert!(result.is_err(), "ssim without --reference must return Err");

    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--reference is required"),
        "error must explain the missing argument, got: {msg}"
    );
}

#[test]
fn test_stats_hausdorff_without_reference_returns_error() {
    let dir = tempdir().unwrap();
    let image = make_binary_mask(32);
    let input = write_nifti_tmp(dir.path(), "mask.nii", &image);

    let result = run(StatsArgs {
        input,
        reference: None,
        metric: StatMetric::Hausdorff,
        max_val: 255.0 });
    assert!(
        result.is_err(),
        "hausdorff without --reference must return Err"
    );

    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--reference is required"),
        "error must explain the missing argument, got: {msg}"
    );
}

// â”€â”€ Boundary: missing input file returns error â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stats_missing_input_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("does_not_exist.nii");

    let result = run(StatsArgs {
        input,
        reference: None,
        metric: StatMetric::Summary,
        max_val: 255.0 });
    assert!(result.is_err(), "missing input must yield an error");
}

// â”€â”€ Positive: mean-surface-distance identical masks returns 0.0 â”€â”€â”€â”€â”€â”€

#[test]
fn test_stats_mean_surface_distance_identical_masks_returns_zero() {
    let dir = tempdir().unwrap();
    let mask = make_binary_mask(32);
    let path_a = write_nifti_tmp(dir.path(), "a.nii", &mask);
    let path_b = write_nifti_tmp(dir.path(), "b.nii", &mask);

    let args = StatsArgs {
        input: path_a,
        reference: Some(path_b),
        metric: StatMetric::MeanSurfaceDistance,
        max_val: 255.0 };
    run(args).expect("mean-surface-distance must succeed");
}

// â”€â”€ Positive: noise-estimate on constant image returns without error â”€â”€

#[test]
fn test_stats_noise_estimate_constant_image_returns_zero() {
    let dir = tempdir().unwrap();
    let img = make_constant_image(128.0);
    let path = write_nifti_tmp(dir.path(), "const.nii", &img);

    let args = StatsArgs {
        input: path,
        reference: None,
        metric: StatMetric::NoiseEstimate,
        max_val: 255.0 };
    run(args).expect("noise-estimate must succeed");
}

// â”€â”€ Negative: mean-surface-distance without --reference returns error â”€

#[test]
fn test_stats_mean_surface_distance_without_reference_returns_error() {
    let dir = tempdir().unwrap();
    let mask = make_binary_mask(32);
    let path = write_nifti_tmp(dir.path(), "a.nii", &mask);

    let args = StatsArgs {
        input: path,
        reference: None,
        metric: StatMetric::MeanSurfaceDistance,
        max_val: 255.0 };
    assert!(run(args).is_err(), "must error without --reference");
}
