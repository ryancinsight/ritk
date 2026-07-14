//! Tests for the `normalize` command.
use super::*;
use crate::commands::NativeBackend;
use ritk_image::native::Image as NativeImage;
use ritk_io::ImageFormat;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::{Path, PathBuf};

// ── Helper: build a 4×4×4 ramp NIfTI image (voxel i = i as f32) ──────────

fn write_ramp_image(path: &Path) {
    let vals: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let backend = NativeBackend::default();
    let image = NativeImage::from_flat_on(
        vals,
        [4, 4, 4],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &backend,
    )
    .expect("invariant: valid native ramp image");
    write_image_native(path, &image, ImageFormat::NIfTI).expect("write native ramp image");
}

fn default_args(method: NormalizeMethod, input: PathBuf, output: PathBuf) -> NormalizeArgs {
    NormalizeArgs {
        input,
        output,
        method,
        reference: None,
        num_bins: 256,
        contrast: None,
        ws_width: None,
        mask: None,
    }
}

// ── zscore ────────────────────────────────────────────────────────────────

#[test]
fn test_normalize_zscore_creates_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    let args = default_args(NormalizeMethod::Zscore, input, output.clone());
    run(args).unwrap();
    assert!(output.exists());
}

#[test]
fn test_normalize_zscore_output_has_near_zero_mean() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    run(default_args(NormalizeMethod::Zscore, input, output.clone())).unwrap();
    let im = read_image_native(&output).expect("read native z-score output");
    let vals = im.data_slice().expect("contiguous native z-score output");
    let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    assert!(mean.abs() < 1e-4, "zscore mean must be ≈0, got {mean}");
}

// ── minmax ────────────────────────────────────────────────────────────────

#[test]
fn test_normalize_minmax_output_in_zero_one() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    run(default_args(NormalizeMethod::Minmax, input, output.clone())).unwrap();
    let im = read_image_native(&output).expect("read native minmax output");
    let vals = im.data_slice().expect("contiguous native minmax output");
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(min >= -1e-5, "minmax output min must be >= 0, got {min}");
    assert!(
        max <= 1.0 + 1e-5,
        "minmax output max must be <= 1, got {max}"
    );
}

// ── histogram-match ───────────────────────────────────────────────────────

#[test]
fn test_normalize_histogram_match_creates_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let reference = dir.path().join("ref.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    write_ramp_image(&reference);
    let args = NormalizeArgs {
        reference: Some(reference),
        ..default_args(NormalizeMethod::HistogramMatch, input, output.clone())
    };
    run(args).unwrap();
    assert!(output.exists());
}

#[test]
fn test_normalize_histogram_match_without_reference_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    let args = default_args(NormalizeMethod::HistogramMatch, input, output);
    let result = run(args);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("reference"),
        "error must mention 'reference', got: {msg}"
    );
}

// ── nyul ──────────────────────────────────────────────────────────────────

#[test]
fn test_normalize_nyul_creates_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    run(default_args(NormalizeMethod::Nyul, input, output.clone())).unwrap();
    assert!(output.exists());
}

#[test]
fn test_normalize_nyul_with_reference_creates_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let reference = dir.path().join("ref.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    write_ramp_image(&reference);
    let args = NormalizeArgs {
        reference: Some(reference),
        ..default_args(NormalizeMethod::Nyul, input, output.clone())
    };
    run(args).unwrap();
    assert!(output.exists());
}

// ── error cases ───────────────────────────────────────────────────────────

// ── zscore masked ─────────────────────────────────────────────────────────

fn write_half_mask_image(path: &Path) {
    // 4×4×4 binary mask: voxels [0..32) = 1.0, voxels [32..64) = 0.0.
    // The ramp image has values 0..63 in the same layout, so the masked
    // region covers ramp values 0..31 with μ = 15.5.
    let mut vals = vec![0.0f32; 64];
    for v in vals[..32].iter_mut() {
        *v = 1.0;
    }
    let backend = NativeBackend::default();
    let image = NativeImage::from_flat_on(
        vals,
        [4, 4, 4],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &backend,
    )
    .expect("invariant: valid native mask image");
    write_image_native(path, &image, ImageFormat::NIfTI).expect("write native mask image");
}

#[test]
fn test_normalize_zscore_masked_creates_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let mask = dir.path().join("mask.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    write_half_mask_image(&mask);
    let args = NormalizeArgs {
        mask: Some(mask),
        ..default_args(NormalizeMethod::Zscore, input, output.clone())
    };
    run(args).unwrap();
    assert!(output.exists(), "output file must be created");
}

#[test]
fn test_normalize_zscore_masked_mean_of_foreground_voxels_near_zero() {
    // Masked region: ramp values 0..31 (first 32 voxels in row-major order).
    // μ_mask = (0 + 1 + … + 31) / 32 = 15.5.
    // After normalization: output_i = (i − 15.5) / σ, so
    //   mean(output_i for i in 0..32) = 0 by construction (μ subtracted).
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let mask = dir.path().join("mask.nii.gz");
    let output = dir.path().join("out.nii.gz");
    write_ramp_image(&input);
    write_half_mask_image(&mask);
    let args = NormalizeArgs {
        mask: Some(mask),
        ..default_args(NormalizeMethod::Zscore, input, output.clone())
    };
    run(args).unwrap();
    let im = read_image_native(&output).expect("read native masked z-score output");
    let vals = im
        .data_slice()
        .expect("contiguous native masked z-score output");
    // First 32 voxels correspond to the masked (foreground) region.
    let mean: f64 = vals[..32].iter().map(|&v| v as f64).sum::<f64>() / 32.0;
    assert!(
        mean.abs() < 1e-4,
        "mean of normalized foreground voxels must be ≈ 0, got {mean}"
    );
}
