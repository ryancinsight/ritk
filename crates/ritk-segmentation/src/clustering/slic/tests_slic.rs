//! Tests for SLIC super-pixel segmentation.
//! Extracted to keep the 500-line structural limit.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::clustering::slic::coords::decode_coords_dyn as decode_coords;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support::make_image;
use ritk_image::Image as NativeImage;

type B = SequentialBackend;

fn make_image_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<f32, B, 2> {
    make_image(data, dims)
}

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    make_image(data, dims)
}

fn get_slice_2d(image: &Image<f32, B, 2>) -> Vec<f32> {
    image.data().to_vec()
}

fn get_slice_3d(image: &Image<f32, B, 3>) -> Vec<f32> {
    image.data().to_vec()
}

fn config(
    n_superpixels: usize,
    compactness: f32,
    max_iterations: usize,
    tolerance: f32,
    min_component_size: usize,
) -> SlicConfig {
    SlicConfig::new(n_superpixels)
        .expect("infallible: validated precondition")
        .with_compactness(compactness)
        .expect("infallible: validated precondition")
        .with_max_iterations(max_iterations)
        .expect("infallible: validated precondition")
        .with_tolerance(tolerance)
        .expect("infallible: validated precondition")
        .with_min_component_size(min_component_size)
}

// ── Test 1: Uniform image ─────────────────────────────────────────────────────

#[test]
fn test_uniform_image_single_label() {
    // All voxels same intensity → constant image → all label 0.
    let data = vec![42.0_f32; 64];
    let image = make_image_3d(data, [4, 4, 4]);
    let config = SlicConfig::new(4).expect("infallible: validated precondition");
    let result = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels = get_slice_3d(&result);
    for (i, &l) in labels.iter().enumerate() {
        assert!(
            (l - 0.0).abs() < f32::EPSILON,
            "constant image voxel {} must have label 0, got {}",
            i,
            l
        );
    }
}

// ── Test 2: Known two-region image ────────────────────────────────────────────

#[test]
fn test_two_region_boundary_respected() {
    // 8×8 2D image: left half intensity 10.0, right half intensity 200.0.
    // With low compactness, SLIC should respect the intensity boundary.
    let mut data = vec![10.0_f32; 32];
    data.extend(vec![200.0_f32; 32]);
    let image = make_image_2d(data, [8, 8]);

    let config = config(4, 1.0, 20, 1e-6, 3);
    let result = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels = get_slice_2d(&result);

    // Compute mean intensity per label for left-half and right-half voxels.
    let mut label_intensity_sum: std::collections::HashMap<u32, (f64, usize)> =
        std::collections::HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        let li = l as u32;
        let entry = label_intensity_sum.entry(li).or_insert((0.0, 0));
        entry.0 += if i < 32 { 10.0 } else { 200.0 };
        entry.1 += 1;
    }

    // Each label should have a mean intensity close to either 10 or 200.
    for (&label, &(sum, count)) in &label_intensity_sum {
        let mean = sum / count as f64;
        let near_low = (mean - 10.0).abs() < 30.0;
        let near_high = (mean - 200.0).abs() < 30.0;
        assert!(
            near_low || near_high,
            "label {} mean intensity {} should be near 10 or 200",
            label,
            mean
        );
    }
}

// ── Test 3: Small image with n_superpixels=2 ──────────────────────────────────

#[test]
fn test_small_image_two_superpixels() {
    // 4×4×4 image with n_superpixels=2: two distinct intensity halves.
    // First 2 z-slices: intensity 0.0; last 2 z-slices: intensity 255.0.
    let mut data = vec![0.0_f32; 32]; // z=0,1: 2*4*4 = 32 voxels
    data.extend(vec![255.0_f32; 32]); // z=2,3: 2*4*4 = 32 voxels
    let image = make_image_3d(data, [4, 4, 4]);

    let config = config(2, 1.0, 20, 1e-6, 0);
    let result = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels = get_slice_3d(&result);

    // The first 32 voxels (z=0,1) should share a label different from
    // the last 32 voxels (z=2,3).
    let first_label = labels[0] as u32;
    let last_label = labels[63] as u32;
    assert_ne!(
        first_label, last_label,
        "two intensity regions must get different labels: {} vs {}",
        first_label, last_label
    );

    let distinct: std::collections::HashSet<u32> = labels.iter().map(|&l| l as u32).collect();
    assert_eq!(
        distinct.len(),
        2,
        "exactly 2 distinct labels expected, got {}",
        distinct.len()
    );
}

// ── Test 4: Single superpixel ─────────────────────────────────────────────────

#[test]
fn test_single_superpixel_all_label_zero() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 4.0).collect();
    let image = make_image_3d(data, [4, 4, 4]);

    let config = SlicConfig::new(1).expect("infallible: validated precondition");
    let result = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels = get_slice_3d(&result);

    for (i, &l) in labels.iter().enumerate() {
        assert!(
            (l - 0.0).abs() < f32::EPSILON,
            "n_superpixels=1 must assign label 0, voxel {} got {}",
            i,
            l
        );
    }
}

// ── Test 5: Label count bounded by n_superpixels ─────────────────────────────

#[test]
fn test_label_count_bounded() {
    let k = 8;
    let data: Vec<f32> = (0..256).map(|i| (i as f32) * 1.0).collect();
    let image = make_image_3d(data, [4, 4, 16]);

    let config = config(k, 10.0, 10, 1e-3, 3);
    let result = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels = get_slice_3d(&result);

    let distinct: std::collections::HashSet<u32> = labels.iter().map(|&l| l as u32).collect();
    assert!(
        distinct.len() <= k,
        "number of distinct labels ({}) must be ≤ n_superpixels ({})",
        distinct.len(),
        k
    );

    // Labels must be consecutive from 0.
    let max_label = *distinct.iter().max().unwrap_or(&0);
    assert!(
        (max_label as usize) < distinct.len(),
        "labels must be consecutive: max_label={}, distinct_count={}",
        max_label,
        distinct.len()
    );
}

// ── Test 6: Determinism ──────────────────────────────────────────────────────

#[test]
fn test_deterministic_output() {
    let mut data = vec![10.0_f32; 32];
    data.extend(vec![200.0_f32; 32]);
    let image = make_image_3d(data, [4, 4, 4]);

    let config = SlicConfig::new(4).expect("infallible: validated precondition");
    let r1 = SlicSuperpixelFilter::new(config.clone())
        .apply(&image)
        .expect("infallible: validated precondition");
    let r2 = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");

    let l1 = get_slice_3d(&r1);
    let l2 = get_slice_3d(&r2);
    assert_eq!(l1, l2, "same config must produce identical results");
}

// ── Test 7: Convergence ──────────────────────────────────────────────────────

#[test]
fn test_algorithm_converges() {
    // With a generous tolerance, the algorithm must terminate quickly.
    let data: Vec<f32> = (0..1000).map(|i| (i % 7) as f32 * 40.0).collect();
    let image = make_image_3d(data, [10, 10, 10]);

    let config = config(10, 10.0, 3, 1.0, 3);
    let result = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels = get_slice_3d(&result);

    // Verify the algorithm produced valid labels (didn't hang/crash).
    assert_eq!(labels.len(), 1000);
    for &l in &labels {
        assert!(l >= 0.0, "label must be non-negative, got {}", l);
    }
}

// ── Test 8: Spatial metadata preservation ─────────────────────────────────────

#[test]
fn test_spatial_metadata_preserved() {
    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 10.0).collect();
    let image = make_image_3d(data, [2, 3, 4]);

    let config = SlicConfig::new(2).expect("infallible: validated precondition");
    let result = SlicSuperpixelFilter::new(config)
        .apply(&image)
        .expect("infallible: validated precondition");

    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Test 9: Compactness effect ────────────────────────────────────────────────

#[test]
fn test_compactness_effect() {
    // Create an image with a sharp boundary.
    let mut data = vec![0.0_f32; 50];
    data.extend(vec![255.0_f32; 50]);
    let image = make_image_2d(data, [10, 10]);

    // Low compactness → labels driven by intensity, more irregular boundaries.
    let config_low = config(4, 1.0, 20, 1e-6, 3);
    let result_low = SlicSuperpixelFilter::new(config_low)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels_low = get_slice_2d(&result_low);

    // High compactness → labels driven by spatial proximity, more regular.
    let config_high = config(4, 100.0, 20, 1e-6, 3);
    let result_high = SlicSuperpixelFilter::new(config_high)
        .apply(&image)
        .expect("infallible: validated precondition");
    let labels_high = get_slice_2d(&result_high);

    // With high compactness, each superpixel should be spatially compact.
    // Compute spatial variance of labels for both.
    let var_low = compute_spatial_variance(&labels_low, &[10, 10], 2);
    let var_high = compute_spatial_variance(&labels_high, &[10, 10], 2);

    // Higher compactness should yield lower spatial variance (more regular shapes).
    assert!(
        var_high <= var_low * 1.5 + 1e-10,
        "high compactness spatial variance ({}) should not greatly exceed low compactness ({})",
        var_high,
        var_low
    );
}

/// Compute average spatial variance of all superpixel labels.
/// Lower variance = more spatially compact superpixels.
fn compute_spatial_variance(labels: &[f32], shape: &[usize], ndim: usize) -> f64 {
    let n = labels.len();
    let mut label_coords: std::collections::HashMap<u32, (Vec<f64>, usize)> =
        std::collections::HashMap::new();

    for i in 0..n {
        let coords = decode_coords(i, shape);
        let l = labels[i] as u32;
        let entry = label_coords.entry(l).or_insert((vec![0.0; ndim], 0));
        for d in 0..ndim {
            entry.0[d] += coords[d] as f64;
        }
        entry.1 += 1;
    }

    // Compute means.
    let mut label_means: std::collections::HashMap<u32, Vec<f64>> =
        std::collections::HashMap::new();
    for (&l, (sums, count)) in &label_coords {
        let means: Vec<f64> = sums.iter().map(|&s| s / *count as f64).collect();
        label_means.insert(l, means);
    }

    // Compute total variance.
    let mut total_var = 0.0_f64;
    for i in 0..n {
        let coords = decode_coords(i, shape);
        let l = labels[i] as u32;
        if let Some(means) = label_means.get(&l) {
            for d in 0..ndim {
                let diff = coords[d] as f64 - means[d];
                total_var += diff * diff;
            }
        }
    }

    total_var / n as f64
}

// ── Test 10: Convenience function ─────────────────────────────────────────────

#[test]
fn test_convenience_fn() {
    let data = vec![10.0_f32; 32];
    let image = make_image_2d(data, [8, 4]);
    let result = slic_superpixel(&image, 4).expect("infallible: validated precondition");
    let labels = get_slice_2d(&result);

    // Constant image → all label 0.
    for &l in &labels {
        assert!(
            (l - 0.0).abs() < f32::EPSILON,
            "constant image via convenience fn must yield label 0, got {}",
            l
        );
    }
}

// ── Test 11: Output shape matches input ───────────────────────────────────────

#[test]
fn test_output_shape_matches_input() {
    let dims = [6, 8, 10];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 50.0).collect();
    let image = make_image_3d(data, dims);
    let result =
        SlicSuperpixelFilter::new(SlicConfig::new(8).expect("infallible: validated precondition"))
            .apply(&image)
            .expect("infallible: validated precondition");
    assert_eq!(result.shape(), dims);
}

// ── Test 12: Default config ──────────────────────────────────────────────────

#[test]
fn test_default_config() {
    let c = SlicConfig::default();
    assert_eq!(c.n_superpixels, 100);
    assert!((c.compactness - 10.0).abs() < 1e-10);
    assert_eq!(c.max_iterations, 10);
    assert!((c.tolerance - 1e-3).abs() < 1e-15);
    assert_eq!(c.min_component_size, 5);
}

fn assert_native_legacy_exact<const D: usize>(values: Vec<f32>, dimensions: [usize; D]) {
    let native = NativeImage::from_flat_on(
        values.clone(),
        dimensions,
        Point::new([2.0; D]),
        Spacing::new([0.5; D]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("infallible: validated precondition");
    let legacy = make_image::<f32, B, D>(values, dimensions);
    let filter = SlicSuperpixelFilter::new(config(4, 5.0, 10, 0.0, 0));
    let native_output = filter
        .apply_native(&native, &SequentialBackend)
        .expect("infallible: validated precondition");
    let legacy_output = filter
        .apply(&legacy)
        .expect("infallible: validated precondition");
    assert_eq!(
        native_output
            .data_slice()
            .expect("infallible: validated precondition"),
        legacy_output
            .data_slice()
            .expect("invariant: contiguous host storage")
    );
    assert_eq!(*native_output.origin(), Point::new([2.0; D]));
    assert_eq!(*native_output.spacing(), Spacing::new([0.5; D]));
    assert_eq!(*native_output.direction(), Direction::identity());
}

#[test]
fn native_and_legacy_standard_slic_are_exact_in_two_and_three_dimensions() {
    assert_native_legacy_exact(
        (0..36).map(|index| (index % 7) as f32 * 3.0).collect(),
        [6, 6],
    );
    assert_native_legacy_exact(
        (0..64).map(|index| (index % 5) as f32 * 11.0).collect(),
        [4, 4, 4],
    );
}

#[test]
fn standard_slic_rejects_invalid_configuration_exactly() {
    assert_eq!(
        SlicConfig::new(0).unwrap_err().to_string(),
        "SLIC superpixel count must be at least 1, got 0"
    );
    assert_eq!(
        SlicConfig::default()
            .with_max_iterations(0)
            .unwrap_err()
            .to_string(),
        "SLIC maximum iterations must be at least 1, got 0"
    );
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        assert_eq!(
            SlicConfig::default()
                .with_tolerance(value)
                .unwrap_err()
                .to_string(),
            format!("SLIC tolerance must be finite and nonnegative, got {value}")
        );
    }
}

#[test]
fn standard_slic_rejects_nonfinite_samples_and_unsupported_rank() {
    let invalid = make_image_2d(vec![0.0, f32::NAN, 1.0, 2.0], [2, 2]);
    assert_eq!(
        SlicSuperpixelFilter::new(SlicConfig::default())
            .apply(&invalid)
            .unwrap_err()
            .to_string(),
        "standard SLIC sample at flat index 1 must be finite, got NaN"
    );
    let rank_one = make_image::<f32, B, 1>(vec![0.0, 1.0], [2]);
    assert_eq!(
        SlicSuperpixelFilter::new(SlicConfig::default())
            .apply(&rank_one)
            .unwrap_err()
            .to_string(),
        "standard SLIC requires rank 2 or 3, got 1"
    );

    let native_invalid = NativeImage::from_flat_on(
        vec![0.0, f32::INFINITY, 1.0, 2.0],
        [2, 2],
        Point::new([0.0; 2]),
        Spacing::new([1.0; 2]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("infallible: validated precondition");
    assert_eq!(
        SlicSuperpixelFilter::new(SlicConfig::default())
            .apply_native(&native_invalid, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "standard SLIC sample at flat index 1 must be finite, got inf"
    );
    let native_rank_four = NativeImage::from_flat_on(
        vec![0.0],
        [1, 1, 1, 1],
        Point::new([0.0; 4]),
        Spacing::new([1.0; 4]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("infallible: validated precondition");
    assert_eq!(
        SlicSuperpixelFilter::new(SlicConfig::default())
            .apply_native(&native_rank_four, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "standard SLIC requires rank 2 or 3, got 4"
    );
    let empty = make_image_2d(Vec::new(), [0, 2]);
    assert_eq!(
        SlicSuperpixelFilter::new(SlicConfig::default())
            .apply(&empty)
            .unwrap_err()
            .to_string(),
        "standard SLIC requires nonzero dimensions, got [0, 2]"
    );
}

#[test]
fn standard_slic_extreme_compactness_produces_finite_exact_labels() {
    let image = make_image_2d((0..16).map(|index| index as f32).collect(), [4, 4]);
    // Search-window deltas are bounded by 2S. Dividing sqrt(MAX) by
    // 2S*sqrt(D) keeps the complete spatial norm representable.
    let compactness = f32::MAX.sqrt() / (4.0 * 2.0_f32.sqrt());
    let labels = SlicSuperpixelFilter::new(
        SlicConfig::new(4)
            .expect("infallible: validated precondition")
            .with_compactness(compactness)
            .expect("infallible: validated precondition")
            .with_min_component_size(0),
    )
    .apply(&image)
    .expect("infallible: validated precondition");
    assert_eq!(
        labels
            .data_slice()
            .expect("invariant: contiguous host storage"),
        &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0]
    );
}
