use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_image::Image;

type TestBackend = NdArray<f32>;


fn get_labels(image: &Image<TestBackend, 3>) -> Vec<u32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|&v| v as u32)
        .collect()
}

// ── Positive tests ────────────────────────────────────────────────────────────

/// Seeds are never overwritten. If image is uniform and seed has two labels,
/// the seed positions must keep their original labels.
#[test]
fn test_seeds_are_immutable() {
    // 1x1x4: uniform intensities, seed=[1,0,0,2]
    let image = make_image(vec![100.0_f32; 4], [1, 1, 4]);
    let seeds = make_image(vec![1.0_f32, 0.0, 0.0, 2.0], [1, 1, 4]);
    let result = growcut(&image, &seeds, 100);
    let labels = get_labels(&result);
    assert_eq!(labels[0], 1, "seed at index 0 must remain label 1");
    assert_eq!(labels[3], 2, "seed at index 3 must remain label 2");
}

/// On a uniform intensity image with seeds at both ends, GrowCut partitions
/// the volume by proximity (Voronoi-like).
#[test]
fn test_uniform_intensity_voronoi_split() {
    // 1x1x8: uniform intensities, seed1 at 0, seed2 at 7
    let image = make_image(vec![100.0_f32; 8], [1, 1, 8]);
    let mut seed_vals = vec![0.0_f32; 8];
    seed_vals[0] = 1.0;
    seed_vals[7] = 2.0;
    let seeds = make_image(seed_vals, [1, 1, 8]);
    let result = growcut(&image, &seeds, 200);
    let labels = get_labels(&result);
    // All voxels must be labeled (no zeros after convergence)
    for (i, &l) in labels.iter().enumerate() {
        assert!(
            l > 0,
            "voxel {i} must be labeled after convergence, got {l}"
        );
    }
    // Symmetry: label 1 should be in indices 0..=3 and label 2 in 4..=7
    // (or 0..=3 and 4..=7 with the boundary at 3.5)
    assert_eq!(labels[0], 1, "index 0 must be label 1");
    assert_eq!(labels[7], 2, "index 7 must be label 2");
    assert_eq!(
        labels[1], 1,
        "index 1 should be label 1 (closest to seed 1)"
    );
    assert_eq!(
        labels[6], 2,
        "index 6 should be label 2 (closest to seed 2)"
    );
}

/// High-contrast barrier separates the two seeds. The high-intensity wall
/// between seeds greatly reduces g(j,i) so seeds cannot cross it.
#[test]
fn test_high_contrast_barrier_limits_propagation() {
    // 1x1x7: [low, low, low, HIGH, low, low, low]
    // seed1 at 0, seed2 at 6. Barrier at index 3.
    // With barrier intensity >> background, g across barrier ≈ 0.
    let mut img = vec![10.0_f32; 7];
    img[3] = 1000.0; // barrier
    let image = make_image(img, [1, 1, 7]);
    let mut seed_vals = vec![0.0_f32; 7];
    seed_vals[0] = 1.0;
    seed_vals[6] = 2.0;
    let seeds = make_image(seed_vals, [1, 1, 7]);
    let result = growcut(&image, &seeds, 200);
    let labels = get_labels(&result);
    // Seed positions must be preserved
    assert_eq!(labels[0], 1);
    assert_eq!(labels[6], 2);
    // Low-intensity region left of barrier belongs to label 1
    assert_eq!(labels[1], 1, "left of barrier must be label 1");
    assert_eq!(labels[2], 1, "left of barrier must be label 1");
    // Low-intensity region right of barrier belongs to label 2
    assert_eq!(labels[4], 2, "right of barrier must be label 2");
    assert_eq!(labels[5], 2, "right of barrier must be label 2");
}

/// Single label seed fills the entire volume when all voxels are uniform.
#[test]
fn test_single_seed_fills_entire_volume() {
    let n = 27;
    let image = make_image(vec![50.0_f32; n], [3, 3, 3]);
    let mut seed_vals = vec![0.0_f32; n];
    seed_vals[0] = 1.0; // single seed at corner
    let seeds = make_image(seed_vals, [3, 3, 3]);
    let result = growcut(&image, &seeds, 500);
    let labels = get_labels(&result);
    assert!(
        labels.iter().all(|&l| l == 1),
        "all voxels must be label 1 after fill"
    );
}

/// Spatial metadata (origin, spacing, direction) is preserved.
#[test]
fn test_spatial_metadata_preserved() {
    let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
    let seeds = make_image(vec![1.0_f32; 8], [2, 2, 2]);
    let result = growcut(&image, &seeds, 10);
    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Negative / boundary tests ─────────────────────────────────────────────────

/// Mismatched shapes between image and seeds must panic.
#[test]
#[should_panic(expected = "identical shapes")]
fn test_shape_mismatch_panics() {
    let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
    let seeds = make_image(vec![1.0_f32; 27], [3, 3, 3]);
    let _ = growcut(&image, &seeds, 10);
}

/// max_iter=0 leaves the label map unchanged (just seed initialization, no iterations).
#[test]
fn test_zero_iterations_returns_seed_map() {
    let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
    let seed_vals = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let seeds = make_image(seed_vals.clone(), [2, 2, 2]);
    let result = growcut(&image, &seeds, 0);
    let labels = get_labels(&result);
    // Without any iteration only the seed at index 0 is labeled.
    assert_eq!(labels[0], 1, "seed must remain labeled");
    for &l in labels.iter().skip(1) {
        assert_eq!(l, 0, "unlabeled voxels remain 0 with 0 iterations");
    }
}

/// GrowCutFilter struct and growcut function produce identical results.
#[test]
fn test_filter_struct_matches_function() {
    let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
    let seed_vals = vec![1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
    let seeds = make_image(seed_vals, [2, 2, 2]);
    let via_fn = growcut(&image, &seeds, 50);
    let via_struct = GrowCutFilter::new(50).apply(&image, &seeds);
    let fn_labels = get_labels(&via_fn);
    let struct_labels = get_labels(&via_struct);
    assert_eq!(fn_labels, struct_labels, "function and struct must match");
}
