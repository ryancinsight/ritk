//! Unit tests for isolated watershed segmentation.
//!
//! Each test exercises the end-to-end path through `IsolatedWatershed::apply`
//! on a 1-D strip image (shape `[1, 1, N]`) where the expected label assignment
//! follows directly from the barrier pixel value.

use super::isolated_watershed;
use super::IsolatedWatershed;
use super::IsolatedWatershedConfig;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_image::Image;

type B = NdArray<f32>;

/// Build a `[1, 1, N]` image from a flat slice.
fn strip(data: Vec<f32>) -> Image<B, 3> {
    let n = data.len();
    make_image(data, [1, 1, n])
}

/// Extract labels as a flat `Vec<f32>` from a label image.
fn labels(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("label image must be f32")
        .to_vec()
}

// ── Test 1: 3-voxel strip, barrier in middle ───────────────────────────────────
//
// Image [0.1, 0.5, 0.1], dims [1, 1, 3].
// Seeds at (0,0,0) = flat 0  and (0,0,2) = flat 2.
//
// At T < 0.5: pixels 0 and 2 active, pixel 1 not → separated.
// At T ≥ 0.5: all active → connected.
// T* = sup{separated} ≈ 0.5 - ε  ⟹  binary search converges to lo ≈ 0.5.
//
// Expected labels: [1.0, 3.0, 2.0]

#[test]
fn test_isolated_watershed_3voxel_barrier_in_middle() {
    let data = vec![0.1_f32, 0.5, 0.1];
    let dims = [1_usize, 1, 3];
    // seed (0,0,0) → flat 0; seed (0,0,2) → flat 2.
    let seed1 = 0_usize;
    let seed2 = 2_usize;
    let config = IsolatedWatershedConfig {
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };

    let result = isolated_watershed(&data, dims, seed1, seed2, &config);

    assert_eq!(result.len(), 3, "label vector must have 3 elements");
    assert_eq!(
        result[0], 1.0,
        "seed1 (pixel 0) must be in label-1 region, got {}",
        result[0]
    );
    assert_eq!(
        result[2], 2.0,
        "seed2 (pixel 2) must be in label-2 region, got {}",
        result[2]
    );
    assert_eq!(
        result[1], 3.0,
        "barrier pixel must be label 3 (unreachable), got {}",
        result[1]
    );
    assert_ne!(
        result[0], result[2],
        "seeds must be in different labelled regions"
    );
}

// ── Test 2: 5-voxel strip, high barrier in middle ─────────────────────────────
//
// Image [0.0, 0.0, 0.9, 0.0, 0.0], dims [1, 1, 5].
// Seeds at flat 0 and flat 4.
//
// Pixels 0,1,3,4 have value 0.0; pixel 2 has value 0.9.
// At T < 0.9: pixels 0,1 active on left, 3,4 active on right — pixel 2 blocks
//             the path → separated.
// At T ≥ 0.9: all active → connected.
// T* ≈ 0.9 - ε.
//
// Expected labels: [1.0, 1.0, 3.0, 2.0, 2.0]

#[test]
fn test_isolated_watershed_5voxel_high_barrier() {
    let data = vec![0.0_f32, 0.0, 0.9, 0.0, 0.0];
    let dims = [1_usize, 1, 5];
    let seed1 = 0_usize;
    let seed2 = 4_usize;
    let config = IsolatedWatershedConfig {
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };

    let result = isolated_watershed(&data, dims, seed1, seed2, &config);

    assert_eq!(result.len(), 5, "label vector must have 5 elements");

    // Seeds in different regions.
    assert_eq!(result[0], 1.0, "seed1 (pixel 0) must be label 1");
    assert_eq!(result[4], 2.0, "seed2 (pixel 4) must be label 2");

    // Voxels reachable from each seed.
    assert_eq!(
        result[1], 1.0,
        "pixel 1 (connected to seed1) must be label 1"
    );
    assert_eq!(
        result[3], 2.0,
        "pixel 3 (connected to seed2) must be label 2"
    );

    // Barrier voxel unreachable at T*.
    assert_eq!(result[2], 3.0, "barrier pixel 2 must be label 3");
}

// ── Test 3: identical seeds → all label 1 ─────────────────────────────────────

#[test]
fn test_isolated_watershed_identical_seeds_all_label1() {
    let data = vec![0.2_f32, 0.5, 0.2, 0.8];
    let dims = [1_usize, 1, 4];
    let seed = 0_usize;
    let config = IsolatedWatershedConfig::default();

    let result = isolated_watershed(&data, dims, seed, seed, &config);

    assert!(
        result.iter().all(|&v| v == 1.0),
        "identical seeds must yield all-label-1 output, got {:?}",
        result
    );
}

// ── Test 4: high-level `IsolatedWatershed::apply` end-to-end ──────────────────

#[test]
fn test_isolated_watershed_apply_api() {
    let image = strip(vec![0.1_f32, 0.5, 0.1]);

    let filter = IsolatedWatershed {
        seed1: [0, 0, 0],
        seed2: [0, 0, 2],
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };

    let result = filter.apply(&image).expect("apply must not error");
    assert_eq!(result.shape(), [1, 1, 3], "output shape must match input");

    let lbl = labels(&result);
    assert_eq!(lbl[0], 1.0, "seed1 voxel must be label 1");
    assert_eq!(lbl[2], 2.0, "seed2 voxel must be label 2");
    assert_eq!(lbl[1], 3.0, "barrier voxel must be label 3");
}

// ── Test 5: spatial metadata preserved through apply ──────────────────────────

#[test]
fn test_isolated_watershed_spatial_metadata_preserved() {
    let image = strip(vec![0.1_f32, 0.5, 0.1]);

    let filter = IsolatedWatershed {
        seed1: [0, 0, 0],
        seed2: [0, 0, 2],
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };

    let result = filter.apply(&image).unwrap();
    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}
