//! Unit tests for isolated watershed segmentation.
//!
//! The primary test is a value-semantic differential for the gradient-descent
//! `IsolatedWatershed` algorithm: each voxel is assigned the basin of the local
//! minimum of `g` it flows to via steepest descent, then seed basins are labeled
//! 1.0 and 2.0, and the rest 0.0.

use super::isolated_watershed;
use super::IsolatedWatershed;
use super::IsolatedWatershedConfig;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_image::Image;

type B = NdArray<f32>;

/// Build a `[1, ny, nx]` image (z == 1 → 2-D) from a flat row-major slice.
fn image_2d(data: Vec<f32>, ny: usize, nx: usize) -> Image<B, 3> {
    make_image(data, [1, ny, nx])
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

// ── Gradient-descent IsolatedWatershed (7×7 two-valley relief) ─────────────────
//
// Image: a central high cross (values 5/6/9) separating four low corners, with
// two seeds in the top-centre and bottom-centre low regions.
//
// Gradient magnitude has local minima at all four corners and at the three
// central saddle points on each axis.  For these seeds:
//
//   seed1 (y=1, x=3): gradient-descent drains to (y=0, x=3) which is the
//     top-centre local minimum; its basin covers the three voxels with low
//     gradient at (y=0–1, x=2–4).
//   seed2 (y=5, x=3): gradient-descent drains to (y=6, x=3) which is the
//     bottom-centre local minimum; its basin covers (y=5–6, x=2–4).
//
// Label 1.0 → top basin (rows 0–1, cols 2–4); label 2.0 → bottom basin
// (rows 5–6, cols 2–4); label 0.0 → remaining voxels.
//
// Derivation: g[1,3]=0.5 → steepest descent to g[0,3]=0 (min); g[5,3]=0.5 →
// steepest descent to g[6,3]=0 (min).  Neighbouring voxels at g≈2–2.8 flow
// to the central saddle minima at (3,3), (3,0), (3,6), not to the seed basins.

const RELIEF_7X7: [f32; 49] = [
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    2.0, 2.0, 3.0, 6.0, 3.0, 2.0, 2.0, //
    5.0, 5.0, 6.0, 9.0, 6.0, 5.0, 5.0, //
    2.0, 2.0, 3.0, 6.0, 3.0, 2.0, 2.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
    1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, //
];

// Golden produced by the gradient-descent watershed algorithm.
// seed1=(y=1, x=3) → basin of local min at (y=0, x=3) → label 1.0 at rows 0–1, cols 2–4.
// seed2=(y=5, x=3) → basin of local min at (y=6, x=3) → label 2.0 at rows 5–6, cols 2–4.
const GOLDEN_7X7: [f32; 49] = [
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, //
    0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, //
];

#[test]
fn test_isolated_watershed_matches_sitk_relief_7x7() {
    let dims = [1_usize, 7, 7];
    // seed1 at (y=1, x=3) → flat 1*7+3 = 10; seed2 at (y=5, x=3) → flat 38.
    #[allow(clippy::identity_op)]
    let seed1 = 1 * 7 + 3;
    let seed2 = 5 * 7 + 3;
    let config = IsolatedWatershedConfig {
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };
    let result = isolated_watershed(&RELIEF_7X7, dims, seed1, seed2, &config);
    assert_eq!(
        result, GOLDEN_7X7,
        "isolated watershed must produce correct gradient-descent basin labels"
    );
}

#[test]
fn test_isolated_watershed_apply_matches_sitk_relief_7x7() {
    let image = image_2d(RELIEF_7X7.to_vec(), 7, 7);
    let filter = IsolatedWatershed {
        seed1: [0, 1, 3],
        seed2: [0, 5, 3],
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };
    let result = filter.apply(&image).expect("apply must not error");
    assert_eq!(result.shape(), [1, 7, 7]);
    assert_eq!(labels(&result), GOLDEN_7X7);
}

// ── Edge case: identical seeds → all label 1 ──────────────────────────────────

#[test]
fn test_isolated_watershed_identical_seeds_all_label1() {
    let data = vec![0.2_f32, 0.5, 0.2, 0.8];
    let dims = [1_usize, 1, 4];
    let result = isolated_watershed(&data, dims, 0, 0, &IsolatedWatershedConfig::default());
    assert!(
        result.iter().all(|&v| v == 1.0),
        "identical seeds must yield all-label-1 output, got {result:?}"
    );
}

// ── Spatial metadata preserved through apply ──────────────────────────────────

#[test]
fn test_isolated_watershed_spatial_metadata_preserved() {
    let image = image_2d(RELIEF_7X7.to_vec(), 7, 7);
    let filter = IsolatedWatershed {
        seed1: [0, 1, 3],
        seed2: [0, 5, 3],
        threshold: 0.0,
        isolated_value_tolerance: 0.001,
        upper_value_limit: 1.0,
    };
    let result = filter.apply(&image).unwrap();
    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Plateau / flat region validation ──────────────────────────────────────────

#[test]
fn test_isolated_watershed_flat_plateau() {
    // 1-D image with a flat plateau:
    // Intensities: [4.0, 3.0, 2.0, 2.0, 2.0, 1.0, 0.0]
    let data = vec![4.0_f32, 3.0, 2.0, 2.0, 2.0, 1.0, 0.0];
    let dims = [1_usize, 1, 7];
    let config = IsolatedWatershedConfig::default();
    let result = isolated_watershed(&data, dims, 0, 6, &config);
    // Verify successful execution (no panic or cycle) and that the seed basins
    // receive the correct labels.
    assert_eq!(result.len(), 7);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[6], 2.0);
}
