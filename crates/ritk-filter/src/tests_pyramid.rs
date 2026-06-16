//! Tests for pyramid
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

fn make_image(shape: [usize; 3]) -> Image<B, 3> {
    ts::fill_image::<B, 3>(shape, 0.0)
}

// ── MultiResolutionPyramid ────────────────────────────────────────────────────

/// Level count matches the schedule length.
#[test]
fn pyramid_level_count_matches_schedule() {
    let img = make_image([8, 8, 8]);
    let shrink: Vec<[usize; 3]> = vec![[4, 4, 4], [2, 2, 2], [1, 1, 1]];
    let sigmas: Vec<[f64; 3]> = vec![[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
    let pyr = MultiResolutionPyramid::<B, 3>::new(&img, &shrink, &sigmas);
    assert_eq!(pyr.levels(), 3, "pyramid must have 3 levels");
}

/// Identity schedule (factor=1, sigma=0): every level is a clone of the input.
#[test]
fn pyramid_identity_schedule_clones_image() {
    let img = make_image([6, 6, 6]);
    let shrink: Vec<[usize; 3]> = vec![[1, 1, 1]];
    let sigmas: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
    let pyr = MultiResolutionPyramid::<B, 3>::new(&img, &shrink, &sigmas);
    assert_eq!(pyr.levels(), 1);
    assert_eq!(
        pyr.get_level(0).shape(),
        img.shape(),
        "identity schedule must preserve shape"
    );
}

/// Coarser levels produce shapes smaller than or equal to finer levels.
///
/// # Derivation
/// Schedule [4,4,4] -> [2,2,2] -> [1,1,1]: shapes are ≈ [2,2,2] < ≈ [4,4,4] < [8,8,8].
#[test]
fn pyramid_coarser_levels_have_smaller_shape() {
    let img = make_image([8, 8, 8]);
    let (shrink, sigmas) = MultiResolutionPyramid::<B, 3>::default_schedule(3);
    let pyr = MultiResolutionPyramid::<B, 3>::new(&img, &shrink, &sigmas);
    // Levels ordered coarsest-to-finest (default_schedule convention).
    let n0: usize = pyr.get_level(0).shape().iter().product();
    let n1: usize = pyr.get_level(1).shape().iter().product();
    let n2: usize = pyr.get_level(2).shape().iter().product();
    assert!(
        n0 <= n1,
        "level 0 (coarsest) must have <= voxels than level 1: {n0} vs {n1}"
    );
    assert!(
        n1 <= n2,
        "level 1 must have <= voxels than level 2 (finest): {n1} vs {n2}"
    );
}

// ── default_schedule ──────────────────────────────────────────────────────

/// default_schedule(3) produces 3 levels with factors [4,2,1] and sigmas [2.0,1.0,0.0].
///
/// # Derivation
/// For `levels=3`, `i=0..2`, `exponent = 2-i = [2,1,0]`.
/// factor = 2^exponent = [4, 2, 1].
/// sigma = 0.5 * factor if factor>1 else 0.0 = [2.0, 1.0, 0.0].
#[test]
fn default_schedule_3_levels_correct_factors_and_sigmas() {
    let (shrink, sigmas) = MultiResolutionPyramid::<B, 3>::default_schedule(3);
    assert_eq!(shrink.len(), 3);
    assert_eq!(shrink[0], [4, 4, 4], "level 0 factor must be 4");
    assert_eq!(shrink[1], [2, 2, 2], "level 1 factor must be 2");
    assert_eq!(shrink[2], [1, 1, 1], "level 2 factor must be 1");
    assert!(
        (sigmas[0][0] - 2.0).abs() < 1e-9,
        "level 0 sigma must be 2.0"
    );
    assert!(
        (sigmas[1][0] - 1.0).abs() < 1e-9,
        "level 1 sigma must be 1.0"
    );
    assert!(
        (sigmas[2][0] - 0.0).abs() < 1e-9,
        "level 2 sigma must be 0.0"
    );
}

/// default_schedule(1) produces one level with factor=1 and sigma=0.0.
#[test]
fn default_schedule_single_level_is_identity() {
    let (shrink, sigmas) = MultiResolutionPyramid::<B, 3>::default_schedule(1);
    assert_eq!(shrink[0], [1, 1, 1]);
    assert!((sigmas[0][0] - 0.0).abs() < 1e-9);
}
