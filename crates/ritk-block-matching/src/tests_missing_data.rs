//! Missing-data boundary oracles for block matching.

use super::{match_block, BlockMatchingConfig, MovingSamples, SubpixelRefinement};

#[cfg(feature = "fft")]
use super::{metric_image, BlockMetric};
#[cfg(feature = "fft")]
use crate::{metric_image_fft, FftPadding};

const DIMS: [usize; 3] = [1, 40, 40];

fn config() -> BlockMatchingConfig {
    BlockMatchingConfig {
        block_radius: [0, 4, 4],
        search_radius: [0, 5, 5],
    }
}

fn texture(y: isize, x: isize) -> f32 {
    let seed = (y as i64)
        .wrapping_mul(104_729)
        .wrapping_add((x as i64).wrapping_mul(15_485_863)) as u64;
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64 / (1_u64 << 53) as f64) as f32
}

fn shifted_image(shift: [isize; 2]) -> Vec<f32> {
    let mut image = vec![0.0; DIMS.into_iter().product()];
    for y in 0..DIMS[1] {
        for x in 0..DIMS[2] {
            image[y * DIMS[2] + x] = texture(y as isize - shift[0], x as isize - shift[1]);
        }
    }
    image
}

fn pair_with_missing_slab() -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let fixed = shifted_image([0, 0]);
    let moving = shifted_image([3, -2]);
    let mut validity = vec![true; moving.len()];
    for y in 0..12 {
        for x in 0..DIMS[2] {
            validity[y * DIMS[2] + x] = false;
        }
    }
    (fixed, moving, validity)
}

/// Internal missing-data boundaries must exclude affected candidates without
/// letting a fill value become correlation evidence.
#[test]
fn moving_missing_samples_exclude_only_dependent_candidates() {
    let (fixed, moving, validity) = pair_with_missing_slab();
    let moving = MovingSamples::try_with_validity(&moving, &validity)
        .expect("fixture validity matches moving samples");
    let result = match_block(
        &fixed,
        moving,
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
    )
    .expect("valid candidates remain away from the unavailable slab");
    assert_eq!(result.displacement[0], 0.0);
    assert_eq!(result.displacement[1], 3.0);
    assert!((result.displacement[2] + 2.0).abs() < 0.05);
    assert!(result.peak_similarity > 0.999);
}

#[test]
fn all_valid_mask_is_identical_to_complete_input() {
    let fixed = shifted_image([0, 0]);
    let moving = shifted_image([3, -2]);
    let validity = vec![true; moving.len()];
    let complete = match_block(
        &fixed,
        MovingSamples::complete(&moving),
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::Parabolic,
    )
    .expect("complete match");
    let masked = match_block(
        &fixed,
        MovingSamples::try_with_validity(&moving, &validity).expect("matching mask length"),
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::Parabolic,
    )
    .expect("all-valid match");
    assert_eq!(masked, complete);
}

#[test]
fn all_invalid_candidates_report_no_finite_peak() {
    let fixed = shifted_image([0, 0]);
    let moving = shifted_image([3, -2]);
    let validity = vec![false; moving.len()];
    let result = match_block(
        &fixed,
        MovingSamples::try_with_validity(&moving, &validity).expect("matching mask length"),
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
    )
    .expect("unavailable candidates are a measured no-evidence outcome");
    assert_eq!(result.peak_similarity, f64::NEG_INFINITY);
}

#[test]
fn refinement_stays_at_the_integer_peak_beside_invalid_support() {
    let fixed = shifted_image([0, 0]);
    let moving = shifted_image([3, -2]);
    let mut validity = vec![true; moving.len()];
    for x in 0..DIMS[2] {
        validity[28 * DIMS[2] + x] = false;
    }
    let result = match_block(
        &fixed,
        MovingSamples::try_with_validity(&moving, &validity).expect("matching mask length"),
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::Parabolic,
    )
    .expect("the integer peak remains supported");
    assert_eq!(result.displacement[0], 0.0);
    assert_eq!(result.displacement[1], 3.0);
    assert!((result.displacement[2] + 2.0).abs() < 0.05);
    assert!(result.peak_similarity.is_finite());
}

#[test]
fn non_finite_moving_samples_are_unavailable() {
    let fixed = shifted_image([0, 0]);
    let mut moving = shifted_image([3, -2]);
    for y in 0..12 {
        for x in 0..DIMS[2] {
            moving[y * DIMS[2] + x] = f32::NAN;
        }
    }
    let result = match_block(
        &fixed,
        MovingSamples::complete(&moving),
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
    )
    .expect("finite candidates remain away from the unavailable slab");
    assert_eq!(result.displacement, [0.0, 3.0, -2.0]);
}

#[test]
fn fixed_missing_samples_are_rejected() {
    let mut fixed = shifted_image([0, 0]);
    let moving = shifted_image([0, 0]);
    fixed[20 * DIMS[2] + 20] = f32::NAN;
    let error = match_block(
        &fixed,
        MovingSamples::complete(&moving),
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
    )
    .expect_err("a non-finite fixed block has no defined correlation");
    assert!(error.to_string().contains("non-finite sample"));
}

#[test]
fn moving_validity_rejects_a_mismatched_length() {
    let moving = shifted_image([0, 0]);
    let validity = vec![true; 10];
    let error = MovingSamples::try_with_validity(&moving, &validity)
        .expect_err("validity must cover every moving sample");
    assert!(error.to_string().contains("does not match sample length"));
}

#[cfg(feature = "fft")]
#[test]
fn fft_matches_direct_with_internal_missing_data() {
    let (fixed, moving, validity) = pair_with_missing_slab();
    let moving = MovingSamples::try_with_validity(&moving, &validity)
        .expect("fixture validity matches moving samples");
    let direct = metric_image(
        &fixed,
        moving,
        DIMS,
        [0, 20, 20],
        config(),
        BlockMetric::NormalizedCrossCorrelation,
    )
    .expect("direct metric");
    let fft = metric_image_fft(
        &fixed,
        moving,
        DIMS,
        [0, 20, 20],
        config(),
        FftPadding::Zero,
    )
    .expect("FFT metric");
    for (index, (&expected, &actual)) in direct.values.iter().zip(&fft.values).enumerate() {
        match (expected.is_finite(), actual.is_finite()) {
            (true, true) => assert!(
                (expected - actual).abs() < 1.0e-9,
                "FFT/direct missing-data mismatch at {index}: {actual} vs {expected}"
            ),
            (false, false) => {}
            _ => panic!("FFT/direct missing-data support mismatch at {index}"),
        }
    }
}
