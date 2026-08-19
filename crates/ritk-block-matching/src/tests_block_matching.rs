//! Block-matching oracles: every test recovers a *known* applied displacement.

use super::*;

const DIMS: [usize; 3] = [1, 40, 40];

fn config() -> BlockMatchingConfig {
    BlockMatchingConfig {
        block_radius: [0, 4, 4],
        search_radius: [0, 5, 5],
    }
}

/// Deterministic white-noise texture, so a block has the variance normalized
/// correlation needs and adjacent samples are uncorrelated.
///
/// The mixing is the splitmix64 finalizer. A single multiply on a seed that is
/// linear in the coordinates is *not* enough: its high bits stay structured, so
/// neighbouring samples correlate and the correlation surface grows spurious
/// off-peak maxima. An earlier version of this fixture did exactly that, and
/// the sub-voxel test below located a peak 2 voxels away from the truth.
fn texture(z: usize, y: isize, x: isize) -> f32 {
    let seed = (z as i64)
        .wrapping_mul(7919)
        .wrapping_add((y as i64).wrapping_mul(104_729))
        .wrapping_add((x as i64).wrapping_mul(15_485_863)) as u64;
    let mut v = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 31;
    ((v >> 11) as f64 / (1_u64 << 53) as f64) as f32
}

/// Build an image whose content is `texture` sampled at an integer offset, so
/// `moving` is `fixed` translated by exactly `shift`.
fn shifted_image(shift: [isize; 3]) -> Vec<f32> {
    let mut out = vec![0.0_f32; DIMS[0] * DIMS[1] * DIMS[2]];
    for z in 0..DIMS[0] {
        for y in 0..DIMS[1] {
            for x in 0..DIMS[2] {
                out[(z * DIMS[1] + y) * DIMS[2] + x] =
                    texture(z, y as isize - shift[1], x as isize - shift[2]);
            }
        }
    }
    out
}

/// The core oracle: an exact integer translation must be recovered exactly.
///
/// The images are the same texture sampled at an offset, so the true peak is a
/// perfect correlation of 1 at exactly that offset. Any error in the search
/// indexing, the offset sign, or the axis order shows up here as a wrong
/// displacement rather than an approximate one.
#[test]
fn recovers_an_exact_integer_translation() {
    let fixed = shifted_image([0, 0, 0]);
    for shift in [[0_isize, 0, 0], [0, 3, 0], [0, 0, -2], [0, -4, 3]] {
        let moving = shifted_image(shift);
        let result = match_block(
            &fixed,
            &moving,
            DIMS,
            [0, 20, 20],
            config(),
            SubpixelRefinement::None,
        )
        .expect("match");

        assert_eq!(
            result.displacement,
            [shift[0] as f64, shift[1] as f64, shift[2] as f64],
            "shift {shift:?} must be recovered exactly"
        );
        assert!(
            result.peak_similarity > 0.999,
            "an exact translation must correlate ~1, got {}",
            result.peak_similarity
        );
    }
}

/// Sub-voxel refinement must not meaningfully move an exact integer match.
///
/// Note the correlation surface is *not* exactly symmetric about a perfect
/// peak: the texture is random, so `s₋` and `s₊` differ slightly even when the
/// true displacement is exactly on the grid. A refinement offset of exactly
/// zero is therefore the wrong expectation, and an earlier version of this test
/// asserting it failed at ~0.002–0.006 voxels.
///
/// What must hold is that the residual stays far below the sub-voxel precision
/// the method claims. Parabolic peak estimation on speckle is good to roughly a
/// tenth of a voxel, so a residual an order of magnitude below that — 0.05 —
/// cannot masquerade as signal. This matters because strain is the *spatial
/// derivative* of displacement: a residual comparable to the real
/// block-to-block variation would manufacture a strain field out of nothing.
#[test]
fn refinement_does_not_disturb_an_exact_match() {
    let fixed = shifted_image([0, 0, 0]);
    let moving = shifted_image([0, 2, -3]);
    for refinement in [SubpixelRefinement::Parabolic, SubpixelRefinement::Cosine] {
        let result =
            match_block(&fixed, &moving, DIMS, [0, 20, 20], config(), refinement).expect("match");
        assert!(
            (result.displacement[1] - 2.0).abs() < 0.05
                && (result.displacement[2] + 3.0).abs() < 0.05,
            "{refinement:?} moved an exact match to {:?}",
            result.displacement
        );
    }
}

/// A sub-voxel shift must land between the neighbouring integers and on the
/// correct side, which integer-only matching cannot do.
///
/// The image is built by linear interpolation of the texture at a half-voxel
/// offset, so the true displacement is 0.5. The assertion is deliberately loose
/// on magnitude — peak-locking bias is real and expected — but strict on the
/// two things a correct estimator must get right: it must be strictly between
/// the integers, and strictly better than the integer answer.
#[test]
fn subvoxel_shift_lands_between_the_integers() {
    let fixed = shifted_image([0, 0, 0]);
    let mut moving = vec![0.0_f32; DIMS[0] * DIMS[1] * DIMS[2]];
    for z in 0..DIMS[0] {
        for y in 0..DIMS[1] {
            for x in 0..DIMS[2] {
                let a = texture(z, y as isize, x as isize - 1);
                let b = texture(z, y as isize, x as isize);
                moving[(z * DIMS[1] + y) * DIMS[2] + x] = 0.5 * (a + b);
            }
        }
    }

    let integer = match_block(
        &fixed,
        &moving,
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
    )
    .expect("match");
    let refined = match_block(
        &fixed,
        &moving,
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::Parabolic,
    )
    .expect("match");

    assert_eq!(
        integer.displacement[2].fract(),
        0.0,
        "the unrefined estimate must stay on the grid"
    );
    let dx = refined.displacement[2];
    assert!(
        dx > 0.0 && dx < 1.0,
        "a half-voxel shift must refine strictly between 0 and 1, got {dx}"
    );
    assert!(
        (dx - 0.5).abs() < (integer.displacement[2] - 0.5).abs(),
        "refinement must beat the integer estimate: {dx} vs {}",
        integer.displacement[2]
    );
}

/// The metric image is the documented seam, so its shape and centre must mean
/// what the docs say: the centre entry is the null displacement.
#[test]
fn metric_image_centre_is_the_null_displacement() {
    let fixed = shifted_image([0, 0, 0]);
    let surface = metric_image(
        &fixed,
        &fixed,
        DIMS,
        [0, 20, 20],
        config(),
        BlockMetric::NormalizedCrossCorrelation,
    )
    .expect("metric image");

    assert_eq!(surface.extent, [1, 11, 11]);
    // Comparing an image with itself: the null offset is a perfect match.
    let centre = surface.at(0, 5, 5);
    assert!(
        (centre - 1.0).abs() < 1.0e-12,
        "centre must be the null displacement with NCC 1, got {centre}"
    );
}

/// Normalized cross-correlation is invariant to affine intensity change; that
/// invariance is why it is used on ultrasound, where gain and depth vary.
#[test]
fn correlation_is_invariant_to_gain_and_offset() {
    let fixed = shifted_image([0, 0, 0]);
    let moving: Vec<f32> = shifted_image([0, 2, 1])
        .iter()
        .map(|&v| 3.5 * v + 12.0)
        .collect();

    let result = match_block(
        &fixed,
        &moving,
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
    )
    .expect("match");
    assert_eq!(result.displacement, [0.0, 2.0, 1.0]);
    assert!(
        result.peak_similarity > 0.999,
        "gain and offset must not reduce correlation, got {}",
        result.peak_similarity
    );
}

/// A constant block has no variance, so correlation is undefined. Returning a
/// displacement there would be indistinguishable from a real match.
#[test]
fn refuses_a_featureless_block() {
    let flat = vec![4.0_f32; DIMS[0] * DIMS[1] * DIMS[2]];
    assert!(match_block(
        &flat,
        &flat,
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None
    )
    .is_err());
}

#[test]
fn rejects_invalid_geometry_and_out_of_bounds_blocks() {
    let fixed = shifted_image([0, 0, 0]);
    // A flat axis is valid — a 2-D acquisition is a 3-D image with a singleton
    // axis — but an all-zero radius is not.
    assert!(BlockMatchingConfig {
        block_radius: [0, 0, 4],
        search_radius: [0, 5, 5],
    }
    .validate()
    .is_ok());
    assert!(BlockMatchingConfig {
        block_radius: [0, 0, 0],
        search_radius: [0, 5, 5],
    }
    .validate()
    .is_err());
    assert!(BlockMatchingConfig {
        block_radius: [0, 4, 4],
        search_radius: [0, 0, 0],
    }
    .validate()
    .is_err());

    // A block whose extent leaves the image is the caller's error.
    assert!(match_block(
        &fixed,
        &fixed,
        DIMS,
        [0, 1, 20],
        config(),
        SubpixelRefinement::None
    )
    .is_err());

    // Mismatched buffer length.
    assert!(match_block(
        &fixed,
        &fixed[..10],
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None
    )
    .is_err());
}

/// The matcher must give the same answer for the same data at either stored
/// precision, so a consumer is never forced through a narrowing conversion.
///
/// Correlation accumulates in `f64` either way, so the only difference is the
/// input rounding; on data that is exactly representable in `f32`, the two must
/// agree exactly. RF-domain consumers hold `f64` and their whole value is
/// sub-sample precision, which a forced `f32` round-trip would erode.
#[test]
fn matches_identically_in_f32_and_f64() {
    let fixed32 = shifted_image([0, 0, 0]);
    let moving32 = shifted_image([0, 2, -3]);
    let fixed64: Vec<f64> = fixed32.iter().map(|&v| f64::from(v)).collect();
    let moving64: Vec<f64> = moving32.iter().map(|&v| f64::from(v)).collect();

    for refinement in [
        SubpixelRefinement::None,
        SubpixelRefinement::Parabolic,
        SubpixelRefinement::Cosine,
    ] {
        let a = match_block(&fixed32, &moving32, DIMS, [0, 20, 20], config(), refinement)
            .expect("f32 match");
        let b = match_block(&fixed64, &moving64, DIMS, [0, 20, 20], config(), refinement)
            .expect("f64 match");
        assert_eq!(
            a.displacement, b.displacement,
            "{refinement:?} must agree across stored precision"
        );
        assert_eq!(a.peak_similarity, b.peak_similarity);
    }
}

/// The 1-D axial case a speckle tracker needs: a single line, block and search
/// on the fast axis only. This is the shape kwavers' elastography tracker uses,
/// and it must work through the same seam rather than a second implementation.
#[test]
fn tracks_a_one_dimensional_line() {
    const N: usize = 64;
    let line: Vec<f64> = (0..N)
        .map(|i| f64::from(texture(0, 0, i as isize)))
        .collect();
    // Same line shifted by +3 samples.
    let shifted: Vec<f64> = (0..N)
        .map(|i| f64::from(texture(0, 0, i as isize - 3)))
        .collect();

    let dims = [1, 1, N];
    let config = BlockMatchingConfig {
        block_radius: [0, 0, 6],
        search_radius: [0, 0, 5],
    };
    let result = match_block(
        &line,
        &shifted,
        dims,
        [0, 0, 32],
        config,
        SubpixelRefinement::None,
    )
    .expect("1-D match");
    assert_eq!(
        result.displacement,
        [0.0, 0.0, 3.0],
        "a 1-D line shift must be recovered on the fast axis"
    );
    assert!(result.peak_similarity > 0.999);
}
