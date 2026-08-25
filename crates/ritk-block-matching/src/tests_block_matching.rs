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

#[cfg(feature = "fft")]
fn assert_fft_matches_direct(centre: [usize; 3]) {
    let fixed = shifted_image([0, 0, 0]);
    let moving = shifted_image([0, 3, -2]);
    let direct = metric_image(
        &fixed,
        &moving,
        DIMS,
        centre,
        config(),
        BlockMetric::NormalizedCrossCorrelation,
    )
    .expect("direct metric image");
    let fft = metric_image_fft(&fixed, &moving, DIMS, centre, config(), FftPadding::Zero)
        .expect("FFT metric image");

    assert_eq!(direct.extent, fft.extent);
    assert_eq!(direct.search_radius, fft.search_radius);
    for (index, (&expected, &actual)) in direct.values.iter().zip(&fft.values).enumerate() {
        match (expected.is_finite(), actual.is_finite()) {
            (true, true) => assert!(
                (expected - actual).abs() < 1.0e-9,
                "FFT/direct NCC mismatch at {index}: {actual} vs {expected}"
            ),
            (false, false) => {}
            _ => panic!("FFT/direct finite-boundary mismatch at {index}: {actual} vs {expected}"),
        }
    }
}

/// Apollo's finite linear NCC must match the direct metric away from image
/// boundaries, including the full candidate surface rather than only its peak.
#[cfg(feature = "fft")]
#[test]
fn fft_ncc_matches_direct_metric_interior() {
    assert_fft_matches_direct([0, 20, 20]);
}

/// Zero padding is an implementation detail, not correlation evidence: the
/// FFT path must agree with the direct metric when negative candidates are
/// excluded at a finite image boundary.
#[cfg(feature = "fft")]
#[test]
fn fft_ncc_matches_direct_metric_at_boundary() {
    assert_fft_matches_direct([0, 8, 8]);
}

/// The FFT matcher must preserve the public displacement convention and peak
/// value for an exact translated texture.
#[cfg(feature = "fft")]
#[test]
fn fft_match_recovers_integer_translation() {
    let fixed = shifted_image([0, 0, 0]);
    let moving = shifted_image([0, 3, -2]);
    let direct = match_block(
        &fixed,
        &moving,
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
    )
    .expect("direct match");
    let fft = match_block_fft(
        &fixed,
        &moving,
        DIMS,
        [0, 20, 20],
        config(),
        SubpixelRefinement::None,
        FftPadding::Zero,
    )
    .expect("FFT match");

    assert_eq!(direct.displacement, [0.0, 3.0, -2.0]);
    assert_eq!(fft.displacement, direct.displacement);
    assert!((fft.peak_similarity - direct.peak_similarity).abs() < 1.0e-9);
}

#[cfg(feature = "fft")]
#[test]
fn fft_ncc_rejects_featureless_and_mismatched_inputs() {
    let flat = vec![1.0_f32; DIMS[0] * DIMS[1] * DIMS[2]];
    assert!(
        metric_image_fft(&flat, &flat, DIMS, [0, 20, 20], config(), FftPadding::Zero,).is_err()
    );
    assert!(metric_image_fft(
        &flat,
        &flat[..10],
        DIMS,
        [0, 20, 20],
        config(),
        FftPadding::Zero,
    )
    .is_err());
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

// ── US-023-D2: Volume-level pipeline and strain estimation ────────────────────

/// `track_volume` must recover a known integer displacement at every block
/// centre in a purely axial shift scenario.
#[test]
fn track_volume_recovers_integer_axial_shift() {
    // 3-D volume: [nz=1, ny=32, nx=32]. Apply shift (+2 in y, +1 in x).
    const DIMS3: [usize; 3] = [1, 32, 32];
    let fixed: Vec<f32> = {
        let mut v = vec![0.0f32; DIMS3[0] * DIMS3[1] * DIMS3[2]];
        for z in 0..DIMS3[0] {
            for y in 0..DIMS3[1] {
                for x in 0..DIMS3[2] {
                    v[(z * DIMS3[1] + y) * DIMS3[2] + x] = texture(z, y as isize, x as isize);
                }
            }
        }
        v
    };
    let moving: Vec<f32> = {
        let mut v = vec![0.0f32; DIMS3[0] * DIMS3[1] * DIMS3[2]];
        for z in 0..DIMS3[0] {
            for y in 0..DIMS3[1] {
                for x in 0..DIMS3[2] {
                    v[(z * DIMS3[1] + y) * DIMS3[2] + x] =
                        texture(z, y as isize - 2, x as isize - 1);
                }
            }
        }
        v
    };

    let config = BlockMatchingConfig {
        block_radius: [0, 4, 4],
        search_radius: [0, 3, 3],
    };
    let grid = BlockGrid::dense(config.block_radius);
    let field = track_volume(
        &fixed,
        &moving,
        DIMS3,
        config,
        grid,
        SubpixelRefinement::None,
    )
    .expect("ok");

    assert!(!field.is_empty(), "grid must yield at least one block");

    // Every block should recover [0, +2, +1] exactly.
    for (i, bd) in field.displacements.iter().enumerate() {
        assert_eq!(
            *bd,
            [0.0, 2.0, 1.0],
            "block {} at {:?} must recover [0,2,1]",
            i,
            field.centres[i]
        );
        assert!(
            field.peak_similarities[i] > 0.99,
            "block {} peak must be close to 1",
            i
        );
    }
}

/// `strain_from_displacement` must recover zero strain when all blocks have the
/// same displacement (rigid-body translation — no compression).
#[test]
fn strain_is_zero_for_uniform_translation() {
    // Fake field: 5 blocks along z, all with displacement +3 voxels axially.
    let centres: Vec<[usize; 3]> = (0..5).map(|k| [k * 9, 0, 0]).collect();
    let displacements: Vec<[f64; 3]> = vec![[3.0, 0.0, 0.0]; 5];
    let peak_similarities: Vec<f64> = vec![0.95; 5];
    let field = DisplacementField {
        centres,
        displacements,
        peak_similarities,
    };
    let strain = strain_from_displacement(&field, 9);
    for (i, &s) in strain.iter().enumerate() {
        assert!(
            s.abs() < 1.0e-12,
            "block {i}: expected 0 strain for uniform translation, got {s}"
        );
    }
}

#[test]
fn fallible_strain_rejects_malformed_fields_and_zero_stride() {
    let malformed = DisplacementField {
        centres: vec![[0, 0, 0]],
        displacements: vec![],
        peak_similarities: vec![1.0],
    };
    assert!(try_strain_from_displacement(&malformed, 1).is_err());

    let valid = DisplacementField {
        centres: vec![[0, 0, 0]],
        displacements: vec![[0.0, 0.0, 0.0]],
        peak_similarities: vec![1.0],
    };
    assert!(try_strain_from_displacement(&valid, 0).is_err());
}

/// `strain_from_displacement` must recover a known constant strain when the
/// displacement increases linearly with axial position.
///
/// For a pure axial compression with strain ε, the displacement at axial
/// position z is `d(z) = ε · z`. For blocks at axial stride `s`, the central
/// difference gives `(d(z+s) - d(z-s)) / (2s) = ε`, which is exact when the
/// displacement field is linear.
#[test]
fn strain_recovers_known_constant_strain() {
    let strain_truth = 0.02; // 2 % compression
    let stride = 9usize;
    let n = 7;

    let centres: Vec<[usize; 3]> = (0..n).map(|k| [k * stride, 0, 0]).collect();
    let displacements: Vec<[f64; 3]> = centres
        .iter()
        .map(|&[z, _, _]| [strain_truth * z as f64, 0.0, 0.0])
        .collect();
    let peak_similarities: Vec<f64> = vec![0.95; n];
    let field = DisplacementField {
        centres,
        displacements,
        peak_similarities,
    };
    let strain = strain_from_displacement(&field, stride);
    // Central difference is exact for a linear field; boundary estimates use
    // one-sided differences and are also exact.
    for (i, &s) in strain.iter().enumerate() {
        assert!(
            (s - strain_truth).abs() < 1.0e-10,
            "block {i}: expected strain {strain_truth}, got {s}"
        );
    }
}

#[test]
fn filtered_strain_skips_invalid_blocks_and_scales_gaps() {
    let field = DisplacementField {
        centres: vec![[0, 0, 0], [9, 0, 0], [18, 0, 0], [27, 0, 0]],
        displacements: vec![
            [0.0, 0.0, 0.0],
            [99.0, 0.0, 0.0],
            [0.18, 0.0, 0.0],
            [0.27, 0.0, 0.0],
        ],
        peak_similarities: vec![1.0, f64::NAN, 1.0, 1.0],
    };
    let strain = strain_from_displacement_filtered(&field, 9, 0.5).expect("filtered strain");

    assert!((strain[0] - 0.01).abs() < 1.0e-12);
    assert!(strain[1].is_nan());
    assert!((strain[2] - 0.01).abs() < 1.0e-12);
    assert!((strain[3] - 0.01).abs() < 1.0e-12);
    assert!(strain_from_displacement_filtered(&field, 0, 0.5).is_err());
    assert!(strain_from_displacement_filtered(&field, 9, 1.1).is_err());
}

/// A single-block field has no neighbour for finite differences; the only
/// correct answer is zero (no gradient computable).
#[test]
fn strain_for_single_block_field_is_zero() {
    let field = DisplacementField {
        centres: vec![[5, 0, 0]],
        displacements: vec![[3.0, 0.0, 0.0]],
        peak_similarities: vec![0.9],
    };
    let strain = strain_from_displacement(&field, 5);
    assert_eq!(strain.len(), 1);
    assert!(strain[0].abs() < 1.0e-12);
}

/// `BlockGrid::dense` must enumerate a non-empty set of centres for a
/// volume that fits at least one block.
#[test]
fn block_grid_dense_enumerates_centres() {
    let config = BlockMatchingConfig {
        block_radius: [0, 4, 4],
        search_radius: [0, 3, 3],
    };
    let grid = BlockGrid::dense(config.block_radius);
    let centres = grid.centres([1, 32, 32], &config);
    assert!(!centres.is_empty());
    // Every centre must be at least block_radius away from each image boundary.
    for &[z, y, x] in &centres {
        assert!(z >= config.block_radius[0]);
        assert!(y >= config.block_radius[1]);
        assert!(x >= config.block_radius[2]);
        assert!(z + config.block_radius[0] < 1);
        assert!(y + config.block_radius[1] < 32);
        assert!(x + config.block_radius[2] < 32);
    }
}

#[test]
fn block_grid_validates_stride_and_overflow() {
    assert!(BlockGrid::try_dense([usize::MAX, 0, 0]).is_err());
    assert!(BlockGrid { stride: [0, 1, 1] }.validate().is_err());

    let grid = BlockGrid { stride: [1, 1, 1] };
    let oversized = BlockMatchingConfig {
        block_radius: [usize::MAX, usize::MAX, usize::MAX],
        search_radius: [1, 1, 1],
    };
    assert!(grid.centres([1, 1, 1], &oversized).is_empty());
}

#[test]
fn displacement_field_validates_and_masks_confidence() {
    let field = DisplacementField {
        centres: vec![[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        displacements: vec![[1.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0], [3.0, 0.0, 0.0]],
        peak_similarities: vec![0.95, 0.99, f64::NAN],
    };

    assert!(field.validate().is_ok());
    assert_eq!(
        field.valid_mask(0.9).expect("valid mask"),
        vec![true, false, false]
    );
    assert_eq!(
        field.valid_mask(0.99).expect("valid mask"),
        vec![false, false, false]
    );
    assert!(field.valid_mask(-0.1).is_err());
    assert!(field.valid_mask(f64::NAN).is_err());

    let malformed = DisplacementField {
        centres: vec![[0, 0, 0]],
        displacements: Vec::new(),
        peak_similarities: vec![1.0],
    };
    assert!(malformed.validate().is_err());
    assert!(malformed.valid_mask(0.5).is_err());
}

/// Sample the same physical texture on one pyramid level. The moving image is
/// translated in finest-resolution voxels, so the expected level displacement
/// is the physical shift divided by `scale`.
fn pyramid_image(scale: usize, shift: [isize; 3]) -> Vec<f32> {
    let dims = [1, 40 / scale, 40 / scale];
    let mut image = vec![0.0_f32; dims[0] * dims[1] * dims[2]];
    for z in 0..dims[0] {
        for y in 0..dims[1] {
            for x in 0..dims[2] {
                image[(z * dims[1] + y) * dims[2] + x] = texture(
                    z * scale,
                    y as isize * scale as isize - shift[1],
                    x as isize * scale as isize - shift[2],
                );
            }
        }
    }

    image
}

/// A coarse level must find the broad motion and the fine level must search
/// around its propagated moving centre rather than restarting at zero offset.
#[test]
fn a_singleton_axis_survives_every_pyramid_level() {
    // A 2-D acquisition is a volume with one out-of-plane sample. That axis
    // carries no resolution to trade away, so it must stay 1 at every scale
    // rather than being divided to zero or rejected as indivisible.
    let fixed = pyramid_image(1, [0, 0, 0]);
    let moving = pyramid_image(1, [0, 4, 2]);

    for owned in [
        OwnedPyramid::nearest(&fixed, &moving, [1, 40, 40], &[4, 2, 1]).expect("nearest pyramid"),
        OwnedPyramid::min_max(&fixed, &moving, [1, 40, 40], &[4, 2, 1]).expect("min_max pyramid"),
    ] {
        let levels = owned.levels();
        assert_eq!(levels.len(), 3);
        for (index, (level, expected_plane)) in levels.iter().zip([10, 20, 40]).enumerate() {
            assert_eq!(level.dims[0], 1, "level {index} lost the singleton axis");
            assert_eq!(level.dims[1], expected_plane, "level {index}");
            // min_max stores two planes per level along the last axis, so the
            // buffer length rather than the extent is what must stay consistent.
            assert_eq!(
                level.fixed.len(),
                level.dims[0] * level.dims[1] * level.dims[2],
                "level {index} buffer does not match its own dims"
            );
        }
    }
}

#[test]
fn an_indivisible_non_singleton_extent_is_still_rejected() {
    // The singleton relaxation must not become "any extent goes": an axis with
    // real resolution that the scale cannot divide is still an error, because
    // the level grid would silently drop samples off the end.
    let fixed = pyramid_image(1, [0, 0, 0]);
    let moving = pyramid_image(1, [0, 4, 2]);
    let err = OwnedPyramid::nearest(&fixed, &moving, [1, 40, 40], &[3, 1])
        .expect_err("40 is not divisible by 3");
    assert!(
        err.to_string().contains("not divisible"),
        "unexpected error: {err}"
    );
}

#[test]
fn pyramid_matching_propagates_coarse_displacement() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 8, 8], 3).expect("valid plan");
    let coarse_fixed = pyramid_image(4, [0, 0, 0]);
    let coarse_moving = pyramid_image(4, [0, 8, 4]);
    let middle_fixed = pyramid_image(2, [0, 0, 0]);
    let middle_moving = pyramid_image(2, [0, 8, 4]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 8, 4]);

    let result = plan
        .match_pyramid(
            &[
                PyramidLevel {
                    fixed: &coarse_fixed,
                    moving: &coarse_moving,
                    dims: [1, 10, 10],
                },
                PyramidLevel {
                    fixed: &middle_fixed,
                    moving: &middle_moving,
                    dims: [1, 20, 20],
                },
                PyramidLevel {
                    fixed: &fine_fixed,
                    moving: &fine_moving,
                    dims: [1, 40, 40],
                },
            ],
            [0, 20, 20],
            SubpixelRefinement::None,
        )
        .expect("pyramid match");

    assert_eq!(result.displacement, [0.0, 8.0, 4.0]);
    assert_eq!(result.levels.len(), 3);
    assert_eq!(result.levels[0].scale, 4);
    assert_eq!(result.levels[0].displacement, [0.0, 2.0, 1.0]);
    assert_eq!(result.levels[1].moving_centre, [0, 14, 12]);
    assert_eq!(result.levels[1].displacement, [0.0, 4.0, 2.0]);
    assert_eq!(result.levels[2].moving_centre, [0, 28, 24]);
    assert!(result.peak_similarity > 0.999);
}

#[cfg(feature = "fft")]
#[test]
fn fft_pyramid_matches_direct_propagation_and_diagnostics() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 8, 8], 3).expect("valid plan");
    let coarse_fixed = pyramid_image(4, [0, 0, 0]);
    let coarse_moving = pyramid_image(4, [0, 8, 4]);
    let middle_fixed = pyramid_image(2, [0, 0, 0]);
    let middle_moving = pyramid_image(2, [0, 8, 4]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 8, 4]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 10, 10],
        },
        PyramidLevel {
            fixed: &middle_fixed,
            moving: &middle_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];

    let direct = plan
        .match_pyramid(&pyramid, [0, 20, 20], SubpixelRefinement::None)
        .expect("direct pyramid match");
    let fft = plan
        .match_pyramid_fft(
            &pyramid,
            [0, 20, 20],
            SubpixelRefinement::None,
            FftPadding::Zero,
        )
        .expect("FFT pyramid match");

    assert_eq!(fft.displacement, direct.displacement);
    assert!((fft.peak_similarity - direct.peak_similarity).abs() < 1.0e-9);
    assert_eq!(fft.levels.len(), direct.levels.len());
    for (index, (expected, actual)) in direct.levels.iter().zip(&fft.levels).enumerate() {
        assert_eq!(actual.scale, expected.scale, "level {index} scale");
        assert_eq!(
            actual.fixed_centre, expected.fixed_centre,
            "level {index} fixed centre"
        );
        assert_eq!(
            actual.moving_centre, expected.moving_centre,
            "level {index} moving centre"
        );
        assert_eq!(
            actual.displacement, expected.displacement,
            "level {index} displacement"
        );
        assert!(
            (actual.peak_similarity - expected.peak_similarity).abs() < 1.0e-9,
            "level {index} peak differs: {} vs {}",
            actual.peak_similarity,
            expected.peak_similarity
        );
    }
}

#[cfg(feature = "fft")]
#[test]
fn fft_pyramid_volume_matches_direct_field() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 4, 2]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 4, 2]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];
    let grid = BlockGrid::dense([0, 4, 4]);
    let direct = plan
        .track_volume_pyramid(&pyramid, grid, SubpixelRefinement::None)
        .expect("direct pyramid volume match");
    let fft = plan
        .track_volume_pyramid_fft(&pyramid, grid, SubpixelRefinement::None, FftPadding::Zero)
        .expect("FFT pyramid volume match");

    assert_eq!(fft.centres, direct.centres);
    assert_eq!(fft.displacements, direct.displacements);
    assert_eq!(fft.peak_similarities.len(), direct.peak_similarities.len());
    for (index, (&expected, &actual)) in direct
        .peak_similarities
        .iter()
        .zip(&fft.peak_similarities)
        .enumerate()
    {
        assert!(
            expected.is_nan() == actual.is_nan()
                && (expected.is_nan() || (expected - actual).abs() < 1.0e-9),
            "block {index} peak differs: {actual} vs {expected}"
        );
    }
}

#[test]
fn pyramid_matching_rejects_a_level_count_mismatch() {
    let plan = MultiResolutionSearch::new([0, 2, 2], [0, 2, 2], 2).expect("valid plan");
    let image = pyramid_image(1, [0, 0, 0]);
    assert!(plan
        .match_pyramid(
            &[PyramidLevel {
                fixed: &image,
                moving: &image,
                dims: [1, 40, 40],
            }],
            [0, 20, 20],
            SubpixelRefinement::None,
        )
        .is_err());
}

#[test]
fn pyramid_regularization_uses_finest_confidence_and_preserves_diagnostics() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 8, 8], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 8, 4]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 8, 4]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];
    let prior =
        BayesianDisplacementPrior::new([0.0; 3], 1.0, 1.0, 0.9).expect("valid Bayesian prior");

    let raw = plan
        .match_pyramid(&pyramid, [0, 20, 20], SubpixelRefinement::None)
        .expect("raw pyramid match");
    let regularized = plan
        .match_pyramid_regularized(&pyramid, [0, 20, 20], SubpixelRefinement::None, &prior)
        .expect("regularized pyramid match");

    assert_eq!(raw.displacement, [0.0, 8.0, 4.0]);
    assert!(raw.peak_similarity > 0.999);
    let confidence = raw.peak_similarity;
    let observation_weight = confidence * confidence / (1.0 + confidence * confidence);
    for axis in 0..3 {
        let expected = raw.displacement[axis] * observation_weight;
        assert!(
            (regularized.displacement[axis] - expected).abs() < 1.0e-10,
            "axis {axis}: expected posterior {expected}, got {}",
            regularized.displacement[axis]
        );
    }

    assert_eq!(regularized.peak_similarity, raw.peak_similarity);
    assert_eq!(regularized.levels, raw.levels);
}

/// Sample the texture with an axial compression applied: the fixed image is
/// `texture(z, y, x)` and the moving image is the same field resampled at a
/// linearly increasing axial shift `d(z) = strain · (z - z0)`, which is what a
/// uniform compression of the tissue produces. `z0` is the reference depth at
/// which displacement is zero (the transducer face).
fn compressed_image(z0: isize, strain: f64, z_scale: usize, y_scale: usize) -> Vec<f32> {
    let dims = [1, 40 / y_scale, 40 / z_scale];
    let mut image = vec![0.0_f32; dims[0] * dims[1] * dims[2]];
    for z in 0..dims[0] {
        for y in 0..dims[1] {
            for x in 0..dims[2] {
                // The compressed sample at depth x came from depth x' = x - d(x)
                // in the uncompressed reference.
                let x_fine = x as isize * z_scale as isize;
                let displacement = strain * (x_fine - z0) as f64;
                let source_x = x_fine as f64 - displacement;
                let source = source_x.round() as isize;
                let value = if (0..40).contains(&source) {
                    texture(0, y as isize * y_scale as isize, source)
                } else {
                    0.0 // outside the reference field
                };
                image[(z * dims[1] + y) * dims[2] + x] = value;
            }
        }
    }

    image
}

/// End-to-end D2 oracle: the pipeline must recover a known constant axial
/// strain from a simulated compression sequence within a derived bound.
///
/// A 2% uniform compression produces a displacement field `d(z) = 0.02·(z-z0)`
/// and a constant strain of 0.02. Block matching with a dense grid, followed
/// by `strain_from_displacement`, must recover that strain to within the
/// quantization error of the half-voxel resampling (the moving image is
/// sampled by rounding the source coordinate, so the displacement is exact at
/// the half-sample level and the strain is exact up to block-boundary effects).
#[test]
fn pipeline_recovers_known_compression_strain() {
    let strain_truth = 0.02;
    let z0 = 20_isize; // reference depth at the image centre

    let fixed = pyramid_image(1, [0, 0, 0]);
    let moving = compressed_image(z0, strain_truth, 1, 1);

    let config = BlockMatchingConfig {
        block_radius: [0, 4, 4],
        search_radius: [0, 5, 5],
    };
    let pipeline = DisplacementPipeline {
        metric: PipelineMetric::Direct,
        refinement: SubpixelRefinement::None,
        grid: BlockGrid::dense(config.block_radius),
        stages: PipelineStages::default(),
    };
    let result = pipeline
        .run(&fixed, &moving, [1, 40, 40], config)
        .expect("pipeline run");

    // The strain at every interior block must approximate the applied strain.
    let strain = strain_from_displacement(&result.field, pipeline.grid.stride[0]);
    let mut interior = 0;
    let mut error_sum = 0.0;
    for (i, &s) in strain.iter().enumerate() {
        // Skip blocks at the axial edges where the one-sided difference and the
        // truncated compression field make the estimate unreliable.
        if result.field.centres[i][2] < 12 || result.field.centres[i][2] > 28 {
            continue;
        }
        interior += 1;
        error_sum += (s - strain_truth).abs();
    }

    assert!(interior >= 2, "expected interior blocks, got {interior}");
    let mean_error = error_sum / interior as f64;
    // The rounding of the resampled source coordinate bounds the per-block
    // displacement error to 0.5 voxel; over the axial block stride (9 voxels)
    // that is a strain error of ~0.056. Interior blocks average much better;
    // assert a derived, non-tuned bound well below the gross error scale.
    assert!(
        mean_error < 0.05,
        "mean strain error {mean_error} exceeds the derived bound for strain {strain_truth}"
    );
}

/// The end-to-end pipeline with a strain window must smooth an outlier-heavy
/// field back toward the true strain without erasing a genuine displacement
/// gradient.
#[test]
fn pipeline_with_least_squares_prior_keeps_linear_strain() {
    let strain_truth = 0.01;
    let z0 = 20_isize;
    let fixed = pyramid_image(1, [0, 0, 0]);
    let moving = compressed_image(z0, strain_truth, 1, 1);

    let config = BlockMatchingConfig {
        block_radius: [0, 4, 4],
        search_radius: [0, 5, 5],
    };
    let window = LeastSquaresDisplacementPrior::new(5, 0.8).expect("valid strain window");
    let pipeline = DisplacementPipeline {
        metric: PipelineMetric::Direct,
        refinement: SubpixelRefinement::None,
        grid: BlockGrid::dense(config.block_radius),
        stages: PipelineStages {
            bayesian_prior: None,
            least_squares_prior: Some(window),
            minimum_peak_similarity: Some(0.0),
        },
    };
    let result = pipeline
        .run(&fixed, &moving, [1, 40, 40], config)
        .expect("pipeline run");

    // With the strain window enabled the pipeline reports strain directly.
    let strain = result.axial_strain.expect("strain window produces strain");
    let mut interior = 0;
    let mut error_sum = 0.0;
    for (i, &s) in strain.iter().enumerate() {
        if result.field.centres[i][2] < 12 || result.field.centres[i][2] > 28 {
            continue;
        }
        interior += 1;
        error_sum += (s - strain_truth).abs();
    }

    assert!(interior >= 2);
    let mean_error = error_sum / interior as f64;
    assert!(
        mean_error < 0.05,
        "regularized mean strain error {mean_error} exceeds the derived bound"
    );
}

#[test]
fn pipeline_rejects_malformed_public_stage_fields_before_matching() {
    let pipeline = DisplacementPipeline {
        metric: PipelineMetric::Direct,
        refinement: SubpixelRefinement::None,
        grid: BlockGrid::dense([0, 1, 1]),
        stages: PipelineStages {
            bayesian_prior: Some(BayesianDisplacementPrior {
                mean: [0.0; 3],
                prior_variance: 0.0,
                observation_variance: 1.0,
                minimum_peak_similarity: 0.5,
            }),
            least_squares_prior: Some(LeastSquaresDisplacementPrior {
                window: 2,
                regularization_strength: 0.5,
            }),
            minimum_peak_similarity: Some(f64::NAN),
        },
    };
    assert!(pipeline.stages.validate().is_err());
    assert!(pipeline
        .run::<f32>(
            &[],
            &[],
            [1, 1, 1],
            BlockMatchingConfig {
                block_radius: [0, 1, 1],
                search_radius: [0, 1, 1],
            }
        )
        .is_err());
}

#[test]
fn pipeline_runs_a_pyramid_and_applies_post_processing() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 4, 2]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 4, 2]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];
    let prior =
        BayesianDisplacementPrior::new([1.0, 0.0, 0.0], 1.0, 1.0, 0.9).expect("valid prior");
    let pipeline = DisplacementPipeline {
        metric: PipelineMetric::Direct,
        refinement: SubpixelRefinement::None,
        grid: BlockGrid::dense([0, 4, 4]),
        stages: PipelineStages {
            bayesian_prior: Some(prior),
            least_squares_prior: None,
            minimum_peak_similarity: None,
        },
    };

    let raw = plan
        .track_volume_pyramid(&pyramid, pipeline.grid, SubpixelRefinement::None)
        .expect("raw pyramid field");
    let expected = prior.regularize(&raw);
    let result = pipeline
        .run_pyramid_with_diagnostics(&plan, &pyramid)
        .expect("pipeline pyramid run");
    let with_diagnostics = pipeline
        .run_pyramid_with_diagnostics(&plan, &pyramid)
        .expect("diagnostic pipeline pyramid run");

    assert_eq!(result.field, with_diagnostics.field);
    assert_eq!(with_diagnostics.diagnostics.centres, raw.centres);
    assert_eq!(
        with_diagnostics.diagnostics.displacements,
        raw.displacements
    );
    assert_eq!(
        with_diagnostics.diagnostics.peak_similarities,
        raw.peak_similarities
    );
    assert_eq!(result.field.centres, expected.centres);
    assert_eq!(result.field.peak_similarities, expected.peak_similarities);
    assert_eq!(result.field.displacements, expected.displacements);
    assert!(result.axial_strain.is_none());
}

#[test]
fn pipeline_owned_pyramid_matches_explicit_levels() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let fixed = pyramid_image(1, [0, 0, 0]);
    let moving = pyramid_image(1, [0, 4, 2]);
    let owned =
        OwnedPyramid::nearest(&fixed, &moving, [1, 40, 40], &[2, 1]).expect("valid owned pyramid");
    let levels = owned.levels();
    let pipeline = DisplacementPipeline {
        metric: PipelineMetric::Direct,
        refinement: SubpixelRefinement::None,
        grid: BlockGrid::dense([0, 4, 4]),
        stages: PipelineStages::default(),
    };

    let explicit = pipeline
        .run_pyramid(&plan, &levels)
        .expect("explicit-level pipeline run");
    let adapted = pipeline
        .run_owned_pyramid(&plan, &owned)
        .expect("owned-pyramid pipeline run");

    assert_eq!(adapted.field, explicit.field);
    assert_eq!(adapted.axial_strain, explicit.axial_strain);

    let explicit_diagnostics = pipeline
        .run_pyramid_with_diagnostics(&plan, &levels)
        .expect("explicit diagnostic pipeline run");
    let owned_diagnostics = pipeline
        .run_owned_pyramid_with_diagnostics(&plan, &owned)
        .expect("owned diagnostic pipeline run");
    assert_eq!(owned_diagnostics.field, explicit_diagnostics.field);
    assert_eq!(
        owned_diagnostics.diagnostics,
        explicit_diagnostics.diagnostics
    );
}

#[cfg(feature = "fft")]
#[test]
fn pipeline_owned_fft_pyramid_matches_direct_adapter() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let fixed = pyramid_image(1, [0, 0, 0]);
    let moving = pyramid_image(1, [0, 4, 2]);
    let owned =
        OwnedPyramid::nearest(&fixed, &moving, [1, 40, 40], &[2, 1]).expect("valid owned pyramid");
    let grid = BlockGrid::dense([0, 4, 4]);
    let direct = DisplacementPipeline {
        metric: PipelineMetric::Direct,
        refinement: SubpixelRefinement::None,
        grid,
        stages: PipelineStages::default(),
    }
    .run_owned_pyramid(&plan, &owned)
    .expect("direct owned-pyramid run");
    let fft = DisplacementPipeline {
        metric: PipelineMetric::Fft,
        refinement: SubpixelRefinement::None,
        grid,
        stages: PipelineStages::default(),
    }
    .run_owned_pyramid(&plan, &owned)
    .expect("FFT owned-pyramid run");

    assert_eq!(fft.field.centres, direct.field.centres);
    assert_eq!(fft.field.displacements, direct.field.displacements);
    for (index, (&expected, &actual)) in direct
        .field
        .peak_similarities
        .iter()
        .zip(&fft.field.peak_similarities)
        .enumerate()
    {
        assert!(
            expected.is_nan() == actual.is_nan()
                && (expected.is_nan() || (expected - actual).abs() < 1.0e-9),
            "owned pipeline block {index} peak differs: {actual} vs {expected}"
        );
    }
}

#[cfg(feature = "fft")]
#[test]
fn pipeline_fft_pyramid_matches_direct_pipeline() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 4, 2]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 4, 2]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];
    let grid = BlockGrid::dense([0, 4, 4]);
    let direct = DisplacementPipeline {
        metric: PipelineMetric::Direct,
        refinement: SubpixelRefinement::None,
        grid,
        stages: PipelineStages::default(),
    }
    .run_pyramid_with_diagnostics(&plan, &pyramid)
    .expect("direct pipeline pyramid run");
    let fft = DisplacementPipeline {
        metric: PipelineMetric::Fft,
        refinement: SubpixelRefinement::None,
        grid,
        stages: PipelineStages::default(),
    }
    .run_pyramid_with_diagnostics(&plan, &pyramid)
    .expect("FFT pipeline pyramid run");

    assert_eq!(fft.field.centres, direct.field.centres);
    assert_eq!(fft.field.displacements, direct.field.displacements);
    for (index, (&expected, &actual)) in direct
        .field
        .peak_similarities
        .iter()
        .zip(&fft.field.peak_similarities)
        .enumerate()
    {
        assert!(
            expected.is_nan() == actual.is_nan()
                && (expected.is_nan() || (expected - actual).abs() < 1.0e-9),
            "pipeline block {index} peak differs: {actual} vs {expected}"
        );
    }
}

#[test]
fn pyramid_volume_tracking_propagates_shift_across_all_valid_blocks() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 4, 2]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 4, 2]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];

    let field = plan
        .track_volume_pyramid(
            &pyramid,
            BlockGrid::dense([0, 4, 4]),
            SubpixelRefinement::None,
        )
        .expect("pyramid volume match");

    let valid = field
        .peak_similarities
        .iter()
        .filter(|peak| peak.is_finite())
        .count();
    assert!(valid >= 9, "expected valid pyramid blocks, got {valid}");
    for (index, (&displacement, &peak)) in field
        .displacements
        .iter()
        .zip(&field.peak_similarities)
        .enumerate()
    {
        if peak.is_finite() {
            assert_eq!(
                displacement,
                [0.0, 4.0, 2.0],
                "block {index} at {:?} missed the known shift",
                field.centres[index]
            );
        } else {
            assert_eq!(displacement, [0.0; 3]);
        }
    }
}

#[test]
fn pyramid_volume_diagnostics_align_with_batch_field() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 4, 2]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 4, 2]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];
    let grid = BlockGrid::dense([0, 4, 4]);
    let field = plan
        .track_volume_pyramid(&pyramid, grid, SubpixelRefinement::None)
        .expect("batch field");
    let diagnostics = plan
        .track_volume_pyramid_diagnostics(&pyramid, grid, SubpixelRefinement::None)
        .expect("diagnostic field");

    assert_eq!(diagnostics.centres, field.centres);
    assert_eq!(diagnostics.displacements, field.displacements);
    assert_eq!(diagnostics.peak_similarities, field.peak_similarities);
    for (index, (&peak, levels)) in diagnostics
        .peak_similarities
        .iter()
        .zip(&diagnostics.level_diagnostics)
        .enumerate()
    {
        if peak.is_finite() {
            assert_eq!(
                levels.as_ref().expect("valid block").len(),
                2,
                "block {index}"
            );
        } else {
            assert!(
                levels.is_none(),
                "skipped block {index} must have no diagnostics"
            );
        }
    }
}

#[test]
fn pyramid_diagnostics_validate_before_field_projection() {
    let malformed = PyramidDisplacementField {
        centres: vec![[0, 1, 1]],
        displacements: vec![[0.0, 0.0, 0.0]],
        peak_similarities: vec![1.0],
        level_diagnostics: vec![Some(vec![PyramidLevelDisplacement {
            scale: 0,
            fixed_centre: [0, 1, 1],
            moving_centre: [0, 1, 1],
            displacement: [0.0, 0.0, 0.0],
            peak_similarity: 1.0,
        }])],
    };
    assert!(malformed.validate().is_err());
    assert!(malformed.try_as_field().is_err());

    let misaligned = PyramidDisplacementField {
        centres: vec![[0, 1, 1]],
        displacements: vec![],
        peak_similarities: vec![1.0],
        level_diagnostics: vec![None],
    };
    assert!(misaligned.validate().is_err());
    assert!(misaligned.try_as_field().is_err());
}

#[cfg(feature = "fft")]
#[test]
fn fft_pyramid_diagnostics_match_direct_diagnostics() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 4, 2]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 4, 2]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];
    let grid = BlockGrid::dense([0, 4, 4]);
    let direct = plan
        .track_volume_pyramid_diagnostics(&pyramid, grid, SubpixelRefinement::None)
        .expect("direct diagnostics");
    let fft = plan
        .track_volume_pyramid_fft_diagnostics(
            &pyramid,
            grid,
            SubpixelRefinement::None,
            FftPadding::Zero,
        )
        .expect("FFT diagnostics");

    assert_eq!(fft.centres, direct.centres);
    assert_eq!(fft.displacements, direct.displacements);
    for (index, (expected, actual)) in direct
        .level_diagnostics
        .iter()
        .zip(&fft.level_diagnostics)
        .enumerate()
    {
        match (expected, actual) {
            (None, None) => {}
            (Some(expected), Some(actual)) => {
                assert_eq!(actual.len(), expected.len(), "block {index} level count");
                for (level, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                    assert_eq!(actual.fixed_centre, expected.fixed_centre);
                    assert_eq!(actual.moving_centre, expected.moving_centre);
                    assert_eq!(actual.displacement, expected.displacement);
                    assert!(
                        (actual.peak_similarity - expected.peak_similarity).abs() < 1.0e-9,
                        "block {index}, level {level} peak differs"
                    );
                }
            }
            _ => panic!("block {index} direct/FFT validity differs"),
        }
    }
}

#[test]
fn pyramid_volume_regularization_uses_each_block_confidence() {
    let plan = MultiResolutionSearch::new([0, 4, 4], [0, 4, 4], 2).expect("valid plan");
    let coarse_fixed = pyramid_image(2, [0, 0, 0]);
    let coarse_moving = pyramid_image(2, [0, 4, 2]);
    let fine_fixed = pyramid_image(1, [0, 0, 0]);
    let fine_moving = pyramid_image(1, [0, 4, 2]);
    let pyramid = [
        PyramidLevel {
            fixed: &coarse_fixed,
            moving: &coarse_moving,
            dims: [1, 20, 20],
        },
        PyramidLevel {
            fixed: &fine_fixed,
            moving: &fine_moving,
            dims: [1, 40, 40],
        },
    ];
    let prior =
        BayesianDisplacementPrior::new([1.0, 0.0, 0.0], 1.0, 1.0, 0.9).expect("valid prior");
    let raw = plan
        .track_volume_pyramid(
            &pyramid,
            BlockGrid::dense([0, 4, 4]),
            SubpixelRefinement::None,
        )
        .expect("raw pyramid volume match");
    let regularized = plan
        .track_volume_pyramid_regularized(
            &pyramid,
            BlockGrid::dense([0, 4, 4]),
            SubpixelRefinement::None,
            &prior,
        )
        .expect("regularized pyramid volume match");

    assert_eq!(regularized.centres, raw.centres);
    assert_eq!(regularized.peak_similarities, raw.peak_similarities);
    for (index, (&peak, &raw_displacement)) in raw
        .peak_similarities
        .iter()
        .zip(&raw.displacements)
        .enumerate()
    {
        let confidence = if peak.is_finite() && peak >= prior.minimum_peak_similarity {
            peak.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let weight = confidence * confidence / (1.0 + confidence * confidence);
        let expected: [f64; 3] = std::array::from_fn(|axis| {
            prior.mean[axis] * (1.0 - weight) + raw_displacement[axis] * weight
        });
        for (axis, &want) in expected.iter().enumerate() {
            assert!(
                (regularized.displacements[index][axis] - want).abs() < 1.0e-12,
                "block {index}, axis {axis}: expected {want}, got {}",
                regularized.displacements[index][axis]
            );
        }
    }
}

#[test]
fn pyramid_volume_tracking_rejects_invalid_level_inputs() {
    let plan = MultiResolutionSearch::new([0, 2, 2], [0, 2, 2], 2).expect("valid plan");
    let image = pyramid_image(1, [0, 0, 0]);
    let level = PyramidLevel {
        fixed: &image,
        moving: &image,
        dims: [1, 40, 40],
    };
    assert!(plan
        .track_volume_pyramid(
            &[level],
            BlockGrid::dense([0, 2, 2]),
            SubpixelRefinement::None
        )
        .is_err());

    let short = vec![0.0_f32; image.len() - 1];
    let pyramid = [
        PyramidLevel {
            fixed: &short,
            moving: &image,
            dims: [1, 40, 40],
        },
        PyramidLevel {
            fixed: &image,
            moving: &image,
            dims: [1, 40, 40],
        },
    ];
    assert!(plan
        .track_volume_pyramid(
            &pyramid,
            BlockGrid::dense([0, 2, 2]),
            SubpixelRefinement::None
        )
        .is_err());
}

#[test]
fn axial_radius_orients_the_transverse_pair_around_every_axis() {
    // The two transverse radii fill the non-axial axes in axis order, whichever
    // axis is axial. Each case is checked against the layout written by hand, so
    // an off-by-one in the index shift cannot pass.
    let cases = [(0usize, [9, 2, 3]), (1, [2, 9, 3]), (2, [2, 3, 9])];
    for (axial_axis, expected) in cases {
        let config =
            BlockMatchingConfig::with_axial_radius(axial_axis, 9, [2, 3], [4, 4, 4]).unwrap();
        assert_eq!(
            config.block_radius, expected,
            "axial axis {axial_axis} placed the radii wrongly"
        );
        assert_eq!(config.search_radius, [4, 4, 4]);
    }

    assert!(
        BlockMatchingConfig::with_axial_radius(3, 9, [2, 3], [4, 4, 4]).is_err(),
        "an out-of-range axial axis must be rejected, not wrapped"
    );
}

#[test]
fn radius_sources_build_oriented_validated_geometry() {
    let bandwidth = BlockMatchingConfig::from_transducer_bandwidth(
        1540.0,
        5.0e6,
        0.6,
        1.0e-4,
        2,
        [0, 2],
        [0, 4, 4],
    )
    .expect("valid transducer geometry");
    assert_eq!(bandwidth.block_radius, [0, 2, 3]);
    assert_eq!(bandwidth.search_radius, [0, 4, 4]);

    let signal = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
    let autocorrelation =
        BlockMatchingConfig::from_axial_autocorrelation(&signal, 0.5, 0, [2, 3], [4, 0, 4])
            .expect("valid autocorrelation geometry");
    assert!(autocorrelation.block_radius[0] >= 1);
    assert_eq!(autocorrelation.block_radius[1], 2);
    assert_eq!(autocorrelation.block_radius[2], 3);
}

#[test]
fn radius_geometry_rejects_invalid_axis_and_source_inputs() {
    assert!(BlockMatchingConfig::with_axial_radius(3, 1, [1, 1], [1, 1, 1]).is_err());
    assert!(BlockMatchingConfig::from_axial_autocorrelation(
        &[1.0, 1.0, 1.0],
        0.5,
        0,
        [1, 1],
        [1, 1, 1],
    )
    .is_err());
    assert!(BlockMatchingConfig::from_transducer_bandwidth(
        1540.0,
        5.0e6,
        0.6,
        1.0e-4,
        3,
        [1, 1],
        [1, 1, 1],
    )
    .is_err());
}
