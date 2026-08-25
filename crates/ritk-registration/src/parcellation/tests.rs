use super::*;

use coeus_core::SequentialBackend;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

const TOLERANCE: f64 = 1.0e-12;

/// Build an image from a flat `[nz, ny, nx]` buffer with the given geometry.
fn image(
    data: &[f32],
    shape: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    direction: Direction<3>,
) -> Image<f32, B, 3> {
    let device = Default::default();
    let tensor = Tensor::<f32, B>::from_slice_on(shape, data, &device);
    Image::new(tensor, Point::new(origin), Spacing::new(spacing), direction).expect("valid image")
}

/// A cube image with unit spacing at the origin, axis aligned.
fn unit_image(data: &[f32], shape: [usize; 3]) -> Image<f32, B, 3> {
    image(data, shape, [1.0; 3], [0.0; 3], Direction::identity())
}

// -- The grid bridge -----------------------------------------------------

/// Physical position of an image voxel, obtained *without* going through the
/// axis convention the bridge is being tested for.
///
/// `index_to_world_native` documents its input columns as innermost-first —
/// `[ix, iy, iz]` — which is the parcellation grid's own order, so feeding it
/// here removes the ambiguity that the axis-major
/// `continuous_index_to_physical_point` introduces. Reaching for that other
/// entry point is what makes a bridge error invisible: pass `[ix, iy, iz]` to an
/// axis-major API and the test reverses exactly the axes the bridge failed to,
/// so the two mistakes cancel and every voxel appears to round-trip.
fn world_of(reference: &Image<f32, B, 3>, ix: usize, iy: usize, iz: usize) -> Point<3> {
    // The native path works in the image's own element type, so the position
    // comes back as f32. Every coordinate in these fixtures is a small exact
    // binary value, and the lookup under test rounds to the nearest voxel, so
    // the narrowing carries no error the assertions could see.
    let index = Tensor::<f32, B>::from_slice([1, 3], &[ix as f32, iy as f32, iz as f32]);
    let world = reference.index_to_world_native(&index);
    let values = world.as_slice();
    Point::new([
        f64::from(values[0]),
        f64::from(values[1]),
        f64::from(values[2]),
    ])
}

/// A hand-computed position, so the convention is pinned by arithmetic rather
/// than by agreement between two library calls.
///
/// For a `[2, 3, 4]` volume with spacing `[4, 2, 1]` and an identity direction,
/// the image's axis 0 is its *slowest* index, so the voxel at flat offset
/// `1*3*4 + 2*4 + 3 = 23` — image index `(i0, i1, i2) = (1, 2, 3)` — sits at
/// `(4*1, 2*2, 1*3) = (4, 4, 3)`.
#[test]
fn the_image_axis_convention_is_outermost_first() {
    let reference = image(
        &[0.0; 24],
        [2, 3, 4],
        [4.0, 2.0, 1.0],
        [0.0; 3],
        Direction::identity(),
    );

    // The same voxel, addressed innermost-first as (ix, iy, iz) = (3, 2, 1).
    let world = world_of(&reference, 3, 2, 1).to_array();
    assert!(
        (world[0] - 4.0).abs() < TOLERANCE
            && (world[1] - 4.0).abs() < TOLERANCE
            && (world[2] - 3.0).abs() < TOLERANCE,
        "expected (4, 4, 3), got {world:?}"
    );
}

/// The bridge's whole job: a label read through the parcellation at a voxel's
/// physical position must be the label the image holds at that voxel.
///
/// The volume is anisotropic on all three axes, which is what makes a
/// transposition detectable — with equal spacings a reversed axis order lands on
/// the right position by coincidence.
#[test]
fn every_voxel_round_trips_through_the_grid_bridge() {
    let shape = [3, 4, 5]; // [nz, ny, nx]
    let voxels = shape[0] * shape[1] * shape[2];
    let labels: Vec<u32> = (1..=voxels as u32).collect();
    let intensity: Vec<f32> = labels.iter().map(|label| *label as f32).collect();
    let reference = image(
        &intensity,
        shape,
        [4.0, 1.5, 0.5], // image axis order: slowest index first
        [-7.0, 3.0, 11.0],
        Direction::identity(),
    );

    let parcellation =
        parcellation_from_labels(labels.clone().into_boxed_slice(), &reference, Vec::new())
            .expect("valid parcellation");

    let [nz, ny, nx] = shape;
    assert_eq!(parcellation.grid().shape(), [nx, ny, nz]);
    // The grid's fastest axis carries the image's fastest spacing.
    assert_eq!(parcellation.grid().spacing(), [0.5, 1.5, 4.0]);

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = iz * ny * nx + iy * nx + ix;
                assert_eq!(
                    parcellation.label_at(&world_of(&reference, ix, iy, iz)),
                    Some(labels[flat]),
                    "voxel (ix {ix}, iy {iy}, iz {iz})"
                );
            }
        }
    }
}

/// The same round trip under an oblique direction matrix, which is what an
/// acquired volume actually carries — and which makes the column reversal
/// matter as well as the spacing one.
#[test]
fn the_grid_bridge_survives_an_oblique_direction_matrix() {
    let shape = [2, 3, 4];
    let voxels = shape[0] * shape[1] * shape[2];
    let labels: Vec<u32> = (1..=voxels as u32).collect();
    let intensity: Vec<f32> = labels.iter().map(|label| *label as f32).collect();

    let angle = 0.5_f64;
    let (sin, cos) = angle.sin_cos();
    let direction = Direction::from_row_major([cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0]);
    let reference = image(
        &intensity,
        shape,
        [3.0, 2.0, 1.0],
        [4.0, -1.0, 0.5],
        direction,
    );

    let parcellation =
        parcellation_from_labels(labels.clone().into_boxed_slice(), &reference, Vec::new())
            .expect("valid parcellation");

    let [nz, ny, nx] = shape;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = iz * ny * nx + iy * nx + ix;
                assert_eq!(
                    parcellation.label_at(&world_of(&reference, ix, iy, iz)),
                    Some(labels[flat]),
                    "voxel (ix {ix}, iy {iy}, iz {iz})"
                );
            }
        }
    }
}

#[test]
fn an_all_background_label_volume_is_rejected() {
    let reference = unit_image(&[0.0; 8], [2, 2, 2]);
    let error = parcellation_from_labels(vec![0; 8].into_boxed_slice(), &reference, Vec::new())
        .expect_err("the rejected input must yield the typed error");
    assert!(matches!(error, ParcellationError::EmptyParcellation));
}

// ── Spacing order into the registration ──────────────────────────────────

/// The registration shares the image's axis order, unlike the parcellation
/// grid, so its spacing passes through unreversed. The two directions are easy
/// to conflate and the difference is silent on any anisotropic volume, so both
/// are pinned here rather than only one.
#[test]
fn registration_spacing_keeps_the_image_axis_order() {
    let reference = image(
        &[0.0; 24],
        [2, 3, 4],
        [0.5, 1.0, 2.0],
        [0.0; 3],
        Direction::identity(),
    );
    assert_eq!(registration_spacing(&reference), [0.5, 1.0, 2.0]);
    assert_eq!(image_dims(&reference), [2, 3, 4]);

    // The grid bridge reverses where the registration does not.
    let parcellation =
        parcellation_from_labels(vec![1; 24].into_boxed_slice(), &reference, Vec::new())
            .expect("valid parcellation");
    assert_eq!(parcellation.grid().spacing(), [2.0, 1.0, 0.5]);
}

// ── Label rounding ───────────────────────────────────────────────────────

/// A nearest-neighbour warp only ever copies values that were already present,
/// so every label must survive the float round trip exactly.
#[test]
fn labels_survive_the_float_round_trip() {
    for label in [0_u32, 1, 2, 17, 1000, 65_535, 16_777_215] {
        assert_eq!(round_to_label(label as f32), label, "label {label}");
    }
}

/// Clamp-to-border fill outside the field of view must read as background, not
/// as a small label.
#[test]
fn out_of_field_values_read_as_background() {
    assert_eq!(round_to_label(0.0), 0);
    assert_eq!(round_to_label(-3.0), 0);
}

// ── Atlas validation ─────────────────────────────────────────────────────

fn atlas(intensity: Vec<f32>, labels: Vec<u32>) -> LabelledAtlas {
    LabelledAtlas {
        intensity,
        labels,
        region_names: vec![(1, "One".into()), (2, "Two".into())],
    }
}

#[test]
fn an_empty_atlas_set_is_rejected() {
    let subject = unit_image(&[0.0; 8], [2, 2, 2]);
    let error = parcellate_with_atlas_set(&subject, &[], &AtlasParcellationConfig::default())
        .expect_err("the rejected input must yield the typed error");
    assert!(matches!(error, RegistrationError::InvalidConfiguration(_)));
}

#[test]
fn an_atlas_on_a_different_grid_is_rejected() {
    let subject = unit_image(&[0.0; 8], [2, 2, 2]);
    let mismatched = atlas(vec![0.0; 27], vec![1; 27]);
    let error = parcellate_with_atlas_set(
        &subject,
        std::slice::from_ref(&mismatched),
        &AtlasParcellationConfig::default(),
    )
    .expect_err("the rejected input must yield the typed error");
    assert!(matches!(error, RegistrationError::DimensionMismatch(_)));
}

#[test]
fn region_names_merge_across_atlases_without_duplicating_labels() {
    let first = LabelledAtlas {
        intensity: Vec::new(),
        labels: Vec::new(),
        region_names: vec![(2, "Two".into()), (1, "One".into())],
    };
    let second = LabelledAtlas {
        intensity: Vec::new(),
        labels: Vec::new(),
        region_names: vec![(1, "One".into()), (3, "Three".into())],
    };

    let merged = merged_region_names(&[first, second]);
    assert_eq!(
        merged,
        vec![
            (1, "One".to_string()),
            (2, "Two".to_string()),
            (3, "Three".to_string())
        ]
    );
}

// ── End to end ───────────────────────────────────────────────────────────

/// A subject built from a bright cube, and an atlas holding the same cube with
/// a label on it. The parcellation must land the label on the subject's cube.
///
/// The registration has real work to do — the atlas cube is displaced from the
/// subject's — but the displacement is small enough that the deformation should
/// close it. What the assertion checks is the *composition*: whether the
/// pipeline recovers the atlas-to-subject direction, resamples with nearest
/// neighbour, and lands the result on the subject's grid the right way round.
/// A sign error in the field composition would place the label on the far side
/// of the subject cube, and a transposed grid would place it on a different
/// axis; both leave a valid-looking parcellation.
#[test]
fn a_displaced_atlas_lands_its_label_on_the_subject_structure() {
    let shape = [12, 12, 12];
    let voxels = shape[0] * shape[1] * shape[2];

    // The subject's bright block sits at x ∈ 5..8; the atlas's at x ∈ 4..7.
    let block = |x_range: std::ops::Range<usize>| -> (Vec<f32>, Vec<u32>) {
        let mut intensity = vec![0.0_f32; voxels];
        let mut labels = vec![0_u32; voxels];
        for iz in 4..8 {
            for iy in 4..8 {
                for ix in x_range.clone() {
                    let flat = iz * shape[1] * shape[2] + iy * shape[2] + ix;
                    intensity[flat] = 1.0;
                    labels[flat] = 1;
                }
            }
        }
        (intensity, labels)
    };

    let (subject_intensity, subject_truth) = block(5..8);
    let (atlas_intensity, atlas_labels) = block(4..7);
    let subject = unit_image(&subject_intensity, shape);

    let config = AtlasParcellationConfig {
        registration: MultiResSyNConfig {
            num_levels: 2,
            iterations_per_level: vec![20, 10],
            sigma_smooth: 1.0,
            convergence_threshold: 1.0e-7,
            convergence_window: 5,
            n_squarings: 5,
            cc_window_radius: 2,
            gradient_step: 0.25,
            enforce_inverse_consistency: InverseConsistency::Relaxed,
        },
        fusion: LabelFusion::MajorityVote,
    };

    let result = parcellate_with_atlas(&subject, &atlas(atlas_intensity, atlas_labels), &config)
        .expect("parcellation succeeds");

    let produced = result.parcellation.labels();
    assert_eq!(produced.len(), voxels);

    // Agreement with the subject's own structure, measured by the Dice
    // coefficient — twice the intersection over the summed sizes — against the
    // unregistered atlas as the baseline. Dice rather than a raw intersection
    // because a field that smeared the label over the whole volume would score
    // a perfect intersection while being useless; the denominator penalises
    // that.
    let (_, unregistered) = block(4..7);
    let dice = |candidate: &[u32]| -> f64 {
        let intersection = candidate
            .iter()
            .zip(&subject_truth)
            .filter(|(produced, truth)| **produced == 1 && **truth == 1)
            .count();
        let candidate_size = candidate.iter().filter(|label| **label == 1).count();
        let truth_size = subject_truth.iter().filter(|label| **label == 1).count();
        2.0 * intersection as f64 / (candidate_size + truth_size) as f64
    };

    let before = dice(&unregistered);
    let after = dice(produced);
    assert!(
        after > before,
        "registration must improve the label agreement with the subject: Dice {after:.3} after versus {before:.3} before"
    );
    // The displacement here is a single voxel against a high-contrast block, so
    // the registration is expected to close it essentially completely rather
    // than merely improve on the baseline. Nine tenths leaves room for the
    // interpolation at the block's edge without admitting a half-closed
    // deformation, which would score around the 0.67 the baseline already gives.
    assert!(
        after > 0.9,
        "a one-voxel displacement of a high-contrast block must close: Dice {after:.3}"
    );

    // A single atlas has nothing to disagree with.
    assert_eq!(result.agreement.len(), voxels);
    for value in &result.agreement {
        assert!((value - 1.0).abs() < 1.0e-6, "got {value}");
    }
    assert_eq!(result.registration_quality.len(), 1);
    assert!(
        result.registration_quality[0].is_finite(),
        "the reported cross-correlation must be a number"
    );
}

/// Fusing several identical atlases must reproduce what one of them gives, with
/// unanimous agreement. Anything else means the vote is not counting.
#[test]
fn identical_atlases_agree_unanimously() {
    let shape = [8, 8, 8];
    let voxels = shape[0] * shape[1] * shape[2];
    let mut intensity = vec![0.0_f32; voxels];
    let mut labels = vec![0_u32; voxels];
    for iz in 2..6 {
        for iy in 2..6 {
            for ix in 2..6 {
                let flat = iz * shape[1] * shape[2] + iy * shape[2] + ix;
                intensity[flat] = 1.0;
                labels[flat] = 1;
            }
        }
    }
    let subject = unit_image(&intensity, shape);

    let config = AtlasParcellationConfig {
        registration: MultiResSyNConfig {
            num_levels: 1,
            iterations_per_level: vec![3],
            sigma_smooth: 1.0,
            convergence_threshold: 1.0e-7,
            convergence_window: 3,
            n_squarings: 4,
            cc_window_radius: 1,
            gradient_step: 0.25,
            enforce_inverse_consistency: InverseConsistency::Relaxed,
        },
        fusion: LabelFusion::MajorityVote,
    };

    let one = atlas(intensity.clone(), labels.clone());
    let set = [one.clone(), one.clone(), one.clone()];

    let single = parcellate_with_atlas(&subject, &one, &config).expect("single atlas");
    let fused = parcellate_with_atlas_set(&subject, &set, &config).expect("three atlases");

    assert_eq!(single.parcellation.labels(), fused.parcellation.labels());
    for value in &fused.agreement {
        assert!(
            (value - 1.0).abs() < 1.0e-6,
            "identical atlases must agree unanimously, got {value}"
        );
    }
    assert_eq!(fused.registration_quality.len(), 3);
}

/// A dissenting atlas must lose the vote and lower the agreement where it
/// disagreed, rather than being silently ignored or silently winning.
#[test]
fn a_dissenting_atlas_loses_the_vote_and_lowers_the_agreement() {
    let shape = [6, 6, 6];
    let voxels = shape[0] * shape[1] * shape[2];
    let mut intensity = vec![0.0_f32; voxels];
    for value in intensity.iter_mut().skip(voxels / 2) {
        *value = 1.0;
    }
    let subject = unit_image(&intensity, shape);

    // Two atlases label everything 1; one labels everything 2.
    let majority = atlas(intensity.clone(), vec![1; voxels]);
    let dissenter = atlas(intensity.clone(), vec![2; voxels]);

    let config = AtlasParcellationConfig {
        registration: MultiResSyNConfig {
            num_levels: 1,
            iterations_per_level: vec![2],
            sigma_smooth: 1.0,
            convergence_threshold: 1.0e-7,
            convergence_window: 2,
            n_squarings: 4,
            cc_window_radius: 1,
            gradient_step: 0.25,
            enforce_inverse_consistency: InverseConsistency::Relaxed,
        },
        fusion: LabelFusion::MajorityVote,
    };

    let result =
        parcellate_with_atlas_set(&subject, &[majority.clone(), majority, dissenter], &config)
            .expect("parcellation succeeds");

    assert_eq!(
        result.parcellation.region_labels(),
        vec![1],
        "the majority label must win everywhere"
    );
    for value in &result.agreement {
        assert!(
            (value - 2.0 / 3.0).abs() < 1.0e-5,
            "two of three atlases agreed, so the agreement must be 2/3, got {value}"
        );
    }
}
