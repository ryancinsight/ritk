use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use ritk_image::Image;

use super::statistics::inverse_covariance;
use super::{segment_values, VectorConfidenceConnectedConfig, VectorConfidenceConnectedFilter};

type LegacyBackend = SequentialBackend;

fn scene() -> (Vec<Vec<f32>>, [usize; 3]) {
    let (height, width) = (10usize, 10usize);
    let mut first = vec![0.0; height * width];
    let mut second = vec![0.0; height * width];
    for y in 0..height {
        for x in 0..width {
            first[y * width + x] = 0.1 * (((x + y) % 3) as f32 - 1.0);
            second[y * width + x] = 0.1 * (((x * y) % 5) as f32 - 2.0);
        }
    }
    for y in 3..8 {
        for x in 3..8 {
            first[y * width + x] += 3.0;
        }
    }
    (vec![first, second], [1, height, width])
}

fn config() -> VectorConfidenceConnectedConfig {
    VectorConfidenceConnectedConfig::new(3.0, 4, 1, 1.0).unwrap()
}

fn expected_blob() -> Vec<f32> {
    let mut expected = vec![0.0; 100];
    for y in 3..8 {
        for x in 3..8 {
            expected[y * 10 + x] = 1.0;
        }
    }
    expected
}

#[test]
fn borrowed_core_matches_simpleitk_blob_exactly() {
    let (channels, dimensions) = scene();
    for iterations in [0, 1, 4] {
        let config = VectorConfidenceConnectedConfig::new(3.0, iterations, 1, 1.0).unwrap();
        let output = segment_values(
            &channels,
            dimensions,
            &[[0, 5, 5].into(), [0, 4, 6].into()],
            config,
        )
        .unwrap();
        assert_eq!(output, expected_blob());
    }
}

#[test]
fn corner_seed_uses_simpleitk_zero_flux_neighborhood() {
    // NumPy default_rng(0), shape (5, 5, 2), split by final component.
    let channels = vec![
        vec![
            0.125_730_22,
            0.640_422_64,
            -0.535_669_4,
            1.304,
            -0.703_735_23,
            -0.623_274_45,
            -2.325_030_8,
            -1.245_911,
            -0.544_259,
            0.411_630_54,
            -0.128_534_66,
            -0.665_194_7,
            0.903_470_16,
            -0.743_499_3,
            -0.457_725_82,
            -1.009_618_2,
            -0.159_225_02,
            0.214_659_12,
            -0.653_828_6,
            0.783_975_5,
            -1.259_065_5,
            1.345_875_4,
            0.264_455_62,
            1.458_020_7,
            1.801_634_9,
        ],
        vec![
            -0.132_104_86,
            0.104_900_114,
            0.361_595_06,
            0.947_080_97,
            -1.265_421_5,
            0.041_325_98,
            -0.218_791_66,
            -0.732_267_4,
            -0.316_300_15,
            1.042_513_4,
            1.366_463_4,
            0.351_510_08,
            0.094_012_3,
            -0.921_725_4,
            0.220_195_13,
            -0.209_175_57,
            0.540_845_6,
            0.355_372_7,
            -0.129_613_64,
            1.493_431_1,
            1.513_923_8,
            0.781_311_4,
            -0.313_922_82,
            1.960_258_4,
            1.315_103_8,
        ],
    ];
    let config = VectorConfidenceConnectedConfig::new(0.5, 0, 1, 1.0).unwrap();
    let output = segment_values(&channels, [1, 5, 5], &[[0, 0, 0].into()], config).unwrap();
    let mut expected = vec![0.0; 25];
    expected[0] = 1.0;
    assert_eq!(output, expected);
}

#[test]
fn empty_seeds_produce_an_exact_empty_mask() {
    let (channels, dimensions) = scene();
    assert_eq!(
        segment_values(&channels, dimensions, &[], config()).unwrap(),
        vec![0.0; 100]
    );
}

#[test]
fn configuration_and_input_errors_are_exact() {
    assert_eq!(
        VectorConfidenceConnectedConfig::new(f64::NAN, 1, 1, 1.0)
            .unwrap_err()
            .to_string(),
        "vector confidence multiplier must be finite and positive, got NaN"
    );
    assert_eq!(
        VectorConfidenceConnectedConfig::new(1.0, 1, 1, f32::INFINITY)
            .unwrap_err()
            .to_string(),
        "vector confidence replacement must be finite, got inf"
    );
    assert_eq!(
        segment_values(&[vec![0.0; 4], vec![1.0; 3]], [1, 2, 2], &[], config())
            .unwrap_err()
            .to_string(),
        "vector confidence channel 1 length 3 != voxel count 4"
    );
    assert_eq!(
        segment_values(
            &[vec![0.0, f32::NAN]],
            [1, 1, 2],
            &[[0, 0, 0].into()],
            config(),
        )
        .unwrap_err()
        .to_string(),
        "vector confidence channel 0 sample 1 must be finite, got NaN"
    );
    let (channels, dimensions) = scene();
    assert_eq!(
        segment_values(&channels, dimensions, &[[0, 10, 0].into()], config()).unwrap(),
        vec![0.0; 100]
    );
}

#[test]
fn maximal_radius_is_bounded_by_unique_image_voxels() {
    let channels = vec![vec![1.0]];
    let config = VectorConfidenceConnectedConfig::new(1.0, 0, usize::MAX, 1.0).unwrap();
    assert_eq!(
        segment_values(&channels, [1, 1, 1], &[[0, 0, 0].into()], config).unwrap(),
        vec![1.0]
    );
}

#[test]
fn covariance_inversion_obeys_itk_determinant_boundary() {
    let singular = inverse_covariance(&[0.000_488_281_25, 0.0, 0.0, 0.000_488_281_25], 2).unwrap();
    assert_eq!(singular[0], f64::MAX.powf(1.0 / 3.0) / 2.0);
    assert_eq!(singular[1], 0.0);
    assert_eq!(singular[2], 0.0);
    assert_eq!(singular[3], singular[0]);

    let regular = inverse_covariance(&[0.001_953_125, 0.0, 0.0, 0.001_953_125], 2).unwrap();
    assert_eq!(regular, vec![512.0, 0.0, 0.0, 512.0]);
}

#[test]
fn face_connectivity_excludes_diagonal_only_members() {
    let channels = vec![vec![1.0, 0.0, 0.0, 1.0]];
    let config = VectorConfidenceConnectedConfig::new(1.0, 0, 0, 1.0).unwrap();
    assert_eq!(
        segment_values(&channels, [1, 2, 2], &[[0, 0, 0].into()], config).unwrap(),
        vec![1.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn seed_distance_raises_the_membership_threshold() {
    let channels = vec![vec![0.0, 1.0]];
    let config = VectorConfidenceConnectedConfig::new(0.01, 0, 1, 1.0).unwrap();
    assert_eq!(
        segment_values(&channels, [1, 1, 2], &[[0, 0, 1].into()], config).unwrap(),
        vec![0.0, 1.0]
    );
}

#[test]
fn invalid_seeds_are_ignored_without_discarding_valid_seeds() {
    let (channels, dimensions) = scene();
    let output = segment_values(
        &channels,
        dimensions,
        &[[0, 10, 0].into(), [0, 5, 5].into(), [0, 4, 6].into()],
        config(),
    )
    .unwrap();
    assert_eq!(output, expected_blob());
}

#[test]
fn legacy_and_native_outputs_are_exact_with_nonidentity_geometry() {
    let (channels, dimensions) = scene();
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let legacy: Vec<Image<f32, LegacyBackend, 3>> = channels
        .iter()
        .map(|values| {
            Image::from_flat_on(
                values.clone(),
                dimensions,
                origin,
                spacing,
                direction,
                &Default::default(),
            )
        })
        .collect();
    let native: Vec<_> = channels
        .into_iter()
        .map(|values| {
            NativeImage::from_flat_on(
                values,
                dimensions,
                origin,
                spacing,
                direction,
                &SequentialBackend,
            )
            .unwrap()
        })
        .collect();
    let filter = VectorConfidenceConnectedFilter::new([[0, 5, 5], [0, 4, 6]], config());
    let legacy_refs: Vec<_> = legacy.iter().collect();
    let native_refs: Vec<_> = native.iter().collect();
    let expected = filter.apply(&legacy_refs).unwrap();
    let actual = filter
        .apply_native(&native_refs, &SequentialBackend)
        .unwrap();
    assert_eq!(actual.data_slice().unwrap(), expected.data_slice().as_ref());
    assert_eq!(*actual.origin(), origin);
    assert_eq!(*actual.spacing(), spacing);
    assert_eq!(*actual.direction(), direction);
}

#[test]
fn channel_geometry_mismatch_is_rejected_exactly() {
    let first = Image::<f32, LegacyBackend, 3>::from_flat_on(
        vec![1.0],
        [1, 1, 1],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &Default::default(),
    );
    let second = Image::<f32, LegacyBackend, 3>::from_flat_on(
        vec![1.0],
        [1, 1, 1],
        Point::new([1.0, 0.0, 0.0]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &Default::default(),
    );
    let filter = VectorConfidenceConnectedFilter::new([[0, 0, 0]], config());
    assert_eq!(
        filter.apply(&[&first, &second]).unwrap_err().to_string(),
        "vector confidence channel 1 geometry differs from channel 0"
    );
}
