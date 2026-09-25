use super::scalar_volume;
use crate::render::{
    ProjectionStatistic, ResliceError, ResliceInterpolation, ResliceOrientation,
    ResliceOrientationError, ReslicePlane,
};

#[test]
fn bounded_angles_rotate_and_wrap_without_losing_orientation() {
    let boundary = ResliceOrientation::try_new(-180.0, 90.0)
        .expect("inclusive yaw and pitch boundaries are valid");
    assert_eq!(boundary.yaw_degrees(), -180.0);
    assert_eq!(boundary.pitch_degrees(), 90.0);

    assert!(matches!(
        ResliceOrientation::try_new(181.0, 0.0),
        Err(ResliceOrientationError::InvalidOrientation { .. })
    ));
    assert!(matches!(
        ResliceOrientation::try_new(0.0, -91.0),
        Err(ResliceOrientationError::InvalidOrientation { .. })
    ));
    assert!(matches!(
        ResliceOrientation::try_new(f64::INFINITY, 0.0),
        Err(ResliceOrientationError::InvalidOrientation { .. })
    ));

    let rotated = ResliceOrientation::try_new(170.0, 85.0)
        .expect("initial angles are valid")
        .rotated_by(20.0, 10.0)
        .expect("finite rotation deltas wrap yaw and clamp pitch");
    assert_eq!(rotated.yaw_degrees(), -170.0);
    assert_eq!(rotated.pitch_degrees(), 90.0);
    assert!(matches!(
        rotated.rotated_by(0.0, f64::NAN),
        Err(ResliceOrientationError::InvalidOrientation { .. })
    ));
}

#[test]
fn centered_plane_uses_one_contract_for_mapping_and_sampling() {
    let mut volume = scalar_volume([3, 4, 5]);
    volume.spacing = [2.0, 3.0, 4.0];
    let plane = ReslicePlane::centered_oblique(
        &volume,
        [1.0, 1.5, 2.0],
        ResliceOrientation::default(),
        ResliceInterpolation::Linear,
    )
    .expect("source-aligned plane is valid");

    assert_eq!(plane.dimensions(), [5, 4]);
    assert_eq!(plane.depth_samples(), 1);
    assert_eq!(plane.origin(), [2.0, 0.0, 0.0]);
    assert_eq!(plane.horizontal_step(), [0.0, 0.0, 4.0]);
    assert_eq!(plane.vertical_step(), [0.0, 3.0, 0.0]);
    assert_eq!(plane.depth_step(), [-2.0, 0.0, 0.0]);
    assert_eq!(
        plane
            .patient_at_pixel([2.0, 1.5])
            .expect("center pixel maps through the rendering plane")
            .coordinates(),
        [2.0, 4.5, 8.0]
    );

    let output = plane
        .compute(&volume, ProjectionStatistic::Maximum)
        .expect("the same plane samples the source");
    let (expected, width, height) = volume.extract_slice(0, 1);
    assert_eq!(output.dimensions(), [width, height]);
    assert_eq!(output.pixels(), expected.as_slice());
}

#[test]
fn quarter_turn_orientation_maps_and_samples_the_rotated_plane() {
    let volume = scalar_volume([5, 5, 5]);
    let orientation = ResliceOrientation::try_new(90.0, 90.0)
        .expect("quarter-turn angles are inside the documented bounds");
    let plane = ReslicePlane::centered_oblique(
        &volume,
        [2.0, 2.0, 2.0],
        orientation,
        ResliceInterpolation::Nearest,
    )
    .expect("the rotated plane fits in the source volume");

    assert_eq!(plane.dimensions(), [5, 5]);
    let horizontal = plane.horizontal_step();
    let vertical = plane.vertical_step();
    assert!(horizontal[0] > 0.0 && horizontal[0].abs() > horizontal[2].abs());
    assert!(vertical[2] > 0.0 && vertical[2].abs() > vertical[1].abs());
    let mapped_center = plane
        .patient_at_pixel([2.0, 2.0])
        .expect("the plane center lies within its pixel bounds")
        .coordinates();
    assert_eq!(mapped_center.map(f64::round), [2.0, 2.0, 2.0]);

    let output = plane
        .compute(&volume, ProjectionStatistic::Maximum)
        .expect("the rotated plane samples the source volume");
    assert_eq!(
        output.pixels(),
        &[
            20.0, 120.0, 220.0, 320.0, 420.0, 21.0, 121.0, 221.0, 321.0, 421.0, 22.0, 122.0, 222.0,
            322.0, 422.0, 23.0, 123.0, 223.0, 323.0, 423.0, 24.0, 124.0, 224.0, 324.0, 424.0,
        ]
    );
}

#[test]
fn failed_depth_translation_preserves_the_valid_plane() {
    let volume = scalar_volume([3, 5, 5]);
    let plane = ReslicePlane::centered_oblique(
        &volume,
        [1.0, 2.0, 2.0],
        ResliceOrientation::default(),
        ResliceInterpolation::Linear,
    )
    .expect("source-aligned plane is valid");
    let before = plane
        .compute(&volume, ProjectionStatistic::Maximum)
        .expect("original plane computes");

    assert!(matches!(
        plane.shifted_along_depth(&volume, 10.0),
        Err(ResliceOrientationError::Source(
            ResliceError::OutOfVolume { .. }
        ))
    ));
    let after = plane
        .compute(&volume, ProjectionStatistic::Maximum)
        .expect("failed translation leaves the original plane usable");
    assert_eq!(after.dimensions(), before.dimensions());
    assert_eq!(after.pixels(), before.pixels());
}

#[test]
fn reorthogonalization_controls_loss_for_nearly_parallel_source_axes() {
    let mut volume = scalar_volume([17, 17, 17]);
    // This 1e-12 determinant is about 35 times the affine validator's
    // 128*epsilon singularity floor at unit scale and reproduces the
    // cancellation-sensitive near-parallel source axes.
    let separation = 1.0e-12;
    volume.direction = [1.0, 0.0, 0.0, 0.0, 1.0 + separation, 1.0, 0.0, 1.0, 1.0];
    let plane = ReslicePlane::centered_oblique(
        &volume,
        [8.0, 8.0, 8.0],
        ResliceOrientation::default(),
        ResliceInterpolation::Linear,
    )
    .expect("the affine remains above its singularity threshold");

    let horizontal = plane.horizontal_step();
    let vertical = plane.vertical_step();
    let dot = horizontal
        .into_iter()
        .zip(vertical)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let scale = horizontal
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        * vertical
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
    // Two length-three projection passes and normalization leave fewer than
    // 32 rounded operations in the normalized dot product. The bound is the
    // standard gamma_n = n*epsilon/(1-n*epsilon), with n conservatively 32.
    const DOT_ROUNDING_OPERATIONS: f64 = 32.0;
    let operation_count = DOT_ROUNDING_OPERATIONS;
    let rounding_bound = operation_count * f64::EPSILON / (1.0 - operation_count * f64::EPSILON);
    assert!(dot.abs() <= rounding_bound * scale);
}
