use super::*;
use std::cell::Cell;
use std::num::NonZeroU8;

const TRANSLATION_RESOLUTION_MM: f64 = 0.75;

fn config() -> RigidSearchConfig {
    RigidSearchConfig::try_new(12.0, 8.0, 0.5, TRANSLATION_RESOLUTION_MM, 256)
        .expect("valid search configuration")
}

fn pose_objective(target_translation_mm: f64) -> impl FnMut(&AffineTransform) -> Result<f64> {
    pose_objective_xyz([target_translation_mm, 0.0, 0.0])
}

fn pose_objective_xyz(
    target_translation_mm: [f64; 3],
) -> impl FnMut(&AffineTransform) -> Result<f64> {
    move |transform| {
        let matrix = transform.as_array();
        let identity_rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let rotation_residual = [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ]
        .into_iter()
        .zip(identity_rotation)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>();
        Ok(-((matrix[3] - target_translation_mm[0]).powi(2)
            + (matrix[7] - target_translation_mm[1]).powi(2)
            + (matrix[11] - target_translation_mm[2]).powi(2)
            + rotation_residual))
    }
}

fn finite_bounded_pose_objective(
    target_translation_mm: [f64; 3],
    half_range_mm: f64,
) -> impl FnMut(&AffineTransform) -> Result<f64> {
    let mut objective = pose_objective_xyz(target_translation_mm);
    move |transform| {
        let matrix = transform.as_array();
        assert!(matrix.iter().all(|value| value.is_finite()));
        assert!([matrix[3], matrix[7], matrix[11]]
            .iter()
            .all(|value| value.abs() <= half_range_mm));
        objective(transform)
    }
}

#[test]
fn centroid_transform_maps_fixed_centroid_to_moving_centroid() {
    let fixed = [20.0, -10.0, 4.0];
    let moving = [-3.0, 12.0, 9.0];
    let transform = rigid_about_centroid(euler_zyx(0.2, -0.1, 0.3), fixed, moving);
    let matrix = transform.as_array();
    let mapped = [
        matrix[0] * fixed[0] + matrix[1] * fixed[1] + matrix[2] * fixed[2] + matrix[3],
        matrix[4] * fixed[0] + matrix[5] * fixed[1] + matrix[6] * fixed[2] + matrix[7],
        matrix[8] * fixed[0] + matrix[9] * fixed[1] + matrix[10] * fixed[2] + matrix[11],
    ];
    assert_eq!(mapped, moving);
}

#[test]
fn search_recovers_coupled_translation_optimum() {
    let target = [1.5, -2.0, 0.8];
    let objective = |transform: &AffineTransform| {
        let matrix = transform.as_array();
        let residual = [
            matrix[3] - target[0],
            matrix[7] - target[1],
            matrix[11] - target[2],
        ];
        Ok(-(residual[0] * residual[0]
            + residual[1] * residual[1]
            + residual[2] * residual[2]
            + 0.75 * (residual[0] + residual[1]).powi(2)
            + 0.75 * (residual[1] + residual[2]).powi(2)))
    };
    let result = search_rigid_pose([0.0; 3], [0.0; 3], config(), objective, objective)
        .expect("finite objective");
    let matrix = result.capture_transform.as_array();
    for (actual, expected) in [matrix[3], matrix[7], matrix[11]].into_iter().zip(target) {
        assert!(
            (actual - expected).abs() < 0.1,
            "got {actual}, expected {expected}"
        );
    }
}

#[test]
fn structural_search_cannot_leave_terminal_capture_cell() {
    let capture = |transform: &AffineTransform| {
        let matrix = transform.as_array();
        Ok(-(matrix[3].powi(2) + matrix[7].powi(2) + matrix[11].powi(2)))
    };
    let structural = |transform: &AffineTransform| {
        let translation = transform.as_array()[3];
        Ok(-(translation - 20.0).powi(2))
    };
    let result = search_rigid_pose([0.0; 3], [0.0; 3], config(), capture, structural)
        .expect("finite objectives");
    assert!(result.structural_transform.as_array()[3].abs() <= 0.75);
    assert!(result.structural_saturated);
}

#[test]
fn default_structural_radius_equals_explicit_one_cell() {
    let implicit = config();
    assert_eq!(implicit.structural_half_range_cells(), NonZeroU8::MIN);
    let explicit = config().with_structural_half_range_cells(NonZeroU8::MIN);

    let implicit_result = search_rigid_pose(
        [0.0; 3],
        [0.0; 3],
        implicit,
        pose_objective(0.0),
        pose_objective(0.5),
    )
    .expect("finite objectives");
    let explicit_result = search_rigid_pose(
        [0.0; 3],
        [0.0; 3],
        explicit,
        pose_objective(0.0),
        pose_objective(0.5),
    )
    .expect("finite objectives");

    assert_eq!(implicit_result, explicit_result);
}

#[test]
fn wider_structural_radius_reaches_manufactured_optimum() {
    let target = 1.25;
    let one_cell = search_rigid_pose(
        [0.0; 3],
        [0.0; 3],
        config(),
        pose_objective(0.0),
        pose_objective(target),
    )
    .expect("finite objectives");
    let two_cells = search_rigid_pose(
        [0.0; 3],
        [0.0; 3],
        config().with_structural_half_range_cells(
            NonZeroU8::new(2).expect("invariant: two is nonzero"),
        ),
        pose_objective(0.0),
        pose_objective(target),
    )
    .expect("finite objectives");

    let one_cell_translation = one_cell.structural_transform.as_array()[3];
    let two_cell_translation = two_cells.structural_transform.as_array()[3];
    assert!(one_cell_translation <= TRANSLATION_RESOLUTION_MM);
    assert!(one_cell.structural_saturated);
    assert!(
        (two_cell_translation - target).abs() <= TRANSLATION_RESOLUTION_MM / 16.0,
        "two-cell translation {two_cell_translation} did not reach {target}"
    );
    assert!(!two_cells.structural_saturated);
}

#[test]
fn widened_structural_search_remains_inside_global_bounds() {
    let global_half_range_mm = 1.0;
    let terminal_resolution_mm = 0.25;
    let structural_target_mm = 4.0;
    let observed_half_range = Cell::new(0.0_f64);
    let structural = |transform: &AffineTransform| {
        let translation = transform.as_array()[3];
        observed_half_range.set(observed_half_range.get().max(translation.abs()));
        Ok(-(translation - structural_target_mm).powi(2))
    };
    let bounded =
        RigidSearchConfig::try_new(2.0, global_half_range_mm, 0.25, terminal_resolution_mm, 256)
            .expect("valid search configuration")
            .with_structural_half_range_cells(
                NonZeroU8::new(8).expect("invariant: eight is nonzero"),
            );

    let result = search_rigid_pose([0.0; 3], [0.0; 3], bounded, pose_objective(0.0), structural)
        .expect("finite objectives");
    let translation = result.structural_transform.as_array()[3];

    assert!(observed_half_range.get() <= global_half_range_mm);
    assert!(translation.abs() <= global_half_range_mm);
    assert!(
        (translation - global_half_range_mm).abs() <= terminal_resolution_mm / 16.0,
        "globally clipped translation {translation} did not reach {global_half_range_mm}"
    );
    assert!(result.structural_saturated);
}

#[test]
fn structural_simplex_moves_inward_from_signed_multiple_global_bounds() {
    let global_half_range_mm = 1.0;
    let terminal_resolution_mm = 0.25;
    let bounded =
        RigidSearchConfig::try_new(2.0, global_half_range_mm, 0.25, terminal_resolution_mm, 256)
            .expect("valid search configuration")
            .with_structural_half_range_cells(
                NonZeroU8::new(8).expect("invariant: eight is nonzero"),
            );

    for sign in [-1.0, 1.0] {
        let signed_bound = sign * global_half_range_mm;
        let result = search_rigid_pose(
            [0.0; 3],
            [0.0; 3],
            bounded,
            finite_bounded_pose_objective([signed_bound, signed_bound, 0.0], global_half_range_mm),
            finite_bounded_pose_objective([0.0; 3], global_half_range_mm),
        )
        .expect("finite objectives");
        let capture = result.capture_transform.as_array();
        let structural = result.structural_transform.as_array();

        assert!((capture[3] - signed_bound).abs() <= terminal_resolution_mm / 16.0);
        assert!((capture[7] - signed_bound).abs() <= terminal_resolution_mm / 16.0);
        assert!(structural[3].abs() <= terminal_resolution_mm / 16.0);
        assert!(structural[7].abs() <= terminal_resolution_mm / 16.0);
    }
}

#[test]
fn overflowing_requested_radius_produces_finite_effective_intervals_and_steps() {
    let finite_bound = 1.0e307;
    let cells = NonZeroU8::new(u8::MAX).expect("invariant: u8 maximum is nonzero");

    let (intervals, steps) = structural_interval_and_step(
        [0.0; PARAMETER_COUNT],
        [finite_bound; PARAMETER_COUNT],
        [finite_bound; PARAMETER_COUNT],
        cells,
    );

    assert!(intervals.iter().flatten().all(|value| value.is_finite()));
    assert!(steps.iter().all(|value| value.is_finite()));
    assert_eq!(intervals, [[-finite_bound, finite_bound]; PARAMETER_COUNT]);
    assert_eq!(steps, [finite_bound; PARAMETER_COUNT]);
}

#[test]
fn extreme_finite_configuration_never_evaluates_a_nonfinite_pose() {
    let extreme = RigidSearchConfig::try_new(2.0, f64::MAX, 0.25, f64::MAX, 32)
        .expect("finite extreme configuration");
    let capture_evaluations = Cell::new(0_usize);
    let structural_evaluations = Cell::new(0_usize);
    let capture = |transform: &AffineTransform| {
        assert!(transform.as_array().iter().all(|value| value.is_finite()));
        capture_evaluations.set(capture_evaluations.get() + 1);
        Ok(0.0)
    };
    let structural = |transform: &AffineTransform| {
        assert!(transform.as_array().iter().all(|value| value.is_finite()));
        structural_evaluations.set(structural_evaluations.get() + 1);
        Ok(0.0)
    };

    let result = search_rigid_pose([0.0; 3], [0.0; 3], extreme, capture, structural)
        .expect("bounded extreme search");

    assert!(capture_evaluations.get() > SIMPLEX_VERTEX_COUNT);
    assert!(structural_evaluations.get() > SIMPLEX_VERTEX_COUNT);
    assert!(result
        .capture_transform
        .as_array()
        .iter()
        .chain(result.structural_transform.as_array().iter())
        .all(|value| value.is_finite()));
}

#[test]
fn extreme_finite_centroid_overflow_is_rejected_before_objective_evaluation() {
    let extreme = RigidSearchConfig::try_new(2.0, f64::MAX, 0.25, f64::MAX, 32)
        .expect("finite extreme configuration");

    for sign in [-1.0, 1.0] {
        let capture_evaluations = Cell::new(0_usize);
        let capture = |transform: &AffineTransform| {
            assert!(transform.as_array().iter().all(|value| value.is_finite()));
            capture_evaluations.set(capture_evaluations.get() + 1);
            Ok(0.0)
        };
        let structural = |transform: &AffineTransform| {
            assert!(transform.as_array().iter().all(|value| value.is_finite()));
            Ok(0.0)
        };

        let error = search_rigid_pose(
            [0.0; 3],
            [sign * f64::MAX, 0.0, 0.0],
            extreme,
            capture,
            structural,
        )
        .expect_err("centroid-plus-residual overflow must fail before the objective");

        assert!(capture_evaluations.get() > 0);
        assert!(matches!(
            error,
            RegistrationError::NumericalFailure(message)
                if message == "rigid-search candidate produced a non-finite transform"
        ));
    }
}

#[test]
fn structural_saturation_uses_configured_half_range() {
    let radius = NonZeroU8::new(2).expect("invariant: two is nonzero");
    let interior = search_rigid_pose(
        [0.0; 3],
        [0.0; 3],
        config().with_structural_half_range_cells(radius),
        pose_objective(0.0),
        pose_objective(0.75),
    )
    .expect("finite objectives");
    let clipped = search_rigid_pose(
        [0.0; 3],
        [0.0; 3],
        config().with_structural_half_range_cells(radius),
        pose_objective(0.0),
        pose_objective(4.0),
    )
    .expect("finite objectives");

    assert!(!interior.structural_saturated);
    assert!(clipped.structural_saturated);
    assert!(
        (clipped.structural_transform.as_array()[3] - 2.0 * TRANSLATION_RESOLUTION_MM).abs()
            <= TRANSLATION_RESOLUTION_MM / 16.0
    );
}

#[test]
fn configuration_rejects_invalid_resource_bounds() {
    let error = RigidSearchConfig::try_new(0.0, 8.0, 0.5, 0.75, 256)
        .expect_err("zero rotation range must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "rigid-search ranges and resolutions must be finite and positive, got [0.0, 8.0, 0.5, 0.75]"
    ));
    let error = RigidSearchConfig::try_new(12.0, 8.0, 13.0, 0.75, 256)
        .expect_err("terminal resolution beyond the range must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "rigid-search terminal resolution [13 deg, 0.75 mm] exceeds half-range [12 deg, 8 mm]"
    ));
    let error = RigidSearchConfig::try_new(12.0, 8.0, 0.5, 0.75, 0)
        .expect_err("zero iteration limit must be rejected");
    assert!(matches!(
        error,
        RegistrationError::InvalidInput(message)
            if message == "rigid-search simplex iteration limit must be positive"
    ));
}

#[test]
fn search_propagates_objective_failure() {
    let failure = |_transform: &AffineTransform| {
        Err(RegistrationError::InvalidInput(
            "fixture objective failure".to_owned(),
        ))
    };
    let result = search_rigid_pose([0.0; 3], [0.0; 3], config(), failure, failure);
    assert!(matches!(
        result,
        Err(RegistrationError::InvalidInput(message))
            if message == "fixture objective failure"
    ));
}
