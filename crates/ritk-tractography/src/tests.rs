use super::*;

fn horizontal(_: &Point<3>) -> Option<Vector<3>> {
    Some(Vector::new([1.0, 0.0, 0.0]))
}

#[test]
fn configuration_rejects_every_invalid_partition() {
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            TractographyConfig::new(step, 10, 60.0, TrackingDirection::Forward),
            Err(TractographyError::InvalidStepSize { .. })
        ));
    }
    assert!(matches!(
        TractographyConfig::new(1.0, 0, 60.0, TrackingDirection::Forward),
        Err(TractographyError::InvalidMaxSteps { .. })
    ));
    for angle in [-1.0, 181.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            TractographyConfig::new(1.0, 10, angle, TrackingDirection::Forward),
            Err(TractographyError::InvalidTurnLimit { .. })
        ));
    }
}

#[test]
fn forward_step_limit_has_exact_geometry_and_reason() -> Result<(), TractographyError> {
    let config = TractographyConfig::new(1.0, 5, 60.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, horizontal)?;
    let line = &result.streamlines()[0];
    assert_eq!(line.geometry().len(), 6);
    assert_eq!(line.geometry().points()[0].x, 0.0);
    assert_eq!(line.geometry().points()[5].x, 5.0);
    assert_eq!(line.forward_termination(), TerminationReason::StepLimit);
    Ok(())
}

#[test]
fn boundary_point_is_not_appended() -> Result<(), TractographyError> {
    let field = |point: &Point<3>| (point[0] <= 5.0).then_some(Vector::new([1.0, 0.0, 0.0]));
    let config = TractographyConfig::new(1.0, 10, 60.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([3.0, 0.0, 0.0])], config, field)?;
    let line = &result.streamlines()[0];
    assert_eq!(line.geometry().len(), 3);
    assert_eq!(line.geometry().points()[2].x, 5.0);
    assert_eq!(line.forward_termination(), TerminationReason::FieldBoundary);
    Ok(())
}

#[test]
fn bidirectional_join_contains_seed_once() -> Result<(), TractographyError> {
    let config = TractographyConfig::new(1.0, 3, 60.0, TrackingDirection::Bidirectional)?;
    let result = euler_tractography(&[Point::new([2.0, 0.0, 0.0])], config, horizontal)?;
    let points = result.streamlines()[0].geometry().points();
    assert_eq!(points.len(), 7);
    assert_eq!(points[0].x, -1.0);
    assert_eq!(points[3].x, 2.0);
    assert_eq!(points[6].x, 5.0);
    assert_eq!(
        result.streamlines()[0].backward_termination(),
        Some(TerminationReason::StepLimit)
    );
    Ok(())
}

#[test]
fn sharp_turn_stops_at_last_valid_point() -> Result<(), TractographyError> {
    let field = |point: &Point<3>| {
        if point[0] < 1.0 {
            Some(Vector::new([1.0, 0.0, 0.0]))
        } else {
            Some(Vector::new([0.0, 1.0, 0.0]))
        }
    };
    let config = TractographyConfig::new(1.0, 5, 45.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, field)?;
    let line = &result.streamlines()[0];
    assert_eq!(line.geometry().len(), 2);
    assert_eq!(line.forward_termination(), TerminationReason::TurningAngle);
    Ok(())
}

#[test]
fn invalid_field_direction_is_an_error_not_a_dropped_line() -> Result<(), TractographyError> {
    let config = TractographyConfig::new(1.0, 5, 45.0, TrackingDirection::Forward)?;
    let error = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, |_| {
        Some(Vector::new([2.0, 0.0, 0.0]))
    })
    .expect_err("non-unit direction");
    assert!(matches!(
        error,
        TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            ..
        }
    ));
    Ok(())
}

#[test]
fn empty_and_untrackable_seed_sets_are_value_exact() -> Result<(), TractographyError> {
    let config = TractographyConfig::default();
    let empty = euler_tractography(&[], config, horizontal)?;
    assert_eq!(empty.seeds_attempted(), 0);
    assert_eq!(empty.streamlines_generated(), 0);
    let absent = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, |_| None)?;
    assert_eq!(absent.seeds_attempted(), 1);
    assert_eq!(absent.streamlines_generated(), 0);
    Ok(())
}

#[test]
fn non_finite_seed_fails_before_field_callback() {
    let callback_called = std::cell::Cell::new(false);
    let error = euler_tractography(
        &[Point::new([f64::NAN, 0.0, 0.0])],
        TractographyConfig::default(),
        |_| {
            callback_called.set(true);
            None
        },
    )
    .expect_err("non-finite seed");
    assert!(!callback_called.get());
    assert!(matches!(
        error,
        TractographyError::NonFinitePoint {
            seed_index: 0,
            step_index: 0,
            ..
        }
    ));
}
