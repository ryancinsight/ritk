use gaia::Polyline;
use ritk_spatial::{Point, Vector};

use crate::types::{
    Streamline, TerminationReason, TrackingDirection, TractographyConfig, TractographyError,
    TractographyResult,
};

const UNIT_NORM_TOLERANCE: f64 = 1.0e-6;

struct IntegrationHalf {
    points: Vec<Point<3>>,
    termination: TerminationReason,
}

/// Track deterministic streamlines from physical seed points.
///
/// `direction_field` returns a unit orientation at a trackable physical point
/// and `None` outside the tracking domain. Orientations are sign-ambiguous;
/// the integrator aligns each sample with the preceding direction.
///
/// # Errors
///
/// Returns a typed error for an invalid field direction, non-finite proposed
/// point, allocation failure, or invalid Gaia geometry. A seed at which the
/// field returns `None`, or whose integration produces fewer than two points,
/// is an expected untrackable seed and produces no line.
pub fn euler_tractography<F>(
    seeds: &[Point<3>],
    config: TractographyConfig,
    direction_field: F,
) -> Result<TractographyResult, TractographyError>
where
    F: Fn(&Point<3>) -> Option<Vector<3>>,
{
    let mut streamlines = Vec::new();
    streamlines
        .try_reserve_exact(seeds.len())
        .map_err(|_| TractographyError::Allocation {
            requested: seeds.len(),
        })?;
    let cosine_limit = config.max_turn_degrees().to_radians().cos();

    for (seed_index, seed) in seeds.iter().copied().enumerate() {
        let coordinates = seed.to_array();
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err(TractographyError::NonFinitePoint {
                seed_index,
                step_index: 0,
                point: coordinates,
            });
        }
        let Some(initial) = direction_field(&seed) else {
            continue;
        };
        let initial = validate_direction(initial, seed_index, 0)?;
        let forward = integrate_half(
            seed,
            initial,
            config,
            cosine_limit,
            seed_index,
            &direction_field,
        )?;

        let (points, backward_termination) = match config.tracking_direction() {
            TrackingDirection::Forward => (forward.points, None),
            TrackingDirection::Bidirectional => {
                let backward = integrate_half(
                    seed,
                    -initial,
                    config,
                    cosine_limit,
                    seed_index,
                    &direction_field,
                )?;
                let capacity = backward
                    .points
                    .len()
                    .saturating_sub(1)
                    .checked_add(forward.points.len())
                    .ok_or(TractographyError::Allocation {
                        requested: usize::MAX,
                    })?;
                let mut joined = Vec::new();
                joined
                    .try_reserve_exact(capacity)
                    .map_err(|_| TractographyError::Allocation {
                        requested: capacity,
                    })?;
                joined.extend(backward.points.into_iter().rev());
                joined.extend(forward.points.into_iter().skip(1));
                (joined, Some(backward.termination))
            }
        };

        if points.len() < 2 {
            continue;
        }
        streamlines.push(Streamline {
            geometry: points_to_polyline(&points)?,
            forward_termination: forward.termination,
            backward_termination,
        });
    }

    Ok(TractographyResult {
        streamlines: streamlines.into_boxed_slice(),
        seeds_attempted: seeds.len(),
    })
}

fn integrate_half<F>(
    start: Point<3>,
    mut current_direction: Vector<3>,
    config: TractographyConfig,
    cosine_limit: f64,
    seed_index: usize,
    direction_field: &F,
) -> Result<IntegrationHalf, TractographyError>
where
    F: Fn(&Point<3>) -> Option<Vector<3>>,
{
    let capacity = config
        .max_steps()
        .checked_add(1)
        .ok_or(TractographyError::InvalidMaxSteps {
            value: config.max_steps(),
        })?;
    let mut points = Vec::new();
    points
        .try_reserve_exact(capacity)
        .map_err(|_| TractographyError::Allocation {
            requested: capacity,
        })?;
    points.push(start);
    let mut current_point = start;

    for step_index in 1..=config.max_steps() {
        let proposed = current_point + current_direction * config.step_size();
        let coordinates = proposed.to_array();
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err(TractographyError::NonFinitePoint {
                seed_index,
                step_index,
                point: coordinates,
            });
        }
        let Some(next_raw) = direction_field(&proposed) else {
            return Ok(IntegrationHalf {
                points,
                termination: TerminationReason::FieldBoundary,
            });
        };
        let mut next_direction = validate_direction(next_raw, seed_index, step_index)?;
        if current_direction.dot(&next_direction) < 0.0 {
            next_direction = -next_direction;
        }

        points.push(proposed);
        if current_direction.dot(&next_direction).clamp(-1.0, 1.0) < cosine_limit {
            return Ok(IntegrationHalf {
                points,
                termination: TerminationReason::TurningAngle,
            });
        }
        current_point = proposed;
        current_direction = next_direction;
    }

    Ok(IntegrationHalf {
        points,
        termination: TerminationReason::StepLimit,
    })
}

fn validate_direction(
    direction: Vector<3>,
    seed_index: usize,
    step_index: usize,
) -> Result<Vector<3>, TractographyError> {
    let components = direction.to_array();
    if components.iter().any(|value| !value.is_finite()) {
        return Err(TractographyError::InvalidDirection {
            seed_index,
            step_index,
            reason: format!("components are not finite: {components:?}"),
        });
    }
    let norm = direction.norm();
    if (norm - 1.0).abs() > UNIT_NORM_TOLERANCE {
        return Err(TractographyError::InvalidDirection {
            seed_index,
            step_index,
            reason: format!("expected unit norm, got {norm}"),
        });
    }
    Ok(direction)
}

fn points_to_polyline(points: &[Point<3>]) -> Result<Polyline<f64>, TractographyError> {
    let mut geometry_points = Vec::new();
    geometry_points
        .try_reserve_exact(points.len())
        .map_err(|_| TractographyError::Allocation {
            requested: points.len(),
        })?;
    geometry_points.extend(points.iter().map(|point| {
        let [x, y, z] = point.to_array();
        leto::geometry::Point3::new(x, y, z)
    }));
    Polyline::new(geometry_points).map_err(TractographyError::from)
}
