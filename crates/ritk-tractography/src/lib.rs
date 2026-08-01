//! Deterministic streamline tractography producing Gaia polyline geometry.
//!
//! RITK owns integration and termination policy; Gaia owns curve geometry.
//! The current strategy is explicit Euler stepping with direction continuity,
//! field-boundary, turning-angle, and step-count termination. It is intended
//! for deterministic examples and baseline algorithms, not as a clinical
//! tractography validation claim.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use gaia::{Polyline, PolylineError};
use ritk_spatial::{Point, Vector};

const UNIT_NORM_TOLERANCE: f64 = 1.0e-6;

/// Direction regimes supported by deterministic tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingDirection {
    /// Integrate only along the initial direction returned at the seed.
    Forward,
    /// Integrate both signs of the initial orientation and join at the seed.
    Bidirectional,
}

/// Reason one integration half stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// The direction field returned no trackable orientation at the proposal.
    FieldBoundary,
    /// The next valid orientation exceeded the configured turn limit.
    TurningAngle,
    /// The configured accepted-step limit was reached.
    StepLimit,
}

/// Validated deterministic Euler tracking configuration.
#[derive(Debug, Clone, Copy)]
pub struct TractographyConfig {
    step_size: f64,
    max_steps: usize,
    max_turn_degrees: f64,
    tracking_direction: TrackingDirection,
}

impl TractographyConfig {
    /// Construct a validated tracking configuration.
    ///
    /// # Errors
    ///
    /// Returns a typed error unless `step_size` is finite and positive,
    /// `max_steps` is nonzero and can represent the seed plus all steps, and
    /// `max_turn_degrees` is finite in `[0, 180]`.
    pub fn new(
        step_size: f64,
        max_steps: usize,
        max_turn_degrees: f64,
        tracking_direction: TrackingDirection,
    ) -> Result<Self, TractographyError> {
        if !step_size.is_finite() || step_size <= 0.0 {
            return Err(TractographyError::InvalidStepSize { value: step_size });
        }
        if max_steps == 0 || max_steps.checked_add(1).is_none() {
            return Err(TractographyError::InvalidMaxSteps { value: max_steps });
        }
        if !max_turn_degrees.is_finite() || !(0.0..=180.0).contains(&max_turn_degrees) {
            return Err(TractographyError::InvalidTurnLimit {
                value: max_turn_degrees,
            });
        }
        Ok(Self {
            step_size,
            max_steps,
            max_turn_degrees,
            tracking_direction,
        })
    }

    /// Integration step in physical image units.
    #[must_use]
    pub const fn step_size(self) -> f64 {
        self.step_size
    }

    /// Maximum accepted steps for each integration half.
    #[must_use]
    pub const fn max_steps(self) -> usize {
        self.max_steps
    }

    /// Maximum turn between consecutive orientations, in degrees.
    #[must_use]
    pub const fn max_turn_degrees(self) -> f64 {
        self.max_turn_degrees
    }

    /// Forward-only or bidirectional tracking regime.
    #[must_use]
    pub const fn tracking_direction(self) -> TrackingDirection {
        self.tracking_direction
    }
}

impl Default for TractographyConfig {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            max_steps: 1_000,
            max_turn_degrees: 60.0,
            tracking_direction: TrackingDirection::Bidirectional,
        }
    }
}

/// Failure while configuring or executing streamline integration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TractographyError {
    /// Step size is zero, negative, NaN, or infinite.
    #[error("tractography step size must be finite and positive, got {value}")]
    InvalidStepSize {
        /// Invalid physical step size.
        value: f64,
    },
    /// Step limit is zero or cannot represent the seed plus steps.
    #[error("tractography maximum steps must be positive and bounded, got {value}")]
    InvalidMaxSteps {
        /// Invalid step count.
        value: usize,
    },
    /// Turning limit is outside `[0, 180]` degrees or non-finite.
    #[error("tractography turn limit must be finite in [0, 180] degrees, got {value}")]
    InvalidTurnLimit {
        /// Invalid angle in degrees.
        value: f64,
    },
    /// A direction-field sample is non-finite or not unit length.
    #[error("invalid direction at seed {seed_index}, step {step_index}: {reason}")]
    InvalidDirection {
        /// Input seed index.
        seed_index: usize,
        /// Proposed integration step, zero at the seed.
        step_index: usize,
        /// Violated direction invariant.
        reason: String,
    },
    /// A proposed physical point became non-finite.
    #[error("non-finite point at seed {seed_index}, step {step_index}: {point:?}")]
    NonFinitePoint {
        /// Input seed index.
        seed_index: usize,
        /// Proposed integration step.
        step_index: usize,
        /// Invalid point coordinates.
        point: [f64; 3],
    },
    /// A bounded point or streamline allocation failed.
    #[error("tractography allocation failed for {requested} elements")]
    Allocation {
        /// Requested element capacity.
        requested: usize,
    },
    /// Gaia rejected generated polyline geometry.
    #[error("generated streamline geometry is invalid: {0}")]
    Geometry(#[from] PolylineError),
}

/// One generated streamline and its termination diagnostics.
#[derive(Debug, Clone)]
pub struct Streamline {
    geometry: Polyline<f64>,
    forward_termination: TerminationReason,
    backward_termination: Option<TerminationReason>,
}

impl Streamline {
    /// Gaia polyline geometry.
    #[must_use]
    pub const fn geometry(&self) -> &Polyline<f64> {
        &self.geometry
    }

    /// Termination reason along the seed's initial orientation.
    #[must_use]
    pub const fn forward_termination(&self) -> TerminationReason {
        self.forward_termination
    }

    /// Opposite-orientation termination for bidirectional tracking.
    #[must_use]
    pub const fn backward_termination(&self) -> Option<TerminationReason> {
        self.backward_termination
    }
}

/// Output of deterministic tractography over a seed set.
#[derive(Debug, Clone)]
pub struct TractographyResult {
    streamlines: Box<[Streamline]>,
    seeds_attempted: usize,
}

impl TractographyResult {
    /// Generated streamlines in seed order, excluding untrackable seeds.
    #[must_use]
    pub fn streamlines(&self) -> &[Streamline] {
        &self.streamlines
    }

    /// Number of input seeds queried.
    #[must_use]
    pub const fn seeds_attempted(&self) -> usize {
        self.seeds_attempted
    }

    /// Number of generated streamlines.
    #[must_use]
    pub fn streamlines_generated(&self) -> usize {
        self.streamlines.len()
    }
}

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
/// field returns `None` is an expected untrackable seed and produces no line.
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
    let cosine_limit = config.max_turn_degrees.to_radians().cos();

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

        let (points, backward_termination) = match config.tracking_direction {
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
        .max_steps
        .checked_add(1)
        .ok_or(TractographyError::InvalidMaxSteps {
            value: config.max_steps,
        })?;
    let mut points = Vec::new();
    points
        .try_reserve_exact(capacity)
        .map_err(|_| TractographyError::Allocation {
            requested: capacity,
        })?;
    points.push(start);
    let mut current_point = start;

    for step_index in 1..=config.max_steps {
        let proposed = current_point + current_direction * config.step_size;
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

#[cfg(test)]
mod tests;
