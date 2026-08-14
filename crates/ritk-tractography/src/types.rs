use gaia::{Polyline, PolylineError};
use ritk_spatial::Point;

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
    pub(crate) geometry: Polyline<f64>,
    pub(crate) forward_termination: TerminationReason,
    pub(crate) backward_termination: Option<TerminationReason>,
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
    pub(crate) streamlines: Box<[Streamline]>,
    pub(crate) seeds_attempted: usize,
}

impl TractographyResult {
    /// Generated streamlines in seed order, excluding untrackable seeds.
    #[must_use]
    pub fn streamlines(&self) -> &[Streamline] {
        &self.streamlines
    }

    /// A copy with every point mapped through `transform`.
    ///
    /// Tracking runs in whatever frame the direction field is defined in, which
    /// for a voxel-index field is not the frame a tractogram should be written
    /// in — a file is only meaningful beside the anatomy it came from. The
    /// export methods on this type all assume physical millimetres, so a
    /// caller tracking in indices converts first.
    ///
    /// Termination reasons are carried over unchanged: they describe why
    /// tracking stopped, which no change of coordinates affects.
    ///
    /// # Errors
    ///
    /// [`TractographyError`] if a mapped streamline is no longer a valid
    /// polyline, which a transform collapsing distinct points would cause.
    pub fn map_points(
        &self,
        transform: impl Fn(&Point<3>) -> Point<3>,
    ) -> Result<Self, TractographyError> {
        let mut streamlines = Vec::new();
        streamlines
            .try_reserve_exact(self.streamlines.len())
            .map_err(|_| TractographyError::Allocation {
                requested: self.streamlines.len(),
            })?;

        for streamline in &self.streamlines {
            let points: Vec<_> = streamline
                .geometry
                .points()
                .iter()
                .map(|point| {
                    let mapped = transform(&Point::new([point.x, point.y, point.z]));
                    let [x, y, z] = mapped.to_array();
                    leto::geometry::Point3::new(x, y, z)
                })
                .collect();
            streamlines.push(Streamline {
                geometry: Polyline::new(points).map_err(TractographyError::from)?,
                forward_termination: streamline.forward_termination,
                backward_termination: streamline.backward_termination,
            });
        }

        Ok(Self {
            streamlines: streamlines.into_boxed_slice(),
            seeds_attempted: self.seeds_attempted,
        })
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
