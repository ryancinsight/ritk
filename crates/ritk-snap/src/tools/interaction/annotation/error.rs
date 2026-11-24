//! Failures raised by annotation coordinates and computations.

/// Failure raised when measurement inputs or results cannot be
/// represented without changing their meaning.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MeasurementError {
    /// An image endpoint contains a nonfinite row or column coordinate.
    #[error("image point {point} component {component} is not finite: {value}")]
    NonFiniteImageCoordinate {
        /// Endpoint index in the supplied pair.
        point: usize,
        /// Row (`0`) or column (`1`) component index.
        component: usize,
        /// Rejected coordinate.
        value: f32,
    },
    /// One of the physical segments has no direction.
    #[error("{kind} needs two nonzero segments")]
    DegenerateSegment { kind: &'static str },
    /// A spacing component is not finite and strictly positive.
    #[error("voxel spacing component {index} is not finite and positive: {value}")]
    InvalidSpacing { index: usize, value: f64 },
    /// A valid physical spacing component cannot be represented by the
    /// f32-backed annotation storage used by the viewer.
    #[error("voxel spacing component {index} cannot be represented as f32: {value}")]
    UnrepresentableSpacing { index: usize, value: f64 },
    /// A computed measurement value overflowed or became non-finite.
    #[error("{kind} measurement result is not finite")]
    NonFiniteResult { kind: &'static str },
    /// A measurement was requested without a loaded volume.
    #[error("measurement requires a loaded volume")]
    MissingVolume,
    /// The source did not provide a trustworthy in-plane physical spacing.
    #[error("physical pixel spacing is unavailable for this image")]
    MissingPhysicalSpacing,
}
