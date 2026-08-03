//! Acquisition-scheme validation errors.

/// Failure while parsing or validating diffusion acquisition metadata.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum GradientSchemeError {
    /// The acquisition scheme contains no volumes.
    #[error("gradient scheme must contain at least one direction")]
    Empty,
    /// A diffusion weighting is negative or non-finite.
    #[error("diffusion weighting at index {index} is invalid: {value} s/mm^2")]
    InvalidWeighting {
        /// Acquisition-order index, or zero for a standalone value.
        index: usize,
        /// Invalid value in seconds per square millimeter.
        value: f64,
    },
    /// A gradient vector violates the finite zero/unit-vector contract.
    #[error("gradient direction at index {index} is invalid: {reason}")]
    InvalidDirection {
        /// Acquisition-order index, or zero for a standalone vector.
        index: usize,
        /// Contextual invariant violation.
        reason: String,
    },
    /// A token in a metadata file is not a finite number.
    #[error("invalid {field} token '{token}'")]
    InvalidToken {
        /// Metadata field being parsed.
        field: &'static str,
        /// Offending token.
        token: String,
    },
    /// FSL b-vector rows do not have the required shape.
    #[error("invalid FSL b-vector table: {0}")]
    InvalidBVectorTable(String),
    /// MRtrix .mif DW_scheme header is malformed or inconsistent.
    #[error("invalid MRtrix DW_scheme: {0}")]
    InvalidMrtrixHeader(String),
    /// Weighting and direction counts differ.
    #[error("weighting count ({weightings}) does not match direction count ({directions})")]
    LengthMismatch {
        /// Number of diffusion weightings.
        weightings: usize,
        /// Number of gradient directions.
        directions: usize,
    },
    /// A rotation matrix is non-finite, non-orthonormal, or not proper.
    #[error("invalid gradient rotation: {0}")]
    InvalidRotation(String),
    /// Per-volume reorientation was given a rotation count other than one per
    /// volume.
    ///
    /// Silently zipping to the shorter length would leave the tail of the
    /// series unrotated, which is the defect per-volume reorientation exists to
    /// prevent.
    #[error("expected {expected} rotations, one per volume, got {actual}")]
    RotationCountMismatch {
        /// Volumes in the scheme.
        expected: usize,
        /// Rotations supplied.
        actual: usize,
    },
}
