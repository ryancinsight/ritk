//! Every way a fused compare slice can be refused.
//!
//! The taxonomy is the contract the renderer offers its caller: each variant
//! names one precondition, so a caller can tell a geometry mismatch from an
//! empty volume without parsing a message.

use thiserror::Error;

use crate::geometry::affine::AffineError;

/// Failure while validating or rendering a fused compare slice.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FusionError {
    /// A volume has a zero spatial dimension.
    #[error("{volume} volume has no voxels")]
    EmptyVolume {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
    },
    /// A volume declares no interleaved channels.
    #[error("{volume} volume declares zero channels")]
    InvalidChannelCount {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
    },
    /// Fused compare currently operates on scalar presentation values only.
    #[error("{volume} fused rendering requires a scalar volume; received {channels} channels")]
    UnsupportedChannelCount {
        /// The side of the comparison with an unsupported channel layout.
        volume: &'static str,
        /// Number of interleaved channels declared by the volume.
        channels: u8,
    },
    /// A volume declares grayscale presentation metadata the renderer cannot
    /// resolve to an admitted DICOM function.
    #[error("{volume} volume has invalid grayscale presentation metadata: {source}")]
    InvalidPresentation {
        /// The side of the comparison with invalid presentation metadata.
        volume: &'static str,
        /// Presentation metadata failure.
        #[source]
        source: crate::render::GrayscalePresentationError,
    },
    /// The volume's shape and channel count overflow the sample index space.
    #[error("{volume} volume has an overflowing sample layout")]
    InvalidVolumeLayout {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
    },
    /// The volume data length does not match its declared shape and channels.
    #[error(
        "{volume} volume contains {actual} samples; the declared geometry requires {expected}"
    )]
    InvalidSampleCount {
        /// The side of the comparison with invalid data.
        volume: &'static str,
        /// Number of samples present in the buffer.
        actual: usize,
        /// Number of samples required by the declared layout.
        expected: usize,
    },
    /// A slice axis is outside the three spatial dimensions.
    #[error("{axis} is not a valid slice axis; expected 0, 1, or 2")]
    InvalidAxis {
        /// The invalid axis value.
        axis: usize,
    },
    /// A selected slice is outside its volume's extent.
    #[error("{volume} slice {slice} is outside axis {axis} extent {extent}")]
    SliceOutOfRange {
        /// The side of the comparison with the invalid selection.
        volume: &'static str,
        /// The slice axis.
        axis: usize,
        /// The requested index.
        slice: usize,
        /// The valid axis extent.
        extent: usize,
    },
    /// The blend weight is not a finite value.
    #[error("secondary blend weight must be finite")]
    InvalidBlendWeight,
    /// A volume's physical geometry cannot be inverted or contains invalid values.
    #[error("{volume} volume geometry is invalid: {source}")]
    InvalidGeometry {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
        /// The affine validation failure.
        #[source]
        source: AffineError,
    },
    /// The two volumes have different explicit DICOM frame identifiers.
    #[error("primary and secondary frame of reference identifiers differ")]
    IncompatibleFrameOfReference,
    /// A differing grid cannot be trusted without two explicit frame identifiers.
    #[error("a differing grid requires frame of reference identifiers on both volumes")]
    MissingFrameOfReference,
    /// The selected primary and secondary planes are not parallel in patient space.
    #[error("primary and secondary slice planes are not parallel")]
    NonParallelPlanes,
    /// The selected secondary plane is not the patient-space plane requested by the primary.
    #[error("secondary slice does not coincide with the selected primary patient-space plane")]
    PlaneMismatch,
    /// The selected planes do not intersect the secondary volume along its normal.
    #[error("primary and secondary volumes do not overlap along the selected slice normal")]
    NoPhysicalOverlap,
}
