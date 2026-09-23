//! Typed failures of Gibbs-ringing removal.

/// Reasons a Gibbs-ringing removal request is rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum GibbsError {
    /// The total-variation window starts after it ends.
    #[error("total-variation window [{start}, {end}] must satisfy start <= end")]
    InvalidWindow {
        /// First neighbour offset of the window.
        start: u16,
        /// Last neighbour offset of the window.
        end: u16,
    },
    /// An in-plane line is too short for the window to measure both sides of
    /// a voxel without the periodic wrap reusing a sample.
    ///
    /// The left and right windows of a voxel together span `2 · (end + 1) + 1`
    /// samples.
    #[error("in-plane axis {axis} has {len} samples; the total-variation window needs at least {minimum}")]
    LineTooShort {
        /// Axis of the image shape that is too short.
        axis: usize,
        /// Its length.
        len: usize,
        /// Minimum length the window needs.
        minimum: usize,
    },
    /// A line is longer than the phase-ramp index arithmetic represents.
    #[error("in-plane axis {axis} has {len} samples; at most {maximum} are supported")]
    LineTooLong {
        /// Axis of the image shape that is too long.
        axis: usize,
        /// Its length.
        len: usize,
        /// Maximum supported length.
        maximum: usize,
    },
    /// A volume's length disagrees with the image shape.
    #[error("volume {volume} holds {len} samples; the shape {shape:?} needs {expected}")]
    VolumeLength {
        /// Index of the offending volume.
        volume: usize,
        /// Its length.
        len: usize,
        /// The length the shape implies.
        expected: usize,
        /// The image shape.
        shape: [usize; 3],
    },
    /// A sample is NaN or infinite; the Fourier transform would spread it
    /// over its whole slice.
    #[error("volume {volume} sample {sample} is not finite")]
    NonFinite {
        /// Index of the offending volume.
        volume: usize,
        /// Index of the offending sample within it.
        sample: usize,
    },
}
