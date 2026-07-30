//! Typed failures for descriptive statistics and histograms.

/// Failure returned when descriptive statistics or histograms cannot be
/// computed without producing an undefined or misleading result.
#[derive(Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum StatisticsError {
    /// The input contains no samples.
    #[error("statistics input must contain at least one sample")]
    EmptyInput,
    /// A sample is NaN or infinite.
    #[error("sample at index {index} must be finite, got {value}")]
    NonFiniteSample {
        /// Zero-based sample index.
        index: usize,
        /// Invalid sample value.
        value: f32,
    },
    /// A mask value is NaN or infinite.
    #[error("mask sample at index {index} must be finite, got {value}")]
    NonFiniteMaskSample {
        /// Zero-based mask index.
        index: usize,
        /// Invalid mask value.
        value: f32,
    },
    /// The requested degrees-of-freedom correction leaves no divisor.
    #[error("ddof {ddof} must be less than the sample count {sample_count}")]
    DegreesOfFreedomOutOfRange {
        /// Number of available samples.
        sample_count: usize,
        /// Requested delta degrees of freedom.
        ddof: usize,
    },
    /// Image and mask buffers have different element counts.
    #[error("image element count {image_count} does not match mask element count {mask_count}")]
    ImageMaskLengthMismatch {
        /// Number of image samples.
        image_count: usize,
        /// Number of mask samples.
        mask_count: usize,
    },
    /// The mask selects no foreground samples.
    #[error("mask contains no foreground samples")]
    EmptyForeground,
    /// A histogram cannot have zero bins.
    #[error("histogram bin count must be greater than zero")]
    ZeroBins,
    /// At least one histogram range bound is NaN or infinite.
    #[error("histogram range bounds must be finite, got min={min}, max={max}")]
    NonFiniteRange {
        /// Lower range bound.
        min: f32,
        /// Upper range bound.
        max: f32,
    },
    /// Histogram bounds are finite but not strictly increasing.
    #[error("histogram min {min} must be strictly less than max {max}")]
    InvalidRange {
        /// Lower range bound.
        min: f32,
        /// Upper range bound.
        max: f32,
    },
    /// The requested count buffer cannot be allocated.
    #[error("histogram count buffer for {bins} bins cannot be allocated")]
    HistogramAllocationFailed {
        /// Requested number of bins.
        bins: usize,
    },
}
