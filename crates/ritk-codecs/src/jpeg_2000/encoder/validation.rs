use std::fmt;

/// Maximum sample precision supported by the current `i32` entropy path.
///
/// DICOM permits larger JPEG 2000 precisions, but accepting them here would
/// either truncate the caller's samples or overflow the reversible lifting
/// arithmetic. The API rejects that unsupported representation explicitly.
const MAX_SAMPLE_PRECISION: u32 = 16;

/// Errors returned by [`super::encode_grayscale_j2k`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Jpeg2000EncodeError {
    /// At least one image dimension is zero.
    EmptyImage {
        /// Requested row count.
        rows: u32,
        /// Requested column count.
        cols: u32,
    },
    /// The dimensions cannot be represented or multiplied on this platform.
    DimensionOverflow {
        /// Requested row count.
        rows: u32,
        /// Requested column count.
        cols: u32,
    },
    /// The sample slice length does not equal `rows × cols`.
    PixelCountMismatch {
        /// Number of samples supplied by the caller.
        actual: usize,
        /// Number of samples required by the geometry.
        expected: usize,
    },
    /// The declared component precision is unsupported.
    UnsupportedPrecision {
        /// Requested precision.
        precision: u32,
        /// Largest precision accepted by the current integer path.
        maximum: u32,
    },
    /// The requested decomposition depth exceeds the image geometry.
    ExcessiveDecomposition {
        /// Requested number of decomposition levels.
        requested: u8,
        /// Largest meaningful depth for the requested geometry.
        maximum: u8,
    },
    /// A sample is outside the range declared by precision and signedness.
    SampleOutOfRange {
        /// Zero-based index of the first invalid sample.
        index: usize,
        /// Invalid sample value.
        value: i32,
        /// Inclusive lower bound.
        minimum: i32,
        /// Inclusive upper bound.
        maximum: i32,
    },
}

impl fmt::Display for Jpeg2000EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage { rows, cols } => {
                write!(
                    formatter,
                    "JPEG 2000 image dimensions must be nonzero; got {rows}×{cols}"
                )
            }
            Self::DimensionOverflow { rows, cols } => write!(
                formatter,
                "JPEG 2000 image dimensions {rows}×{cols} exceed this platform's address space"
            ),
            Self::PixelCountMismatch { actual, expected } => write!(
                formatter,
                "JPEG 2000 sample count mismatch: got {actual}, expected {expected}"
            ),
            Self::UnsupportedPrecision { precision, maximum } => write!(
                formatter,
                "JPEG 2000 precision {precision} is unsupported; expected 1..={maximum}"
            ),
            Self::ExcessiveDecomposition { requested, maximum } => write!(
                formatter,
                "JPEG 2000 decomposition depth {requested} exceeds geometry limit {maximum}"
            ),
            Self::SampleOutOfRange {
                index,
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "JPEG 2000 sample[{index}]={value} is outside [{minimum}, {maximum}]"
            ),
        }
    }
}

impl std::error::Error for Jpeg2000EncodeError {}

pub(super) fn validate_geometry(
    actual_samples: usize,
    rows: u32,
    cols: u32,
    num_decomp_levels: u8,
) -> Result<(usize, usize), Jpeg2000EncodeError> {
    if rows == 0 || cols == 0 {
        return Err(Jpeg2000EncodeError::EmptyImage { rows, cols });
    }
    let height =
        usize::try_from(rows).map_err(|_| Jpeg2000EncodeError::DimensionOverflow { rows, cols })?;
    let width =
        usize::try_from(cols).map_err(|_| Jpeg2000EncodeError::DimensionOverflow { rows, cols })?;
    let expected = width
        .checked_mul(height)
        .ok_or(Jpeg2000EncodeError::DimensionOverflow { rows, cols })?;
    if actual_samples != expected {
        return Err(Jpeg2000EncodeError::PixelCountMismatch {
            actual: actual_samples,
            expected,
        });
    }

    let max_dimension = width.max(height);
    let maximum_levels = usize::BITS - (max_dimension - 1).leading_zeros();
    let maximum_levels = u8::try_from(maximum_levels)
        .expect("invariant: usize::BITS fits in u8 on supported Rust targets");
    if num_decomp_levels > maximum_levels {
        return Err(Jpeg2000EncodeError::ExcessiveDecomposition {
            requested: num_decomp_levels,
            maximum: maximum_levels,
        });
    }

    Ok((width, height))
}

pub(super) fn validate_precision(
    precision: u32,
    is_signed: bool,
) -> Result<(i32, i32), Jpeg2000EncodeError> {
    if !(1..=MAX_SAMPLE_PRECISION).contains(&precision) {
        return Err(Jpeg2000EncodeError::UnsupportedPrecision {
            precision,
            maximum: MAX_SAMPLE_PRECISION,
        });
    }

    let magnitude = 1i32 << (precision - 1);
    if is_signed {
        Ok((-magnitude, magnitude - 1))
    } else {
        Ok((0, (magnitude << 1) - 1))
    }
}
