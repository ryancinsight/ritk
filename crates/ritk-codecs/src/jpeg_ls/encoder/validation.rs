use std::fmt;

const MIN_SAMPLE_PRECISION: u32 = 8;
const MAX_SAMPLE_PRECISION: u32 = 16;
const MAX_FRAME_DIMENSION: u32 = u16::MAX as u32;
const MAX_HEADER_NEAR: u32 = u8::MAX as u32;

/// Errors returned by [`super::encode_grayscale_jpeg_ls`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum JpegLsEncodeError {
    /// At least one image dimension is zero.
    EmptyImage {
        /// Requested row count.
        rows: u32,
        /// Requested column count.
        cols: u32,
    },
    /// A dimension cannot be represented by the JPEG-LS frame header.
    DimensionOutOfRange {
        /// Requested row count.
        rows: u32,
        /// Requested column count.
        cols: u32,
        /// Largest representable row or column count.
        maximum: u32,
    },
    /// The dimensions cannot be multiplied or represented on this platform.
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
        /// Smallest supported precision.
        minimum: u32,
        /// Largest supported precision.
        maximum: u32,
    },
    /// The near-lossless error bound exceeds the coded sample range.
    NearOutOfRange {
        /// Requested near-lossless error bound.
        near: u32,
        /// Largest bound supported by the precision and SOS representation.
        maximum: u32,
    },
    /// A sample is outside the range declared by the precision.
    SampleOutOfRange {
        /// Zero-based index of the first invalid sample.
        index: usize,
        /// Invalid sample value.
        value: u16,
        /// Inclusive upper bound.
        maximum: u16,
    },
}

impl fmt::Display for JpegLsEncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage { rows, cols } => write!(
                formatter,
                "JPEG-LS image dimensions must be nonzero; got {rows}×{cols}"
            ),
            Self::DimensionOutOfRange {
                rows,
                cols,
                maximum,
            } => write!(
                formatter,
                "JPEG-LS image dimensions {rows}×{cols} exceed the {maximum}-sample frame-header limit"
            ),
            Self::DimensionOverflow { rows, cols } => write!(
                formatter,
                "JPEG-LS image dimensions {rows}×{cols} exceed this platform's address space"
            ),
            Self::PixelCountMismatch { actual, expected } => write!(
                formatter,
                "JPEG-LS sample count mismatch: got {actual}, expected {expected}"
            ),
            Self::UnsupportedPrecision {
                precision,
                minimum,
                maximum,
            } => write!(
                formatter,
                "JPEG-LS precision {precision} is unsupported; expected {minimum}..={maximum}"
            ),
            Self::NearOutOfRange { near, maximum } => write!(
                formatter,
                "JPEG-LS NEAR={near} exceeds the precision-dependent limit {maximum}"
            ),
            Self::SampleOutOfRange {
                index,
                value,
                maximum,
            } => write!(
                formatter,
                "JPEG-LS sample[{index}]={value} exceeds the declared maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for JpegLsEncodeError {}

pub(super) struct ValidatedEncoding {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) precision: u8,
    pub(super) near: i32,
}

pub(super) fn validate_encoding(
    samples: &[u16],
    rows: u32,
    cols: u32,
    precision: u32,
    near: u32,
) -> Result<ValidatedEncoding, JpegLsEncodeError> {
    if rows == 0 || cols == 0 {
        return Err(JpegLsEncodeError::EmptyImage { rows, cols });
    }
    if rows > MAX_FRAME_DIMENSION || cols > MAX_FRAME_DIMENSION {
        return Err(JpegLsEncodeError::DimensionOutOfRange {
            rows,
            cols,
            maximum: MAX_FRAME_DIMENSION,
        });
    }

    let rows_usize =
        usize::try_from(rows).map_err(|_| JpegLsEncodeError::DimensionOverflow { rows, cols })?;
    let cols_usize =
        usize::try_from(cols).map_err(|_| JpegLsEncodeError::DimensionOverflow { rows, cols })?;
    let expected = rows_usize
        .checked_mul(cols_usize)
        .ok_or(JpegLsEncodeError::DimensionOverflow { rows, cols })?;
    if samples.len() != expected {
        return Err(JpegLsEncodeError::PixelCountMismatch {
            actual: samples.len(),
            expected,
        });
    }

    if !(MIN_SAMPLE_PRECISION..=MAX_SAMPLE_PRECISION).contains(&precision) {
        return Err(JpegLsEncodeError::UnsupportedPrecision {
            precision,
            minimum: MIN_SAMPLE_PRECISION,
            maximum: MAX_SAMPLE_PRECISION,
        });
    }
    let maximum_sample = ((1u32 << precision) - 1) as u16;
    let maximum_near = MAX_HEADER_NEAR.min(u32::from(maximum_sample) / 2);
    if near > maximum_near {
        return Err(JpegLsEncodeError::NearOutOfRange {
            near,
            maximum: maximum_near,
        });
    }
    if let Some((index, &value)) = samples
        .iter()
        .enumerate()
        .find(|(_, value)| **value > maximum_sample)
    {
        return Err(JpegLsEncodeError::SampleOutOfRange {
            index,
            value,
            maximum: maximum_sample,
        });
    }

    let precision = u8::try_from(precision)
        .expect("invariant: validated JPEG-LS precision fits in the SOS field");
    let near = i32::try_from(near).expect("invariant: validated JPEG-LS NEAR fits in an i32");
    Ok(ValidatedEncoding {
        rows: rows_usize,
        cols: cols_usize,
        precision,
        near,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_geometry() {
        assert_eq!(
            validate_encoding(&[], 0, 1, 8, 0).unwrap_err(),
            JpegLsEncodeError::EmptyImage { rows: 0, cols: 1 }
        );
    }

    #[test]
    fn rejects_dimension_not_representable_by_sof55() {
        assert_eq!(
            validate_encoding(&[], u32::from(u16::MAX) + 1, 1, 8, 0).unwrap_err(),
            JpegLsEncodeError::DimensionOutOfRange {
                rows: u32::from(u16::MAX) + 1,
                cols: 1,
                maximum: u32::from(u16::MAX),
            }
        );
    }

    #[test]
    fn rejects_sample_count_mismatch() {
        assert_eq!(
            validate_encoding(&[1, 2, 3], 2, 2, 8, 0).unwrap_err(),
            JpegLsEncodeError::PixelCountMismatch {
                actual: 3,
                expected: 4,
            }
        );
    }

    #[test]
    fn rejects_unsupported_precision_boundaries() {
        for precision in [7, 17] {
            assert_eq!(
                validate_encoding(&[0], 1, 1, precision, 0).unwrap_err(),
                JpegLsEncodeError::UnsupportedPrecision {
                    precision,
                    minimum: 8,
                    maximum: 16,
                }
            );
        }
    }

    #[test]
    fn enforces_precision_dependent_near_bound() {
        let validated =
            validate_encoding(&[0], 1, 1, 8, 127).expect("8-bit NEAR=127 is representable");
        assert_eq!(validated.near, 127);
        assert_eq!(
            validate_encoding(&[0], 1, 1, 8, 128).unwrap_err(),
            JpegLsEncodeError::NearOutOfRange {
                near: 128,
                maximum: 127,
            }
        );
    }

    #[test]
    fn rejects_first_sample_outside_declared_precision() {
        assert_eq!(
            validate_encoding(&[0, 256, 255], 1, 3, 8, 0).unwrap_err(),
            JpegLsEncodeError::SampleOutOfRange {
                index: 1,
                value: 256,
                maximum: 255,
            }
        );
    }
}
