//! Typed browser selection of one scalar projection statistic.

use crate::render::ProjectionStatistic;

/// Failure while validating a browser projection statistic index.
#[derive(Debug, thiserror::Error, PartialEq)]
pub(crate) enum BrowserProjectionError {
    /// A JavaScript number is not finite.
    #[error("browser projection index {value} must be finite")]
    CoordinateNotFinite {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript number does not denote an integer.
    #[error("browser projection index {value} must be an integer")]
    CoordinateNotInteger {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript integer is outside the projection table.
    #[error("browser projection index {value} is outside 0..2")]
    IndexOutOfRange {
        /// Rejected integer value.
        value: usize,
    },
    /// A JavaScript integer cannot be represented by the WASM ABI.
    #[error("browser projection index {value} is outside 0..={maximum}")]
    CoordinateOutOfRange {
        /// Rejected JavaScript number.
        value: f64,
        /// Inclusive upper bound of the WASM ABI integer representation.
        maximum: u32,
    },
}

/// Validates one JavaScript projection index and maps it to the RITK statistic.
pub(crate) fn parse_browser_projection_request(
    value: f64,
) -> Result<ProjectionStatistic, BrowserProjectionError> {
    if !value.is_finite() {
        return Err(BrowserProjectionError::CoordinateNotFinite { value });
    }
    if value.fract() != 0.0 {
        return Err(BrowserProjectionError::CoordinateNotInteger { value });
    }
    if value < 0.0 || value > f64::from(u32::MAX) {
        return Err(BrowserProjectionError::CoordinateOutOfRange {
            value,
            maximum: u32::MAX,
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite integral input is bounded to the complete u32 range above"
    )]
    let value = value as u32;
    let value =
        usize::try_from(value).map_err(|_| BrowserProjectionError::CoordinateOutOfRange {
            value: f64::from(value),
            maximum: u32::MAX,
        })?;
    match value {
        0 => Ok(ProjectionStatistic::Maximum),
        1 => Ok(ProjectionStatistic::Minimum),
        2 => Ok(ProjectionStatistic::Average),
        value => Err(BrowserProjectionError::IndexOutOfRange { value }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_maps_all_scalar_statistics() {
        assert_eq!(
            parse_browser_projection_request(0.0),
            Ok(ProjectionStatistic::Maximum)
        );
        assert_eq!(
            parse_browser_projection_request(1.0),
            Ok(ProjectionStatistic::Minimum)
        );
        assert_eq!(
            parse_browser_projection_request(2.0),
            Ok(ProjectionStatistic::Average)
        );
    }

    #[test]
    fn parser_rejects_non_integral_and_out_of_range_values() {
        assert_eq!(
            parse_browser_projection_request(1.5),
            Err(BrowserProjectionError::CoordinateNotInteger { value: 1.5 })
        );
        assert!(matches!(
            parse_browser_projection_request(f64::NAN),
            Err(BrowserProjectionError::CoordinateNotFinite { value }) if value.is_nan()
        ));
        assert_eq!(
            parse_browser_projection_request(-1.0),
            Err(BrowserProjectionError::CoordinateOutOfRange {
                value: -1.0,
                maximum: u32::MAX,
            })
        );
        assert_eq!(
            parse_browser_projection_request(3.0),
            Err(BrowserProjectionError::IndexOutOfRange { value: 3 })
        );
    }
}
