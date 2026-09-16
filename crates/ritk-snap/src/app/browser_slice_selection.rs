//! Exact browser slice selection over the shared viewer state.

use super::SnapApp;
use crate::ui::axis_total;

/// Failure to select a browser slice without changing viewer state.
#[derive(Debug, thiserror::Error, PartialEq)]
pub(crate) enum BrowserSliceSelectionError {
    /// No browser viewer is mounted.
    #[cfg(target_arch = "wasm32")]
    #[error("RITK browser viewer is not mounted")]
    ViewerNotMounted,
    /// Another browser callback currently owns the viewer state.
    #[cfg(target_arch = "wasm32")]
    #[error("RITK browser viewer is handling another callback")]
    ViewerBusy,
    /// The mounted viewer has not loaded a study.
    #[error("RITK browser viewer has no loaded study")]
    StudyNotLoaded,
    /// The requested axis is outside the orthogonal axis set.
    #[error("browser slice axis {axis} is outside 0..=2")]
    AxisOutOfRange {
        /// Rejected zero-based axis.
        axis: usize,
    },
    /// The requested index is outside its axis extent.
    #[error("browser slice index {index} is outside 0..{slice_count} for axis {axis}")]
    IndexOutOfRange {
        /// Validated zero-based axis.
        axis: usize,
        /// Rejected zero-based slice index.
        index: usize,
        /// Number of available slices on the axis.
        slice_count: usize,
    },
    /// A JavaScript number is not finite.
    #[error("browser slice {coordinate} {value} must be finite")]
    CoordinateNotFinite {
        /// Axis or index argument rejected at the WASM boundary.
        coordinate: BrowserSliceCoordinate,
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript number does not denote an integer.
    #[error("browser slice {coordinate} {value} must be an integer")]
    CoordinateNotInteger {
        /// Axis or index argument rejected at the WASM boundary.
        coordinate: BrowserSliceCoordinate,
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript integer cannot be represented by a WASM `usize`.
    #[error("browser slice {coordinate} {value} is outside 0..={maximum}")]
    CoordinateOutOfRange {
        /// Axis or index argument rejected at the WASM boundary.
        coordinate: BrowserSliceCoordinate,
        /// Rejected JavaScript number.
        value: f64,
        /// Inclusive upper bound of the WASM ABI integer representation.
        maximum: u32,
    },
}

/// Numeric coordinate supplied by the JavaScript slice-selection boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BrowserSliceCoordinate {
    /// Orthogonal axis number.
    Axis,
    /// Zero-based slice index.
    Index,
}

impl std::fmt::Display for BrowserSliceCoordinate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Axis => "axis",
            Self::Index => "index",
        })
    }
}

/// Validates JavaScript numbers before the WASM ABI can narrow them.
pub(crate) fn parse_browser_slice_request(
    axis: f64,
    index: f64,
) -> Result<(usize, usize), BrowserSliceSelectionError> {
    Ok((
        parse_coordinate(BrowserSliceCoordinate::Axis, axis)?,
        parse_coordinate(BrowserSliceCoordinate::Index, index)?,
    ))
}

fn parse_coordinate(
    coordinate: BrowserSliceCoordinate,
    value: f64,
) -> Result<usize, BrowserSliceSelectionError> {
    if !value.is_finite() {
        return Err(BrowserSliceSelectionError::CoordinateNotFinite { coordinate, value });
    }
    if value.fract() != 0.0 {
        return Err(BrowserSliceSelectionError::CoordinateNotInteger { coordinate, value });
    }
    if value < 0.0 || value > f64::from(u32::MAX) {
        return Err(BrowserSliceSelectionError::CoordinateOutOfRange {
            coordinate,
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
    usize::try_from(value).map_err(|_| BrowserSliceSelectionError::CoordinateOutOfRange {
        coordinate,
        value: f64::from(value),
        maximum: u32::MAX,
    })
}

impl SnapApp {
    /// Selects one exact zero-based slice without clamping invalid input.
    ///
    /// Returns whether the selected index changed. A change uses the shared
    /// slice reducer so visual revision and linked-cursor state stay coherent.
    pub(crate) fn select_browser_slice(
        &mut self,
        axis: usize,
        index: usize,
    ) -> Result<bool, BrowserSliceSelectionError> {
        if axis > 2 {
            return Err(BrowserSliceSelectionError::AxisOutOfRange { axis });
        }
        let shape = self
            .loaded
            .as_ref()
            .ok_or(BrowserSliceSelectionError::StudyNotLoaded)?
            .shape;
        let slice_count = axis_total(shape, axis);
        if index >= slice_count {
            return Err(BrowserSliceSelectionError::IndexOutOfRange {
                axis,
                index,
                slice_count,
            });
        }
        if self.axis_slice_info(axis).0 == index {
            return Ok(false);
        }
        self.set_slice_for_axis(axis, index);
        Ok(true)
    }
}
