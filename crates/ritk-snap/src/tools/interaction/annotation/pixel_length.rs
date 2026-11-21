//! Image-index distances independent of physical pixel spacing.

use super::MeasurementError;
use serde::{Deserialize, Serialize};

/// A finite Euclidean distance between two image-pixel endpoints.
///
/// Endpoints use `[row, column]` order in the existing `f32` image-coordinate
/// representation. Distance is `hypot(end_row - start_row, end_col - start_col)`
/// in pixels, computed in that precision. No physical spacing or aspect ratio
/// participates. Coincident endpoints are valid; a calibration reference must
/// independently require a positive distance.
///
/// Construction and deserialization reject nonfinite endpoints, overflowing
/// coordinate differences, and a norm outside the finite representation.
///
/// # Examples
///
/// ```
/// use ritk_snap::tools::interaction::{MeasurementError, PixelLength};
///
/// let reference = PixelLength::try_from([[0.0, 0.0], [3.0, 4.0]])?;
/// assert_eq!(reference.pixels(), 5.0);
/// assert_eq!(reference.start_px(), [0.0, 0.0]);
/// assert_eq!(reference.end_px(), [3.0, 4.0]);
/// # Ok::<(), MeasurementError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[[f32; 2]; 2]", into = "[[f32; 2]; 2]")]
#[repr(transparent)]
#[must_use]
pub struct PixelLength {
    endpoints: [[f32; 2]; 2],
}

impl PixelLength {
    /// Return the first endpoint in row-column order, without a spacing conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::tools::interaction::{MeasurementError, PixelLength};
    /// let length = PixelLength::try_from([[1.0, 2.0], [4.0, 6.0]])?;
    /// assert_eq!(length.start_px(), [1.0, 2.0]);
    /// # Ok::<(), MeasurementError>(())
    /// ```
    #[must_use]
    pub const fn start_px(self) -> [f32; 2] {
        self.endpoints[0]
    }

    /// Return the second endpoint in row-column order, without a spacing conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::tools::interaction::{MeasurementError, PixelLength};
    /// let length = PixelLength::try_from([[1.0, 2.0], [4.0, 6.0]])?;
    /// assert_eq!(length.end_px(), [4.0, 6.0]);
    /// # Ok::<(), MeasurementError>(())
    /// ```
    #[must_use]
    pub const fn end_px(self) -> [f32; 2] {
        self.endpoints[1]
    }

    /// Return image-index distance without assigning a physical unit.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::tools::interaction::{MeasurementError, PixelLength};
    /// let length = PixelLength::try_from([[1.0, 2.0], [4.0, 6.0]])?;
    /// assert_eq!(length.pixels(), 5.0);
    /// # Ok::<(), MeasurementError>(())
    /// ```
    #[must_use]
    pub fn pixels(self) -> f32 {
        let [[start_row, start_col], [end_row, end_col]] = self.endpoints;
        (end_row - start_row).hypot(end_col - start_col)
    }
}

impl TryFrom<[[f32; 2]; 2]> for PixelLength {
    type Error = MeasurementError;

    /// Validate both endpoints and the distance they represent.
    ///
    /// # Errors
    ///
    /// Returns [`MeasurementError::NonFiniteImageCoordinate`] for a nonfinite
    /// endpoint or [`MeasurementError::NonFiniteResult`] for an overflowing
    /// displacement or norm.
    fn try_from(endpoints: [[f32; 2]; 2]) -> Result<Self, Self::Error> {
        for (point, coordinates) in endpoints.iter().enumerate() {
            for (component, value) in coordinates.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(MeasurementError::NonFiniteImageCoordinate {
                        point,
                        component,
                        value,
                    });
                }
            }
        }
        let length = Self { endpoints };
        if !length.pixels().is_finite() {
            return Err(MeasurementError::NonFiniteResult {
                kind: "pixel length",
            });
        }
        Ok(length)
    }
}

impl From<PixelLength> for [[f32; 2]; 2] {
    fn from(length: PixelLength) -> Self {
        let PixelLength { endpoints } = length;
        endpoints
    }
}

#[cfg(test)]
mod tests;
