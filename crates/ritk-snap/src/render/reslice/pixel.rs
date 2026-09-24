//! Continuous pixel-to-patient mapping for reslice planes.

use crate::geometry::{PatientPointError, PatientPointMm};
use thiserror::Error;

use super::sampling::add_scaled;
use super::ReslicePlane;

mod interval;
mod projection;
pub use projection::PatientPlaneProjection;

/// An error mapping a continuous output pixel to patient millimetres.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum PixelMappingError {
    /// The pixel coordinate contains a non-finite component.
    #[error("pixel coordinate {coordinate:?} must be finite")]
    InvalidCoordinate {
        /// The original column-row coordinate.
        coordinate: [f64; 2],
    },
    /// The pixel coordinate is outside the output dimensions.
    #[error("pixel coordinate {coordinate:?} is outside output dimensions {dimensions:?}")]
    OutOfBounds {
        /// The original column-row coordinate.
        coordinate: [f64; 2],
        /// The plane's output width and height.
        dimensions: [usize; 2],
    },
    /// The mapped point is not finite patient millimetres.
    #[error("mapped patient point is invalid: {source}")]
    InvalidPatientPoint {
        /// The error produced by patient-point validation.
        #[source]
        source: PatientPointError,
    },
    /// Finite patient coordinates overflow during plane projection.
    #[error("patient point {coordinate:?} overflows during plane projection")]
    ProjectionOverflow {
        /// The finite patient-space point that could not be projected.
        coordinate: [f64; 3],
    },
    /// The pixel enclosure extends at least half a pixel from its nominal value.
    #[error("patient point {coordinate:?} has a pixel projection unresolved within half a pixel")]
    ProjectionUnresolved {
        /// The finite patient-space point whose projection could not be resolved.
        coordinate: [f64; 3],
    },
}

impl ReslicePlane {
    /// Map a continuous output-pixel coordinate to patient millimetres.
    ///
    /// Coordinates are in column-row order, and integer values identify pixel
    /// centres. The validated plane's origin and physical pixel steps define
    /// the affine mapping.
    ///
    /// # Errors
    ///
    /// Returns [`PixelMappingError::InvalidCoordinate`] for non-finite input,
    /// [`PixelMappingError::OutOfBounds`] outside the output dimensions, or
    /// [`PixelMappingError::InvalidPatientPoint`] when the mapped coordinates
    /// are not finite patient millimetres.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use ritk_snap::geometry::PatientPointMm;
    /// # use ritk_snap::render::{PixelMappingError, ReslicePlane};
    /// # fn map(plane: ReslicePlane) -> Result<PatientPointMm, PixelMappingError> {
    /// plane.patient_at_pixel([12.5, 8.0])
    /// # }
    /// ```
    pub fn patient_at_pixel(self, pixel: [f64; 2]) -> Result<PatientPointMm, PixelMappingError> {
        self.validate_pixel_coordinate(pixel)?;
        let coordinates = add_scaled(
            add_scaled(self.origin, self.horizontal_step, pixel[0]),
            self.vertical_step,
            pixel[1],
        );
        PatientPointMm::try_from(coordinates)
            .map_err(|source| PixelMappingError::InvalidPatientPoint { source })
    }

    fn validate_pixel_coordinate(self, pixel: [f64; 2]) -> Result<(), PixelMappingError> {
        if !pixel.into_iter().all(f64::is_finite) {
            return Err(PixelMappingError::InvalidCoordinate { coordinate: pixel });
        }
        let maximum = self.maximum_pixel_coordinate();
        if pixel[0] < 0.0 || pixel[0] > maximum[0] || pixel[1] < 0.0 || pixel[1] > maximum[1] {
            return Err(PixelMappingError::OutOfBounds {
                coordinate: pixel,
                dimensions: self.dimensions,
            });
        }
        Ok(())
    }

    fn maximum_pixel_coordinate(self) -> [f64; 2] {
        self.dimensions.map(|dimension| {
            let maximum = dimension
                .checked_sub(1)
                .expect("invariant: validated plane dimensions are nonzero");
            let maximum =
                u32::try_from(maximum).expect("invariant: validated output dimensions fit in u32");
            f64::from(maximum)
        })
    }
}
