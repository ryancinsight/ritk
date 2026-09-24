//! Pixel operations for validated reslice planes.
//!
//! This module owns both continuous pixel-to-patient mapping and scalar
//! sampling into caller-owned output storage. Plane construction and source
//! validation remain in the parent module; this boundary keeps all output
//! pixel production in one operation family.

use crate::geometry::{PatientPointError, PatientPointMm};
use crate::LoadedVolume;
use thiserror::Error;

use super::sampling::{add_scaled, patient_step_to_voxel, sample_volume, validate_volume};
use super::{ResliceError, ReslicePlane};
use crate::render::slab::ProjectionStatistic;

mod interval;
mod projection;
pub use projection::PatientPlaneProjection;

/// Scalar output produced by [`ReslicePlane::compute`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResliceOutput {
    dimensions: [usize; 2],
    depth_samples: usize,
    statistic: ProjectionStatistic,
    pixels: Box<[f32]>,
}

impl ResliceOutput {
    /// Return output dimensions in `[width, height]` order.
    #[must_use]
    pub const fn dimensions(&self) -> [usize; 2] {
        self.dimensions
    }

    /// Return the number of source samples reduced for every output pixel.
    #[must_use]
    pub const fn depth_samples(&self) -> usize {
        self.depth_samples
    }

    /// Return the reduction applied to each output pixel.
    #[must_use]
    pub const fn statistic(&self) -> ProjectionStatistic {
        self.statistic
    }

    /// Borrow row-major scalar pixels.
    #[must_use]
    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }
}

impl ReslicePlane {
    /// Compute a scalar plane into a newly allocated output.
    pub fn compute(
        self,
        volume: &LoadedVolume,
        statistic: ProjectionStatistic,
    ) -> Result<ResliceOutput, ResliceError> {
        let mut pixels = Vec::new();
        self.compute_into(volume, statistic, &mut pixels)?;
        Ok(ResliceOutput {
            dimensions: self.dimensions,
            depth_samples: self.depth_samples,
            statistic,
            pixels: pixels.into_boxed_slice(),
        })
    }

    /// Compute a scalar plane into caller-owned storage.
    pub fn compute_into(
        self,
        volume: &LoadedVolume,
        statistic: ProjectionStatistic,
        pixels: &mut Vec<f32>,
    ) -> Result<[usize; 2], ResliceError> {
        let transform = validate_volume(volume)?;
        if volume.shape != self.shape {
            return Err(ResliceError::ShapeChanged {
                expected: self.shape,
                actual: volume.shape,
            });
        }
        if transform != self.transform {
            return Err(ResliceError::GeometryChanged);
        }
        let [width, height] = self.dimensions;
        let output_len = width
            .checked_mul(height)
            .ok_or(ResliceError::OutputTooLarge {
                dimensions: self.dimensions,
            })?;
        pixels.resize(output_len, 0.0);
        let origin_voxel = transform.patient_to_voxel(self.origin);
        let horizontal_voxel = patient_step_to_voxel(&transform, self.origin, self.horizontal_step);
        let vertical_voxel = patient_step_to_voxel(&transform, self.origin, self.vertical_step);
        let depth_voxel = patient_step_to_voxel(&transform, self.origin, self.depth_step);
        let mut output_position = 0;
        for row in 0..height {
            let row_origin = add_scaled(origin_voxel, vertical_voxel, row as f64);
            for column in 0..width {
                let pixel_origin = add_scaled(row_origin, horizontal_voxel, column as f64);
                let mut value = match statistic {
                    ProjectionStatistic::Maximum => f32::NEG_INFINITY,
                    ProjectionStatistic::Minimum => f32::INFINITY,
                    ProjectionStatistic::Average => 0.0,
                };
                for sample in 0..self.depth_samples {
                    let coordinate = add_scaled(pixel_origin, depth_voxel, sample as f64);
                    let source = sample_volume(volume, coordinate, self.interpolation)?;
                    value = match statistic {
                        ProjectionStatistic::Maximum => value.max(source),
                        ProjectionStatistic::Minimum => value.min(source),
                        ProjectionStatistic::Average => value + source,
                    };
                }
                if statistic == ProjectionStatistic::Average {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "validated slab depth is finite and the viewer preserves the existing f32 presentation contract"
                    )]
                    let count = self.depth_samples as f32;
                    value /= count;
                }
                pixels[output_position] = value;
                output_position += 1;
            }
        }
        Ok(self.dimensions)
    }
}

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
