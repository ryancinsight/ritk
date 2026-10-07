//! Validated stored samples with their three-dimensional image metadata.

use ritk_codecs::SampleBuffer;
use ritk_image::ImageMetadata;
use ritk_spatial::{CoordinateMap, InvalidCoordinateMap};
use thiserror::Error;

use crate::calibration::{CalibrationShapeError, IntensityCalibration, IntensityUnit};

/// A volume whose samples retain their fixed-width on-disk representation.
///
/// The shape is RITK's depth, row, column order. The constructor checks the
/// sample count, finite and invertible physical geometry, representable
/// spacing-scaled axes and column lengths, three-dimensional coordinate map, per-slice transform
/// count, and per-frame calibration before the value can cross a format boundary.
///
/// Physical metadata uses patient LPS coordinates and millimeters. Format
/// adapters convert their source basis and units before constructing a volume;
/// this type validates numeric geometry but cannot infer the basis supplied by
/// its caller.
#[derive(Debug)]
pub struct StoredVolume {
    shape: [usize; 3],
    samples: SampleBuffer,
    metadata: ImageMetadata<3>,
    coordinate_map: CoordinateMap,
    calibration: IntensityCalibration,
    intensity_unit: Option<IntensityUnit>,
}

impl StoredVolume {
    /// Creates a stored volume after validating its structural invariants.
    ///
    /// # Errors
    ///
    /// Returns VolumeError if an axis is empty, the voxel count overflows,
    /// the sample count disagrees with the shape, the physical metadata is
    /// invalid or cannot represent finite nonzero scaled axes, the coordinate
    /// map is not three-dimensional, per-slice transforms do not match depth,
    /// a transform is non-finite, or per-frame calibration does not match depth.
    pub fn new(
        shape: [usize; 3],
        samples: SampleBuffer,
        metadata: ImageMetadata<3>,
        coordinate_map: CoordinateMap,
        calibration: IntensityCalibration,
    ) -> Result<Self, VolumeError> {
        for (axis, length) in shape.into_iter().enumerate() {
            if length == 0 {
                return Err(VolumeError::EmptyAxis { axis });
            }
        }
        let sample_count = shape[0]
            .checked_mul(shape[1])
            .and_then(|plane| plane.checked_mul(shape[2]))
            .ok_or(VolumeError::ShapeProductOverflow { shape })?;
        if samples.len() != sample_count {
            return Err(VolumeError::SampleCountMismatch {
                expected: sample_count,
                actual: samples.len(),
            });
        }
        validate_physical_geometry(&metadata)?;
        validate_coordinate_map(&coordinate_map, shape)?;
        calibration.validate_for_shape(shape)?;
        Ok(Self {
            shape,
            samples,
            metadata,
            coordinate_map,
            calibration,
            intensity_unit: None,
        })
    }

    /// Attaches an uninterpreted label for the calibrated intensity values.
    ///
    /// The label remains unchanged. Format adapters report a typed loss when
    /// a target cannot preserve it.
    #[must_use]
    pub fn with_intensity_unit(mut self, unit: IntensityUnit) -> Self {
        self.intensity_unit = Some(unit);
        self
    }

    /// Returns the volume shape in depth, row, column order.
    #[must_use]
    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Returns the stored sample buffer without converting its values.
    #[must_use]
    pub const fn samples(&self) -> &SampleBuffer {
        &self.samples
    }

    /// Returns LPS-millimeter origin, spacing, and axis direction.
    #[must_use]
    pub const fn metadata(&self) -> &ImageMetadata<3> {
        &self.metadata
    }

    /// Returns the non-affine index-to-physical coordinate mapping.
    #[must_use]
    pub const fn coordinate_map(&self) -> &CoordinateMap {
        &self.coordinate_map
    }

    /// Returns the intensity transform associated with the stored samples.
    #[must_use]
    pub const fn calibration(&self) -> &IntensityCalibration {
        &self.calibration
    }

    /// Returns the source-provided label for calibrated intensity values.
    #[must_use]
    pub fn intensity_unit(&self) -> Option<&IntensityUnit> {
        self.intensity_unit.as_ref()
    }
}

/// Checks that a coordinate map is valid for a stored volume's shape.
///
/// # Errors
///
/// Returns a typed error when the map does not support three-dimensional
/// images or a per-slice transform count differs from the depth. Slice-series
/// transforms are finite by construction through
/// [`ritk_spatial::SliceSeries::try_new`].
pub fn validate_coordinate_map(
    coordinate_map: &CoordinateMap,
    shape: [usize; 3],
) -> Result<(), VolumeError> {
    coordinate_map.validate_dimensionality(3)?;
    if let CoordinateMap::SliceSeries(series) = coordinate_map
        && series.len() != shape[0]
    {
        return Err(VolumeError::CoordinateMapSliceCountMismatch {
            expected: shape[0],
            actual: series.len(),
        });
    }
    Ok(())
}

/// Checks that image geometry remains representable as physical axis vectors.
///
/// Format adapters call this before creating output so unrepresentable
/// metadata cannot truncate an existing destination. `StoredVolume::new`
/// applies the same contract when admitting input samples.
///
/// # Errors
///
/// Returns a typed error when the origin, spacing, direction, scaled physical
/// components, normalized direction components, or physical axis lengths are
/// invalid or cannot be represented as finite `f64` values.
pub fn validate_physical_geometry(metadata: &ImageMetadata<3>) -> Result<(), VolumeError> {
    if let Some(axis) = metadata
        .origin()
        .as_slice()
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(VolumeError::NonFiniteOrigin { axis });
    }
    for (axis, value) in metadata.spacing().to_array().into_iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(VolumeError::InvalidSpacing { axis });
        }
    }
    if let Some(index) = metadata
        .direction()
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(VolumeError::NonFiniteDirection {
            row: index / 3,
            column: index % 3,
        });
    }
    let determinant = metadata.direction().determinant();
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(VolumeError::SingularDirection);
    }
    let spacing = metadata.spacing().to_array();
    for (index, (coefficient, axis_spacing)) in metadata
        .direction()
        .iter()
        .copied()
        .zip(spacing.into_iter().cycle())
        .enumerate()
    {
        let scaled_component = coefficient * axis_spacing;
        if !scaled_component.is_finite() || (coefficient != 0.0 && scaled_component == 0.0) {
            return Err(VolumeError::UnrepresentablePhysicalAxis {
                row: index / 3,
                column: index % 3,
            });
        }
    }
    let mut normalized_direction = [[0.0; 3]; 3];
    for column in 0..3 {
        let spacing = spacing[column];
        let components = [
            metadata.direction()[(0, column)] * spacing,
            metadata.direction()[(1, column)] * spacing,
            metadata.direction()[(2, column)] * spacing,
        ];
        let length = components[0].hypot(components[1]).hypot(components[2]);
        if !length.is_finite() || length == 0.0 {
            return Err(VolumeError::UnrepresentablePhysicalAxisNorm { column });
        }
        for (row, component) in components.into_iter().enumerate() {
            let normalized = component / length;
            if component != 0.0 && normalized == 0.0 {
                return Err(VolumeError::UnrepresentablePhysicalAxisDirection { row, column });
            }
            normalized_direction[row][column] = normalized;
        }
    }
    let normalized_direction = ritk_spatial::Direction::from_rows(normalized_direction);
    let normalized_determinant = normalized_direction.determinant();
    if !normalized_determinant.is_finite() || normalized_determinant == 0.0 {
        return Err(VolumeError::UnrepresentableNormalizedDirection);
    }
    Ok(())
}

/// A stored volume violates its sample, shape, geometry, or calibration
/// contract.
#[derive(Debug, Error)]
pub enum VolumeError {
    /// An axis contains no samples.
    #[error("stored volume axis {axis} has zero samples")]
    EmptyAxis {
        /// The zero-length axis in depth, row, column order.
        axis: usize,
    },
    /// Multiplying the volume dimensions overflows usize.
    #[error("stored volume shape {shape:?} overflows usize")]
    ShapeProductOverflow {
        /// The rejected depth, row, column shape.
        shape: [usize; 3],
    },
    /// The number of stored samples does not match the volume shape.
    #[error("stored volume requires {expected} samples but contains {actual}")]
    SampleCountMismatch {
        /// Number of samples required by the shape.
        expected: usize,
        /// Number of samples in the buffer.
        actual: usize,
    },
    /// The physical origin contains a non-finite coordinate.
    #[error("stored volume origin coordinate {axis} is not finite")]
    NonFiniteOrigin {
        /// Coordinate index in physical XYZ order.
        axis: usize,
    },
    /// A physical spacing is not finite and strictly positive.
    #[error("stored volume spacing on axis {axis} is not finite and positive")]
    InvalidSpacing {
        /// Axis index in depth, row, column order.
        axis: usize,
    },
    /// The direction matrix contains a non-finite coefficient.
    #[error("stored volume direction coefficient ({row}, {column}) is not finite")]
    NonFiniteDirection {
        /// Matrix row.
        row: usize,
        /// Matrix column.
        column: usize,
    },
    /// The direction matrix is singular or its determinant is non-finite.
    #[error("stored volume direction matrix must be finite and invertible")]
    SingularDirection,
    /// Normalizing physical axes produces a singular direction matrix.
    #[error("stored volume normalized physical directions are singular")]
    UnrepresentableNormalizedDirection,
    /// A direction coefficient times its spacing overflows or underflows to zero.
    #[error("stored volume physical axis component ({row}, {column}) is not representable")]
    UnrepresentablePhysicalAxis {
        /// Matrix row.
        row: usize,
        /// Matrix column.
        column: usize,
    },
    /// A physical axis length overflows or underflows when its components are combined.
    #[error("stored volume physical axis {column} has an unrepresentable length")]
    UnrepresentablePhysicalAxisNorm {
        /// Matrix column in depth, row, column order.
        column: usize,
    },
    /// A normalized direction component underflows to zero.
    #[error(
        "stored volume normalized physical axis component ({row}, {column}) is not representable"
    )]
    UnrepresentablePhysicalAxisDirection {
        /// Matrix row.
        row: usize,
        /// Matrix column.
        column: usize,
    },
    /// The coordinate map cannot describe a three-dimensional volume.
    #[error(transparent)]
    CoordinateMapDimensionality(#[from] InvalidCoordinateMap),
    /// A per-slice coordinate map does not contain one transform per depth slice.
    #[error("slice-series coordinate map has {actual} transforms; volume depth is {expected}")]
    CoordinateMapSliceCountMismatch {
        /// Required transform count.
        expected: usize,
        /// Supplied transform count.
        actual: usize,
    },
    /// Per-frame calibration does not match the depth axis.
    #[error(transparent)]
    CalibrationShape(#[from] CalibrationShapeError),
}

#[cfg(test)]
#[path = "tests/volume.rs"]
mod tests;
