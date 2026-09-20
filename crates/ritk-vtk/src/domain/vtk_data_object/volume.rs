use std::sync::Arc;

use thiserror::Error;

use super::{AttributeArray, VtkImageData};

/// Failure while constructing a validated spatial VTK volume.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VtkImageVolumeError {
    /// One of the three dimensions is zero.
    #[error("volume dimension {axis} is zero")]
    EmptyDimension { axis: usize },
    /// A dimension cannot be represented by VTK's signed extent indices.
    #[error("volume dimension {axis} does not fit in a VTK extent: {value}")]
    DimensionOverflow { axis: usize, value: usize },
    /// A volume must carry at least one scalar channel.
    #[error("volume channel count must be positive")]
    ZeroChannels,
    /// The dimensions and channel count overflow the addressable sample count.
    #[error("volume sample count overflows the addressable range")]
    SampleCountOverflow,
    /// The payload length does not match the dimensions and channel count.
    #[error("volume payload has {actual} samples; expected {expected}")]
    PayloadLength { actual: usize, expected: usize },
    /// An origin coordinate is not finite.
    #[error("volume origin coordinate {axis} is not finite")]
    NonFiniteOrigin { axis: usize },
    /// A spacing coordinate is not finite or positive.
    #[error("volume spacing coordinate {axis} is not finite or positive")]
    InvalidSpacing { axis: usize },
    /// A direction entry is not finite.
    #[error("volume direction entry {index} is not finite")]
    NonFiniteDirection { index: usize },
    /// The direction matrix cannot define an invertible physical grid.
    #[error("volume direction matrix is singular")]
    SingularDirection,
    /// The legacy VTK representation rejected the materialized arrays.
    #[error("materialized VTK image data is invalid: {reason}")]
    InvalidImageData { reason: String },
}

/// A direction-aware VTK-compatible image volume with shared scalar storage.
///
/// The dimensions, origin and spacing use VTK's `[x, y, z]` convention. The
/// direction matrix is row-major and its columns describe those same physical
/// axes. Scalar samples are point-centered, x-fastest, and channel-interleaved.
/// The payload is shared with the source through [`Arc`]; callers that need the
/// legacy owned [`VtkImageData`] representation must opt into
/// [`Self::to_vtk_image_data`].
#[derive(Debug, Clone, PartialEq)]
pub struct VtkImageVolume {
    dimensions: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
    channels: usize,
    scalars: Arc<Vec<f32>>,
}

impl VtkImageVolume {
    /// Construct a direction-aware VTK volume without copying its payload.
    ///
    /// `dimensions` and `spacing` are ordered `[x, y, z]`. `scalars` must
    /// contain `x × y × z × channels` samples in x-fastest, channel-interleaved
    /// order.
    ///
    /// # Errors
    /// Returns a typed error when geometry or payload invariants are invalid.
    pub fn from_parts(
        dimensions: [usize; 3],
        origin: [f64; 3],
        spacing: [f64; 3],
        direction: [f64; 9],
        channels: usize,
        scalars: Arc<Vec<f32>>,
    ) -> Result<Self, VtkImageVolumeError> {
        for (axis, dimension) in dimensions.into_iter().enumerate() {
            if dimension == 0 {
                return Err(VtkImageVolumeError::EmptyDimension { axis });
            }
            if i64::try_from(dimension - 1).is_err() {
                return Err(VtkImageVolumeError::DimensionOverflow {
                    axis,
                    value: dimension,
                });
            }
        }
        if channels == 0 {
            return Err(VtkImageVolumeError::ZeroChannels);
        }
        let expected = dimensions
            .into_iter()
            .try_fold(channels, usize::checked_mul)
            .ok_or(VtkImageVolumeError::SampleCountOverflow)?;
        if scalars.len() != expected {
            return Err(VtkImageVolumeError::PayloadLength {
                actual: scalars.len(),
                expected,
            });
        }
        for (axis, value) in origin.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(VtkImageVolumeError::NonFiniteOrigin { axis });
            }
        }
        for (axis, value) in spacing.into_iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(VtkImageVolumeError::InvalidSpacing { axis });
            }
        }
        for (index, value) in direction.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(VtkImageVolumeError::NonFiniteDirection { index });
            }
        }
        if !invertible(direction) {
            return Err(VtkImageVolumeError::SingularDirection);
        }
        Ok(Self {
            dimensions,
            origin,
            spacing,
            direction,
            channels,
            scalars,
        })
    }

    /// Return the VTK-order point dimensions `[x, y, z]`.
    #[must_use]
    pub const fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }

    /// Return the inclusive VTK point extent.
    #[must_use]
    pub fn whole_extent(&self) -> [i64; 6] {
        [
            0,
            (self.dimensions[0] - 1) as i64,
            0,
            (self.dimensions[1] - 1) as i64,
            0,
            (self.dimensions[2] - 1) as i64,
        ]
    }

    /// Return the physical origin of the first point.
    #[must_use]
    pub const fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Return positive voxel spacing in VTK `[x, y, z]` order.
    #[must_use]
    pub const fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    /// Return the row-major physical direction matrix.
    #[must_use]
    pub const fn direction(&self) -> [f64; 9] {
        self.direction
    }

    /// Return the number of interleaved channels per point.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Borrow the shared point-centered scalar payload.
    #[must_use]
    pub fn scalars(&self) -> &[f32] {
        self.scalars.as_slice()
    }

    /// Materialize the legacy owned VTK image representation.
    ///
    /// This is the explicit copy boundary for serializers and filters whose
    /// `AttributeArray` contract still owns a `Vec<f32>`. The spatial direction
    /// remains available from [`Self::direction`] because legacy VTK ImageData
    /// has no direction field.
    pub fn to_vtk_image_data(&self) -> Result<VtkImageData, VtkImageVolumeError> {
        let mut image = VtkImageData {
            whole_extent: self.whole_extent(),
            origin: self.origin,
            spacing: self.spacing,
            ..VtkImageData::default()
        };
        image.point_data.insert(
            "scalars".to_owned(),
            AttributeArray::Scalars {
                values: self.scalars.as_ref().clone(),
                num_components: self.channels,
            },
        );
        image
            .validate()
            .map_err(|reason| VtkImageVolumeError::InvalidImageData { reason })?;
        Ok(image)
    }
}

fn invertible(matrix: [f64; 9]) -> bool {
    let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    let scale = matrix.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    let threshold = 128.0 * f64::EPSILON * scale * scale * scale;
    determinant.is_finite() && scale.is_finite() && scale > 0.0 && determinant.abs() > threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROTATED_DIRECTION: [f64; 9] = [
        0.0, -1.0, 0.0, // row 0
        1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, // row 2
    ];

    fn volume() -> VtkImageVolume {
        VtkImageVolume::from_parts(
            [3, 2, 2],
            [10.0, 20.0, 30.0],
            [0.5, 2.0, 3.0],
            ROTATED_DIRECTION,
            2,
            Arc::new((0..24).map(|value| value as f32).collect()),
        )
        .expect("valid spatial volume")
    }

    #[test]
    fn preserves_geometry_and_channel_order() {
        let image = volume();
        assert_eq!(image.whole_extent(), [0, 2, 0, 1, 0, 1]);
        assert_eq!(image.origin(), [10.0, 20.0, 30.0]);
        assert_eq!(image.spacing(), [0.5, 2.0, 3.0]);
        assert_eq!(image.direction(), ROTATED_DIRECTION);
        assert_eq!(image.channels(), 2);
        assert_eq!(image.scalars()[..4], [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn materialization_is_valid_and_keeps_values() {
        let image = volume();
        let materialized = image.to_vtk_image_data().expect("valid VTK data");
        assert_eq!(materialized.whole_extent, [0, 2, 0, 1, 0, 1]);
        assert_eq!(materialized.spacing, [0.5, 2.0, 3.0]);
        assert!(materialized.validate().is_ok());
        let AttributeArray::Scalars {
            values,
            num_components,
        } = materialized
            .point_data
            .get("scalars")
            .expect("scalar array")
        else {
            panic!("materialized array is not scalar data");
        };
        assert_eq!(*num_components, 2);
        assert_eq!(
            values,
            &(0..24).map(|value| value as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_invalid_geometry_and_payload() {
        let scalars = Arc::new(vec![0.0; 8]);
        assert_eq!(
            VtkImageVolume::from_parts(
                [0, 2, 2],
                [0.0; 3],
                [1.0; 3],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                1,
                Arc::clone(&scalars),
            ),
            Err(VtkImageVolumeError::EmptyDimension { axis: 0 })
        );
        assert_eq!(
            VtkImageVolume::from_parts(
                [2, 2, 2],
                [0.0; 3],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                1,
                Arc::clone(&scalars),
            ),
            Err(VtkImageVolumeError::InvalidSpacing { axis: 1 })
        );
        assert_eq!(
            VtkImageVolume::from_parts(
                [2, 2, 2],
                [0.0; 3],
                [1.0; 3],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                1,
                Arc::clone(&scalars),
            ),
            Err(VtkImageVolumeError::SingularDirection)
        );
        assert_eq!(
            VtkImageVolume::from_parts(
                [2, 2, 2],
                [0.0; 3],
                [1.0; 3],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                2,
                Arc::new(vec![0.0; 8]),
            ),
            Err(VtkImageVolumeError::PayloadLength {
                actual: 8,
                expected: 16,
            })
        );
        assert_eq!(
            VtkImageVolume::from_parts(
                [i64::MAX as usize, 3, 1],
                [0.0; 3],
                [1.0; 3],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                1,
                Arc::new(Vec::new()),
            ),
            Err(VtkImageVolumeError::SampleCountOverflow)
        );
        assert_eq!(
            VtkImageVolume::from_parts(
                [2, 2, 2],
                [f64::NAN, 0.0, 0.0],
                [1.0; 3],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                1,
                Arc::clone(&scalars),
            ),
            Err(VtkImageVolumeError::NonFiniteOrigin { axis: 0 })
        );
        assert_eq!(
            VtkImageVolume::from_parts(
                [2, 2, 2],
                [0.0; 3],
                [1.0; 3],
                [f64::INFINITY, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                1,
                Arc::clone(&scalars),
            ),
            Err(VtkImageVolumeError::NonFiniteDirection { index: 0 })
        );
    }

    #[test]
    fn scalar_borrow_shares_allocation() {
        let source = Arc::new((0..24).map(|value| value as f32).collect::<Vec<_>>());
        let source_pointer = source.as_ptr();
        let image = VtkImageVolume::from_parts(
            [3, 2, 2],
            [0.0; 3],
            [1.0; 3],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            2,
            source,
        )
        .expect("valid spatial volume");
        assert_eq!(image.scalars().as_ptr(), source_pointer);
    }
}
