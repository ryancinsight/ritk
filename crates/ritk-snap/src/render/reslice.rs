//! Physical-plane resampling for scalar viewer volumes.
//!
//! A [`ReslicePlane`] describes the patient-space location of the first
//! output pixel and three patient-space steps: horizontal, vertical, and
//! through-plane.  The request is validated against the source affine before
//! any samples are taken.  This keeps oblique MPR and slab projection in the
//! viewer domain while leaving host surfaces format-neutral.

use thiserror::Error;

use crate::geometry::affine::{AffineError, AffineTransform};
use crate::LoadedVolume;

use super::slab::{ProjectionStatistic, SlabProjection};
use sampling::{
    add_scaled, axis_index, patient_step_to_voxel, sample_volume, validate_dimensions,
    validate_plane_vectors, validate_volume, validate_volume_bounds, vector_norm, voxel_step,
};

/// Interpolation used when a plane lands between source voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResliceInterpolation {
    /// Select the nearest source voxel.
    Nearest,
    /// Trilinearly interpolate the eight neighbouring source voxels.
    Linear,
}

/// A validated physical-space plane and optional through-plane slab.
///
/// `origin` is the patient-space centre of output pixel `[0, 0]`.  The two
/// in-plane steps advance one output column or row.  `depth_step` advances one
/// sample for a slab; it may be zero only when `depth_samples` is one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReslicePlane {
    origin: [f64; 3],
    horizontal_step: [f64; 3],
    vertical_step: [f64; 3],
    depth_step: [f64; 3],
    dimensions: [usize; 2],
    depth_samples: usize,
    interpolation: ResliceInterpolation,
    shape: [usize; 3],
    transform: AffineTransform,
}

impl ReslicePlane {
    /// Validate and build a plane from patient-space pixel steps.
    pub fn try_new(
        volume: &LoadedVolume,
        origin: [f64; 3],
        horizontal_step: [f64; 3],
        vertical_step: [f64; 3],
        depth_step: [f64; 3],
        dimensions: [usize; 2],
        depth_samples: usize,
        interpolation: ResliceInterpolation,
    ) -> Result<Self, ResliceError> {
        let transform = validate_volume(volume)?;
        validate_plane_vectors(origin, horizontal_step, vertical_step, depth_step)?;
        validate_dimensions(dimensions, depth_samples)?;
        if depth_samples > 1 && vector_norm(depth_step) == 0.0 {
            return Err(ResliceError::InvalidDepthStep);
        }
        let origin_voxel = transform.patient_to_voxel(origin);
        let horizontal_voxel = patient_step_to_voxel(&transform, origin, horizontal_step);
        let vertical_voxel = patient_step_to_voxel(&transform, origin, vertical_step);
        let depth_voxel = patient_step_to_voxel(&transform, origin, depth_step);
        validate_volume_bounds(
            origin_voxel,
            [horizontal_voxel, vertical_voxel, depth_voxel],
            dimensions,
            depth_samples,
            volume.shape,
        )?;
        Ok(Self {
            origin,
            horizontal_step,
            vertical_step,
            depth_step,
            dimensions,
            depth_samples,
            interpolation,
            shape: volume.shape,
            transform,
        })
    }

    /// Build a one-sample plane matching one existing axis-aligned slice.
    pub fn axis_aligned(
        volume: &LoadedVolume,
        axis: usize,
        index: usize,
        interpolation: ResliceInterpolation,
    ) -> Result<Self, ResliceError> {
        validate_volume(volume)?;
        if axis > 2 {
            return Err(ResliceError::InvalidAxis { axis });
        }
        let extent = volume.shape[axis];
        if index >= extent {
            return Err(ResliceError::IndexOutOfBounds {
                axis,
                index,
                extent,
            });
        }
        let slab = SlabProjection::try_new(volume, axis, index, 0)
            .map_err(ResliceError::AxisAlignedSlab)?;
        Self::from_slab(volume, slab, interpolation)
    }

    /// Build an axis-aligned plane for an existing validated slab request.
    pub fn from_slab(
        volume: &LoadedVolume,
        slab: SlabProjection,
        interpolation: ResliceInterpolation,
    ) -> Result<Self, ResliceError> {
        let transform = validate_volume(volume)?;
        let axis = slab.axis();
        if axis > 2 {
            return Err(ResliceError::InvalidAxis { axis });
        }
        if slab.shape() != volume.shape {
            return Err(ResliceError::ShapeChanged {
                expected: slab.shape(),
                actual: volume.shape,
            });
        }
        let base = axis_index(slab.start(), axis);
        let origin = transform.voxel_to_patient(base);
        let horizontal_axis = match axis {
            0 | 1 => 2,
            2 => 1,
            _ => return Err(ResliceError::InvalidAxis { axis }),
        };
        let vertical_axis = match axis {
            0 => 1,
            1 | 2 => 0,
            _ => return Err(ResliceError::InvalidAxis { axis }),
        };
        let horizontal_step = voxel_step(&transform, base, horizontal_axis);
        let vertical_step = voxel_step(&transform, base, vertical_axis);
        let depth_step = voxel_step(&transform, base, axis);
        Self::try_new(
            volume,
            origin,
            horizontal_step,
            vertical_step,
            depth_step,
            slab.dimensions(),
            slab.sample_count(),
            interpolation,
        )
    }

    /// Return the patient-space origin of output pixel `[0, 0]`.
    #[must_use]
    pub const fn origin(self) -> [f64; 3] {
        self.origin
    }

    /// Return the patient-space horizontal pixel step.
    #[must_use]
    pub const fn horizontal_step(self) -> [f64; 3] {
        self.horizontal_step
    }

    /// Return the patient-space vertical pixel step.
    #[must_use]
    pub const fn vertical_step(self) -> [f64; 3] {
        self.vertical_step
    }

    /// Return the patient-space through-plane sample step.
    #[must_use]
    pub const fn depth_step(self) -> [f64; 3] {
        self.depth_step
    }

    /// Return output dimensions in `[width, height]` order.
    #[must_use]
    pub const fn dimensions(self) -> [usize; 2] {
        self.dimensions
    }

    /// Return the number of source samples reduced for every output pixel.
    #[must_use]
    pub const fn depth_samples(self) -> usize {
        self.depth_samples
    }

    /// Return the interpolation policy.
    #[must_use]
    pub const fn interpolation(self) -> ResliceInterpolation {
        self.interpolation
    }

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

/// Failure while validating or evaluating a physical-plane request.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ResliceError {
    /// The source volume has an empty spatial extent or no channels.
    #[error("reslicing requires a non-empty volume with channels")]
    EmptyVolume,
    /// Reslicing currently accepts scalar source volumes only.
    #[error("reslicing requires one scalar channel, received {channels}")]
    UnsupportedChannels { channels: u8 },
    /// The source layout overflows a flat sample count.
    #[error("volume shape {shape:?} with {channels} channels overflows a sample count")]
    SampleCountOverflow { shape: [usize; 3], channels: u8 },
    /// The source payload does not match its declared layout.
    #[error("volume payload length {actual} does not match the declared sample count {expected}")]
    MalformedPayload { expected: usize, actual: usize },
    /// The source physical geometry is invalid.
    #[error("volume geometry is invalid: {source}")]
    InvalidGeometry {
        #[source]
        source: AffineError,
    },
    /// The requested axis is outside the three spatial axes.
    #[error("reslice axis {axis} is outside the spatial range 0..=2")]
    InvalidAxis { axis: usize },
    /// An axis-aligned slice index is outside the source extent.
    #[error("reslice index {index} is outside axis {axis} extent {extent}")]
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        extent: usize,
    },
    /// A plane vector is non-finite or has zero length.
    #[error("reslice plane vectors must be finite and non-zero")]
    InvalidPlaneBasis,
    /// The through-plane step is zero for a multi-sample slab.
    #[error("a multi-sample slab requires a non-zero depth step")]
    InvalidDepthStep,
    /// Output dimensions or depth are zero.
    #[error("reslice dimensions and depth sample count must be non-zero")]
    EmptyRequest,
    /// A request's source-sample work product is too large.
    #[error("reslice request has too many source samples")]
    WorkTooLarge,
    /// A plane corner lies outside the source volume.
    #[error("reslice plane lies outside the source volume at voxel coordinate {coordinate:?}")]
    OutOfVolume { coordinate: [f64; 3] },
    /// The source shape changed after request validation.
    #[error("reslice volume shape changed from {expected:?} to {actual:?}")]
    ShapeChanged {
        expected: [usize; 3],
        actual: [usize; 3],
    },
    /// The source physical geometry changed after request validation.
    #[error("reslice volume geometry changed after request validation")]
    GeometryChanged,
    /// The output dimensions overflow a flat sample count.
    #[error("reslice output dimensions {dimensions:?} overflow a sample count")]
    OutputTooLarge { dimensions: [usize; 2] },
    /// The wrapped axis-aligned slab request was invalid.
    #[error("axis-aligned slab request is invalid: {0}")]
    AxisAlignedSlab(#[source] super::slab::SlabProjectionError),
}

mod sampling;

#[cfg(test)]
#[path = "tests_reslice.rs"]
mod tests;
