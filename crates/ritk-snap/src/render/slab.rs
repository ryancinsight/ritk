//! Host-neutral scalar slab projection over a [`LoadedVolume`].
//!
//! The request is axis-aligned and voxel-bounded. It deliberately does not
//! resample an oblique physical plane: the caller receives an exact reduction
//! over the source samples and can apply DICOM window/level and a colormap at
//! its presentation boundary.

use thiserror::Error;

use crate::LoadedVolume;

/// Reduction applied to each output pixel of a slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionStatistic {
    /// Select the greatest source sample in the slab.
    Maximum,
    /// Select the least source sample in the slab.
    Minimum,
    /// Compute the arithmetic mean of the source samples in the slab.
    Average,
}

/// A validated axis-aligned slab request.
///
/// The request is tied to the source shape observed at construction. This
/// prevents a caller from reusing a centre/range against a differently shaped
/// study without an explicit new validation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabProjection {
    axis: usize,
    center: usize,
    half_width: usize,
    start: usize,
    end: usize,
    shape: [usize; 3],
}

impl SlabProjection {
    /// Validate an axis, centre and inclusive voxel half-width for `volume`.
    ///
    /// The selected source range is `center - half_width ..= center +
    /// half_width`; requests that extend beyond the volume are rejected rather
    /// than silently clamped.
    pub fn try_new(
        volume: &LoadedVolume,
        axis: usize,
        center: usize,
        half_width: usize,
    ) -> Result<Self, SlabProjectionError> {
        validate_volume(volume)?;
        let Some(&extent) = volume.shape.get(axis) else {
            return Err(SlabProjectionError::InvalidAxis { axis });
        };
        if extent == 0 {
            return Err(SlabProjectionError::EmptyExtent { axis });
        }
        if center >= extent {
            return Err(SlabProjectionError::CenterOutOfBounds {
                axis,
                center,
                extent,
            });
        }
        let Some(start) = center.checked_sub(half_width) else {
            return Err(SlabProjectionError::RangeOutOfBounds {
                axis,
                center,
                half_width,
                extent,
            });
        };
        let Some(end) = center.checked_add(half_width) else {
            return Err(SlabProjectionError::RangeOutOfBounds {
                axis,
                center,
                half_width,
                extent,
            });
        };
        if end >= extent {
            return Err(SlabProjectionError::RangeOutOfBounds {
                axis,
                center,
                half_width,
                extent,
            });
        }
        Ok(Self {
            axis,
            center,
            half_width,
            start,
            end,
            shape: volume.shape,
        })
    }

    /// Return the source axis reduced by this request (`0` depth, `1` row,
    /// `2` column).
    #[must_use]
    pub const fn axis(self) -> usize {
        self.axis
    }

    /// Return the centre index of the selected source slab.
    #[must_use]
    pub const fn center(self) -> usize {
        self.center
    }

    /// Return the inclusive voxel half-width around [`Self::center`].
    #[must_use]
    pub const fn half_width(self) -> usize {
        self.half_width
    }

    /// Return the first selected source index.
    #[must_use]
    pub const fn start(self) -> usize {
        self.start
    }

    /// Return the last selected source index.
    #[must_use]
    pub const fn end(self) -> usize {
        self.end
    }

    /// Return the number of source samples reduced for each output pixel.
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.end - self.start + 1
    }

    /// Return the output dimensions in `[width, height]` order.
    #[must_use]
    pub const fn dimensions(self) -> [usize; 2] {
        match self.axis {
            0 => [self.shape[2], self.shape[1]],
            1 => [self.shape[2], self.shape[0]],
            2 => [self.shape[1], self.shape[0]],
            _ => [0, 0],
        }
    }

    /// Compute an owned projection plane with the selected statistic.
    pub fn compute(
        self,
        volume: &LoadedVolume,
        statistic: ProjectionStatistic,
    ) -> Result<ProjectionPlane, SlabProjectionError> {
        let mut pixels = Vec::new();
        let dimensions = self.compute_into(volume, statistic, &mut pixels)?;
        Ok(ProjectionPlane {
            axis: self.axis,
            dimensions,
            sample_count: self.sample_count(),
            pixels: pixels.into_boxed_slice(),
        })
    }

    /// Compute into caller-owned storage, preserving its capacity between
    /// projections. The returned dimensions are `[width, height]`.
    pub fn compute_into(
        self,
        volume: &LoadedVolume,
        statistic: ProjectionStatistic,
        pixels: &mut Vec<f32>,
    ) -> Result<[usize; 2], SlabProjectionError> {
        validate_volume(volume)?;
        if volume.shape != self.shape {
            return Err(SlabProjectionError::ShapeChanged {
                expected: self.shape,
                actual: volume.shape,
            });
        }
        let dimensions = self.dimensions();
        let output_len = dimensions[0]
            .checked_mul(dimensions[1])
            .ok_or(SlabProjectionError::OutputTooLarge { dimensions })?;
        pixels.resize(output_len, 0.0);
        for (position, (row, column)) in self.output_coordinates().enumerate() {
            pixels[position] = self.reduce_pixel(volume, statistic, row, column);
        }
        Ok(dimensions)
    }

    fn output_coordinates(self) -> impl Iterator<Item = (usize, usize)> {
        let [width, height] = self.dimensions();
        (0..height).flat_map(move |row| (0..width).map(move |column| (row, column)))
    }

    fn reduce_pixel(
        self,
        volume: &LoadedVolume,
        statistic: ProjectionStatistic,
        row: usize,
        column: usize,
    ) -> f32 {
        let mut value = match statistic {
            ProjectionStatistic::Maximum => f32::NEG_INFINITY,
            ProjectionStatistic::Minimum => f32::INFINITY,
            ProjectionStatistic::Average => 0.0,
        };
        for source_index in self.start..=self.end {
            let sample = match self.axis {
                0 => volume.pixel_at(source_index, row, column),
                1 => volume.pixel_at(row, source_index, column),
                2 => volume.pixel_at(row, column, source_index),
                _ => unreachable!("validated slab axis"),
            };
            value = match statistic {
                ProjectionStatistic::Maximum => value.max(sample),
                ProjectionStatistic::Minimum => value.min(sample),
                ProjectionStatistic::Average => value + sample,
            };
        }
        if statistic == ProjectionStatistic::Average {
            value / self.sample_count() as f32
        } else {
            value
        }
    }
}

/// Scalar output plane produced by [`SlabProjection::compute`].
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionPlane {
    axis: usize,
    dimensions: [usize; 2],
    sample_count: usize,
    pixels: Box<[f32]>,
}

impl ProjectionPlane {
    /// Return the source axis reduced by this plane.
    #[must_use]
    pub const fn axis(&self) -> usize {
        self.axis
    }

    /// Return the plane dimensions in `[width, height]` order.
    #[must_use]
    pub const fn dimensions(&self) -> [usize; 2] {
        self.dimensions
    }

    /// Return the source sample count contributing to each output pixel.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Borrow the row-major scalar output samples.
    #[must_use]
    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }
}

/// Failure while validating or computing a slab projection.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SlabProjectionError {
    /// The source volume has no scalar or RGB channels.
    #[error("slab projection requires a non-empty volume with channels")]
    EmptyVolume,
    /// The source contains channels other than one scalar channel.
    #[error("slab projection requires one scalar channel, received {channels}")]
    UnsupportedChannels {
        /// Number of interleaved channels in the source volume.
        channels: u8,
    },
    /// The requested axis is outside the three spatial axes.
    #[error("slab projection axis {axis} is outside the spatial range 0..=2")]
    InvalidAxis {
        /// Requested axis.
        axis: usize,
    },
    /// The selected spatial extent is empty.
    #[error("slab projection axis {axis} has an empty extent")]
    EmptyExtent {
        /// Axis whose extent is empty.
        axis: usize,
    },
    /// The slab centre lies outside its axis extent.
    #[error("slab projection centre {center} is outside axis {axis} extent {extent}")]
    CenterOutOfBounds {
        /// Requested centre index.
        center: usize,
        /// Requested axis.
        axis: usize,
        /// Available extent along the axis.
        extent: usize,
    },
    /// The inclusive slab range extends outside the source extent.
    #[error(
        "slab projection range around centre {center} with half-width {half_width} exceeds axis {axis} extent {extent}"
    )]
    RangeOutOfBounds {
        /// Requested centre index.
        center: usize,
        /// Requested inclusive half-width.
        half_width: usize,
        /// Requested axis.
        axis: usize,
        /// Available extent along the axis.
        extent: usize,
    },
    /// The declared shape cannot be represented as a flat sample count.
    #[error("volume shape {shape:?} with {channels} channels overflows a flat sample count")]
    SampleCountOverflow {
        /// Spatial volume shape.
        shape: [usize; 3],
        /// Interleaved channel count.
        channels: u8,
    },
    /// The volume shape changed after request validation.
    #[error("slab projection shape changed from {expected:?} to {actual:?}")]
    ShapeChanged {
        /// Shape captured when the request was created.
        expected: [usize; 3],
        /// Shape observed during computation.
        actual: [usize; 3],
    },
    /// The source payload cannot contain the declared samples.
    #[error("volume payload length {actual} does not match the declared sample count {expected}")]
    MalformedPayload {
        /// Declared sample count.
        expected: usize,
        /// Actual payload length.
        actual: usize,
    },
    /// The output dimensions overflow a `usize` product.
    #[error("slab projection output dimensions {dimensions:?} overflow a sample count")]
    OutputTooLarge {
        /// Output dimensions.
        dimensions: [usize; 2],
    },
}

fn validate_volume(volume: &LoadedVolume) -> Result<(), SlabProjectionError> {
    if volume.channels == 0 || volume.shape.contains(&0) {
        return Err(SlabProjectionError::EmptyVolume);
    }
    if volume.channels != 1 {
        return Err(SlabProjectionError::UnsupportedChannels {
            channels: volume.channels,
        });
    }
    let expected = volume
        .shape
        .iter()
        .try_fold(usize::from(volume.channels), |count, extent| {
            count.checked_mul(*extent)
        })
        .ok_or(SlabProjectionError::SampleCountOverflow {
            shape: volume.shape,
            channels: volume.channels,
        })?;
    if volume.data.len() != expected {
        return Err(SlabProjectionError::MalformedPayload {
            expected,
            actual: volume.data.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests_slab.rs"]
mod tests;
