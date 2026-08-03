//! MINC2 stored-integer to real-intensity scaling.
//!
//! Integer image samples map from the image dataset's `valid_range` to the
//! scalar or per-slice `image-min` / `image-max` range. Floating-point image
//! datasets bypass this module, as required by the MINC conversion contract.

use anyhow::{bail, Context, Result};
use consus_core::Datatype;

#[derive(Debug, Clone, Copy)]
struct RealRange {
    minimum: f64,
    maximum: f64,
}

impl RealRange {
    fn new(minimum: f64, maximum: f64, index: usize) -> Result<Self> {
        if !minimum.is_finite() || !maximum.is_finite() {
            bail!("MINC2 image range {index} must be finite, got [{minimum}, {maximum}]");
        }
        if minimum > maximum {
            bail!(
                "MINC2 image range {index} has image-min {minimum} greater than image-max {maximum}"
            );
        }
        if !(minimum as f32).is_finite() || !(maximum as f32).is_finite() {
            bail!("MINC2 image range {index} [{minimum}, {maximum}] exceeds the f32 output range");
        }
        Ok(Self { minimum, maximum })
    }
}

#[derive(Debug)]
enum ImageRanges {
    Global(RealRange),
    PerSlice(Box<[RealRange]>),
}

/// Validated scaling metadata for one integer image dataset.
#[derive(Debug)]
pub(crate) struct IntegerScaling {
    valid_minimum: f64,
    valid_maximum: f64,
    image_ranges: ImageRanges,
    slice_length: usize,
    total_elements: usize,
}

impl IntegerScaling {
    /// Validate and construct the scaling contract.
    pub(crate) fn new(
        valid_range: [f64; 2],
        storage_range: [f64; 2],
        image_minima: Vec<f64>,
        image_maxima: Vec<f64>,
        slice_length: usize,
        total_elements: usize,
    ) -> Result<Self> {
        let [first, second] = valid_range;
        if !first.is_finite() || !second.is_finite() {
            bail!("MINC2 valid_range must contain finite endpoints, got [{first}, {second}]");
        }
        let (valid_minimum, valid_maximum) = if first <= second {
            (first, second)
        } else {
            (second, first)
        };
        if valid_minimum == valid_maximum {
            bail!("MINC2 valid_range endpoints must differ, got {valid_minimum}");
        }
        let storage_minimum = storage_range[0].min(storage_range[1]);
        let storage_maximum = storage_range[0].max(storage_range[1]);
        if valid_minimum < storage_minimum || valid_maximum > storage_maximum {
            bail!(
                "MINC2 valid_range [{valid_minimum}, {valid_maximum}] exceeds the stored datatype range [{storage_minimum}, {storage_maximum}]"
            );
        }
        if slice_length == 0 || total_elements == 0 || !total_elements.is_multiple_of(slice_length)
        {
            bail!(
                "MINC2 scaling geometry is inconsistent: {total_elements} voxels, {slice_length} voxels per slice"
            );
        }
        if image_minima.len() != image_maxima.len() {
            bail!(
                "MINC2 image-min/image-max length mismatch: {} versus {}",
                image_minima.len(),
                image_maxima.len()
            );
        }

        let slice_count = total_elements / slice_length;
        let ranges: Vec<RealRange> = image_minima
            .into_iter()
            .zip(image_maxima)
            .enumerate()
            .map(|(index, (minimum, maximum))| RealRange::new(minimum, maximum, index))
            .collect::<Result<_>>()?;
        let image_ranges = match ranges.as_slice() {
            [range] => ImageRanges::Global(*range),
            _ if ranges.len() == slice_count => ImageRanges::PerSlice(ranges.into_boxed_slice()),
            _ => bail!(
                "MINC2 image ranges must be scalar or have one entry per slice ({slice_count}), got {}",
                ranges.len()
            ),
        };

        Ok(Self {
            valid_minimum,
            valid_maximum,
            image_ranges,
            slice_length,
            total_elements,
        })
    }

    /// Map one stored integer value to its real `f32` intensity.
    pub(crate) fn scale(&self, stored: f64, linear_index: usize) -> Result<f32> {
        if linear_index >= self.total_elements {
            bail!(
                "MINC2 voxel index {linear_index} exceeds declared element count {}",
                self.total_elements
            );
        }
        if stored < self.valid_minimum || stored > self.valid_maximum {
            bail!(
                "MINC2 stored voxel {linear_index} value {stored} is outside valid_range [{}, {}]",
                self.valid_minimum,
                self.valid_maximum
            );
        }

        let range = match &self.image_ranges {
            ImageRanges::Global(range) => *range,
            ImageRanges::PerSlice(ranges) => {
                let slice = linear_index / self.slice_length;
                *ranges
                    .get(slice)
                    .context("MINC2 scaling slice index exceeds image ranges")?
            }
        };
        if stored == self.valid_minimum || range.minimum == range.maximum {
            return Ok(range.minimum as f32);
        }
        if stored == self.valid_maximum {
            return Ok(range.maximum as f32);
        }

        let scale = (range.maximum - range.minimum) / (self.valid_maximum - self.valid_minimum);
        let real = (stored - self.valid_minimum).mul_add(scale, range.minimum);
        let output = real as f32;
        if !output.is_finite() {
            bail!(
                "MINC2 scaled voxel {linear_index} value {real} exceeds the finite f32 output range"
            );
        }
        Ok(output)
    }
}

/// Default MINC valid range for an integer-like HDF5 image datatype.
pub(crate) fn default_integer_valid_range(datatype: &Datatype) -> Result<Option<[f64; 2]>> {
    match datatype {
        Datatype::Integer { bits, signed, .. } => {
            let width = bits.get();
            let range = match (width, signed) {
                (8, true) => [f64::from(i8::MIN), f64::from(i8::MAX)],
                (8, false) => [0.0, f64::from(u8::MAX)],
                (16, true) => [f64::from(i16::MIN), f64::from(i16::MAX)],
                (16, false) => [0.0, f64::from(u16::MAX)],
                (32, true) => [f64::from(i32::MIN), f64::from(i32::MAX)],
                (32, false) => [0.0, f64::from(u32::MAX)],
                (64, true) => [i64::MIN as f64, i64::MAX as f64],
                (64, false) => [0.0, u64::MAX as f64],
                _ => bail!("Unsupported MINC2 integer width for scaling: {width}"),
            };
            Ok(Some(range))
        }
        Datatype::Boolean => Ok(Some([0.0, 1.0])),
        Datatype::Float { .. } => Ok(None),
        other => bail!("Unsupported MINC2 voxel datatype for scaling: {other:?}"),
    }
}

#[cfg(test)]
#[path = "tests_scaling.rs"]
mod tests;
