//! MINC2 stored-integer to real-intensity scaling as per-slice real-value maps.
//!
//! Integer image samples map from the image dataset's `valid_range`
//! \[`valid_min`, `valid_max`\] to the scalar or per-slice `image-min` /
//! `image-max` real range \[`image_min`, `image_max`\] by the MINC pixel
//! conversion:
//!
//! ```text
//! real = (stored - valid_min) / (valid_max - valid_min) * (image_max - image_min) + image_min
//! ```
//!
//! This is the map `real = (stored - valid_min) * slope + intercept` with
//! `slope = (image_max - image_min) / (valid_max - valid_min)` and
//! `intercept = image_min`, so each slice is one [`RealValueMap`]. A scalar
//! real range gives every slice the same map. A slice with
//! `image_min == image_max` is uniform: slope zero, intercept `image_min`.
//! Floating-point image datasets bypass this module, as the MINC conversion
//! contract requires.
//!
//! Definitions:
//! <https://www.bic.mni.mcgill.ca/software/minc/prog_guide/node19.html> and
//! <https://www.bic.mni.mcgill.ca/software/minc/minc1_format/node5.html>.

use crate::real_map::RealValueMap;
use anyhow::{bail, Context, Result};
use eunomia::NumericElement;
use ritk_codecs::sample::{Sample, SampleBuffer, SampleType};

/// Validated scaling metadata for one integer image dataset: its stored
/// `valid_range` and the real-value map of each slice.
///
/// The maps are held as the file states them, one for a scalar real range and
/// one per slice otherwise, so the allocation follows the range datasets the
/// file holds and never the slice count its header claims.
#[derive(Debug)]
pub(crate) struct IntegerScaling {
    valid_minimum: f64,
    valid_maximum: f64,
    maps: Box<[RealValueMap]>,
    slice_count: usize,
}

impl IntegerScaling {
    /// Validate and construct the scaling contract.
    ///
    /// `image_minima` and `image_maxima` hold one entry per slice, or one entry
    /// for every slice. `slice_length` voxels make one slice and
    /// `total_elements` voxels make the volume.
    pub(crate) fn new(
        valid_range: [f64; 2],
        storage_range: [f64; 2],
        image_minima: &[f64],
        image_maxima: &[f64],
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
        let valid_width = valid_maximum - valid_minimum;
        let maps: Vec<RealValueMap> = image_minima
            .iter()
            .zip(image_maxima)
            .enumerate()
            .map(|(index, (&minimum, &maximum))| {
                slice_map(index, [minimum, maximum], valid_minimum, valid_width)
            })
            .collect::<Result<_>>()?;
        if maps.len() != 1 && maps.len() != slice_count {
            bail!(
                "MINC2 image ranges must be scalar or have one entry per slice ({slice_count}), got {}",
                maps.len()
            );
        }

        Ok(Self {
            valid_minimum,
            valid_maximum,
            maps: maps.into_boxed_slice(),
            slice_count,
        })
    }

    /// The real-value map of each slice along the first spatial axis.
    ///
    /// Materialises one map per slice, so call it only once the voxel payload
    /// has proved the slice count is backed by the file.
    pub(crate) fn slice_maps(&self) -> Vec<RealValueMap> {
        match self.maps.as_ref() {
            [map] => vec![*map; self.slice_count],
            maps => maps.to_vec(),
        }
    }

    /// Reject the first stored sample outside `valid_range`.
    ///
    /// Such samples denote missing data; the image contract has no mask for
    /// them, so a read fails instead of mapping them.
    pub(crate) fn check_stored(&self, samples: &SampleBuffer) -> Result<()> {
        match samples {
            SampleBuffer::U8(values) => self.check(values),
            SampleBuffer::I8(values) => self.check(values),
            SampleBuffer::U16(values) => self.check(values),
            SampleBuffer::I16(values) => self.check(values),
            SampleBuffer::U32(values) => self.check(values),
            SampleBuffer::I32(values) => self.check(values),
            SampleBuffer::U64(values) => self.check(values),
            SampleBuffer::I64(values) => self.check(values),
            SampleBuffer::F32(values) => self.check(values),
            SampleBuffer::F64(values) => self.check(values),
        }
    }

    fn check<S: Sample>(&self, values: &[S]) -> Result<()> {
        for (index, &value) in values.iter().enumerate() {
            let stored = NumericElement::to_f64(value);
            if stored < self.valid_minimum || stored > self.valid_maximum {
                bail!(
                    "MINC2 stored voxel {index} value {stored} is outside valid_range [{}, {}]",
                    self.valid_minimum,
                    self.valid_maximum
                );
            }
        }
        Ok(())
    }
}

/// The real-value map of slice `index` from its real range.
fn slice_map(
    index: usize,
    [minimum, maximum]: [f64; 2],
    valid_minimum: f64,
    valid_width: f64,
) -> Result<RealValueMap> {
    if !minimum.is_finite() || !maximum.is_finite() {
        bail!("MINC2 image range {index} must be finite, got [{minimum}, {maximum}]");
    }
    if minimum > maximum {
        bail!("MINC2 image range {index} has image-min {minimum} greater than image-max {maximum}");
    }
    let slope = if minimum == maximum {
        0.0
    } else {
        (maximum - minimum) / valid_width
    };
    RealValueMap::new(valid_minimum, slope, minimum)
        .with_context(|| format!("MINC2 image range {index} [{minimum}, {maximum}]"))
}

/// The range of values an integer sample type stores, `None` for a float type.
pub(crate) fn integer_storage_range(sample_type: SampleType) -> Option<[f64; 2]> {
    Some(match sample_type {
        SampleType::U8 => [0.0, f64::from(u8::MAX)],
        SampleType::I8 => [f64::from(i8::MIN), f64::from(i8::MAX)],
        SampleType::U16 => [0.0, f64::from(u16::MAX)],
        SampleType::I16 => [f64::from(i16::MIN), f64::from(i16::MAX)],
        SampleType::U32 => [0.0, f64::from(u32::MAX)],
        SampleType::I32 => [f64::from(i32::MIN), f64::from(i32::MAX)],
        SampleType::U64 => [0.0, f64::from_unsigned_sample(u64::MAX)],
        SampleType::I64 => [
            f64::from_signed_sample(i64::MIN),
            f64::from_signed_sample(i64::MAX),
        ],
        SampleType::F32 | SampleType::F64 => return None,
    })
}

#[cfg(test)]
#[path = "tests_scaling.rs"]
mod tests;
