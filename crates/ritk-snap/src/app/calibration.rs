mod units;

pub(crate) use units::{CalibrationInputError, CalibrationUnit};

use super::SnapApp;
use crate::tools::interaction::{Annotation, AnnotationLocation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LengthCalibrationReference {
    location: AnnotationLocation,
    revision: u64,
    raw_mm: Option<u32>,
}

impl SnapApp {
    pub(crate) fn length_calibration_reference(&self) -> LengthCalibrationReference {
        let location = self.annotation_location();
        let raw_mm = self
            .annotations
            .iter()
            .rev()
            .find_map(|annotation| match annotation {
                Annotation::Length {
                    location: Some(source),
                    length_mm,
                    ..
                } if *source == location && length_mm.is_finite() && *length_mm > 0.0 => {
                    Some(length_mm.to_bits())
                }
                _ => None,
            });
        LengthCalibrationReference {
            location,
            revision: self.visual_revision,
            raw_mm,
        }
    }

    pub(crate) fn can_calibrate_length(&self) -> bool {
        self.length_calibration_reference().raw_mm.is_some()
    }

    pub(crate) fn has_length_calibration(&self) -> bool {
        let location = self.annotation_location();
        self.length_calibrations
            .iter()
            .any(|(source, _)| *source == location)
    }

    pub(crate) fn set_length_calibration(
        &mut self,
        reference: LengthCalibrationReference,
        known_mm: Option<f32>,
    ) -> bool {
        if reference != self.length_calibration_reference() {
            return false;
        }
        let scale = match (known_mm, reference.raw_mm.map(f32::from_bits)) {
            // Both operands are positive finite f32 values, so the scale
            // magnitude is below 2^277. A finite stored area has magnitude
            // below 2^128; applying scale squared stays below 2^682, while
            // the smallest nonzero result stays above 2^-703. Both bounds fit
            // the f64 display range (maximum exponent 1024, minimum -1074).
            (Some(known), Some(raw)) if known.is_finite() && known > 0.0 => {
                Some(f64::from(known) / f64::from(raw))
            }
            (None, _) => None,
            _ => return false,
        };
        self.length_calibrations
            .retain(|(location, _)| *location != reference.location);
        if let Some(scale) = scale {
            self.length_calibrations.push((reference.location, scale));
        }
        true
    }

    pub(crate) fn length_scale(&self, location: Option<AnnotationLocation>) -> f64 {
        self.length_calibrations
            .iter()
            .find(|(source, _)| Some(*source) == location)
            .map_or(1.0, |(_, scale)| *scale)
    }
}
