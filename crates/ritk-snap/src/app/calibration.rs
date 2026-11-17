//! Manual length calibration owned by the viewer model.

use crate::tools::interaction::AnnotationLocation;

/// A validated multiplicative correction for lengths on one displayed image.
///
/// The caller derives `measured_mm` from the original stored endpoints for
/// every recalibration. The displayed, previously corrected length is never a
/// calibration input, so repeated calibration remains well-defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ManualLengthCalibration {
    location: AnnotationLocation,
    scale: f32,
}

impl ManualLengthCalibration {
    /// Calibrate a measured reference length to a known physical distance.
    #[cfg(any(windows, test))]
    pub(crate) fn from_reference(
        location: AnnotationLocation,
        measured_mm: f32,
        known_mm: f32,
    ) -> Option<Self> {
        if !measured_mm.is_finite()
            || measured_mm <= 0.0
            || !known_mm.is_finite()
            || known_mm <= 0.0
        {
            return None;
        }
        let scale = known_mm / measured_mm;
        (scale.is_finite() && scale > 0.0).then_some(Self { location, scale })
    }

    pub(crate) fn applies_to(self, location: AnnotationLocation) -> bool {
        self.location == location
    }

    pub(crate) fn apply(self, measured_mm: f32) -> Option<f32> {
        let calibrated = measured_mm * self.scale;
        calibrated.is_finite().then_some(calibrated)
    }

    #[cfg(test)]
    fn scale(self) -> f32 {
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::ManualLengthCalibration;
    use crate::tools::interaction::AnnotationLocation;

    #[test]
    fn known_reference_scales_and_retains_location() {
        let location = AnnotationLocation { plane: 1, slice: 4 };
        let calibration = ManualLengthCalibration::from_reference(location, 20.0, 7.0)
            .expect("invariant: positive finite reference values produce calibration");
        assert_eq!(calibration.scale(), 0.35);
        assert!(calibration.applies_to(location));
        assert_eq!(calibration.apply(10.0), Some(3.5));
        assert!(!calibration.applies_to(AnnotationLocation { plane: 1, slice: 5 }));
        let recalibrated = ManualLengthCalibration::from_reference(location, 20.0, 14.0)
            .expect("invariant: original endpoint length remains a valid reference");
        assert_eq!(recalibrated.apply(10.0), Some(7.0));
    }

    #[test]
    fn rejects_non_positive_or_non_finite_reference() {
        let location = AnnotationLocation { plane: 0, slice: 0 };
        assert!(ManualLengthCalibration::from_reference(location, 0.0, 5.0).is_none());
        assert!(ManualLengthCalibration::from_reference(location, 5.0, f32::NAN).is_none());
    }
}
