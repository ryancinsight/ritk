//! Manual length calibration owned by the viewer model.

use crate::tools::interaction::AnnotationLocation;

/// Failure raised while applying or restoring image length calibration.
#[cfg(any(windows, test))]
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub(crate) enum CalibrationError {
    /// No selected length annotation is available for calibration.
    #[error("no selected length annotation")]
    #[cfg(windows)]
    MissingReference,
    /// The selected length is anchored to another image.
    #[error("selected length is not anchored to the active image")]
    #[cfg(windows)]
    WrongImage,
    /// The known or measured reference distance cannot define a factor.
    #[error("reference distance must be finite and strictly positive")]
    InvalidReference,
    /// A corrected annotation value cannot be represented by viewer storage.
    #[error("corrected length is outside the finite positive range")]
    #[cfg(windows)]
    InvalidReplacement,
    /// The active image has no usable physical spacing.
    #[error("cannot calculate image length: {0}")]
    #[cfg(windows)]
    Measurement(#[from] crate::tools::interaction::MeasurementError),
}

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
    ) -> Result<Self, CalibrationError> {
        if !measured_mm.is_finite()
            || measured_mm <= 0.0
            || !known_mm.is_finite()
            || known_mm <= 0.0
        {
            return Err(CalibrationError::InvalidReference);
        }
        let scale = known_mm / measured_mm;
        (scale.is_finite() && scale > 0.0)
            .then_some(Self { location, scale })
            .ok_or(CalibrationError::InvalidReference)
    }

    pub(crate) fn applies_to(self, location: AnnotationLocation) -> bool {
        self.location == location
    }

    pub(crate) fn apply(self, measured_mm: f32) -> Option<f32> {
        let calibrated = measured_mm * self.scale;
        (calibrated.is_finite() && calibrated > 0.0).then_some(calibrated)
    }

    #[cfg(test)]
    fn scale(self) -> f32 {
        self.scale
    }
}

#[cfg(windows)]
impl super::state::SnapApp {
    /// Report whether the active image carries a manual length correction.
    pub(crate) fn has_active_length_calibration(&self) -> bool {
        let location = self.annotation_location();
        self.length_calibrations
            .iter()
            .any(|calibration| calibration.applies_to(location))
    }

    /// Apply a known distance to the selected length and all anchored lengths.
    pub(crate) fn calibrate_selected_length(
        &mut self,
        known_mm: f32,
    ) -> Result<(), CalibrationError> {
        let selected = self
            .pointer
            .selected_annotation
            .ok_or(CalibrationError::MissingReference)?;
        let location = self.annotation_location();
        let (p1, p2) = match self.annotations.get(selected) {
            Some(crate::tools::interaction::Annotation::Length {
                p1,
                p2,
                location: Some(source),
                ..
            }) if *source == location => (*p1, *p2),
            Some(crate::tools::interaction::Annotation::Length { .. }) => {
                return Err(CalibrationError::WrongImage);
            }
            _ => return Err(CalibrationError::MissingReference),
        };
        let spacing = self.slice_plane_spacing()?;
        let measured_mm =
            crate::tools::interaction::Annotation::compute_length_checked(p1, p2, spacing)?;
        let calibration = ManualLengthCalibration::from_reference(location, measured_mm, known_mm)?;
        let raw_updates = self.raw_length_updates(location, spacing)?;
        let updates = raw_updates
            .into_iter()
            .map(|(index, measured)| {
                if index == selected {
                    Ok((index, known_mm))
                } else {
                    calibration
                        .apply(measured)
                        .map(|replacement| (index, replacement))
                        .ok_or(CalibrationError::InvalidReplacement)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.replace_length_values(updates, location, Some(calibration));
        Ok(())
    }

    /// Restore every anchored length on the active image to raw DICOM spacing.
    ///
    /// The operation is transactional: all raw values are computed before any
    /// annotation or calibration state is changed. Lengths on other images and
    /// their calibrations remain untouched.
    pub(crate) fn reset_active_length_calibration(&mut self) -> Result<bool, CalibrationError> {
        let location = self.annotation_location();
        if !self.has_active_length_calibration() {
            return Ok(false);
        }
        let spacing = self.slice_plane_spacing()?;
        let updates = self.raw_length_updates(location, spacing)?;
        self.replace_length_values(updates, location, None);
        Ok(true)
    }

    fn raw_length_updates(
        &self,
        location: AnnotationLocation,
        spacing: [f64; 2],
    ) -> Result<Vec<(usize, f32)>, CalibrationError> {
        self.annotations
            .iter()
            .enumerate()
            .filter_map(|(index, annotation)| match annotation {
                crate::tools::interaction::Annotation::Length {
                    p1,
                    p2,
                    location: Some(source),
                    ..
                } if *source == location => Some((
                    index,
                    crate::tools::interaction::Annotation::compute_length_checked(
                        *p1, *p2, spacing,
                    ),
                )),
                _ => None,
            })
            .map(|(index, result)| result.map(|length| (index, length)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(CalibrationError::from)
    }

    fn replace_length_values(
        &mut self,
        updates: Vec<(usize, f32)>,
        location: AnnotationLocation,
        calibration: Option<ManualLengthCalibration>,
    ) {
        for (index, raw_length) in updates {
            if let Some(crate::tools::interaction::Annotation::Length { length_mm, .. }) =
                self.annotations.get_mut(index)
            {
                *length_mm = raw_length;
            }
        }
        self.length_calibrations
            .retain(|current| !current.applies_to(location));
        if let Some(calibration) = calibration {
            self.length_calibrations.push(calibration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CalibrationError, ManualLengthCalibration};
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
        assert_eq!(
            ManualLengthCalibration::from_reference(location, 0.0, 5.0),
            Err(CalibrationError::InvalidReference)
        );
        assert_eq!(
            ManualLengthCalibration::from_reference(location, 5.0, f32::NAN),
            Err(CalibrationError::InvalidReference)
        );
        assert_eq!(
            ManualLengthCalibration::from_reference(location, f32::MAX, f32::MIN_POSITIVE),
            Err(CalibrationError::InvalidReference)
        );
    }
}
