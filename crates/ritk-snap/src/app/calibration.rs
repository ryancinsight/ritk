use crate::tools::interaction::{Annotation, AnnotationLocation, MeasurementError};

#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub(crate) enum CalibrationError {
    #[error("no distance annotation exists on the active image")]
    MissingReference,
    #[error("calibration reference no longer matches the active image")]
    StaleReference,
    #[error("known distance must be finite and strictly positive")]
    InvalidReference,
    #[error("calibrated distance is outside the supported range")]
    InvalidReplacement,
    #[error("measurement failed: {0}")]
    Measurement(#[from] MeasurementError),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ManualLengthCalibration {
    location: AnnotationLocation,
    scale: f32,
}

impl ManualLengthCalibration {
    fn from_reference(
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

    pub(crate) fn apply(self, measured_mm: f32) -> Result<f32, CalibrationError> {
        let calibrated = measured_mm * self.scale;
        (calibrated.is_finite() && calibrated > 0.0)
            .then_some(calibrated)
            .ok_or(CalibrationError::InvalidReplacement)
    }
}

#[cfg(windows)]
impl super::state::SnapApp {
    fn active_length_location(&self) -> AnnotationLocation {
        AnnotationLocation {
            plane: self.axis,
            slice: self.axis_slice_info(self.axis).0,
        }
    }

    pub(crate) fn has_active_length_calibration(&self) -> bool {
        let location = self.active_length_location();
        self.length_calibrations
            .iter()
            .any(|item| item.applies_to(location))
    }

    pub(crate) fn calibrate_length_reference(
        &mut self,
        location: AnnotationLocation,
        p1: [f32; 2],
        p2: [f32; 2],
        known_mm: f32,
    ) -> Result<(), CalibrationError> {
        if location != self.active_length_location() {
            return Err(CalibrationError::StaleReference);
        }
        let (selected, current_p1, current_p2) = self
            .last_active_length()
            .ok_or(CalibrationError::MissingReference)?;
        if current_p1 != p1 || current_p2 != p2 {
            return Err(CalibrationError::StaleReference);
        }
        let spacing = self.slice_plane_spacing()?;
        let measured = Annotation::compute_length_checked(p1, p2, spacing)?;
        let calibration = ManualLengthCalibration::from_reference(location, measured, known_mm)?;
        let updates = self
            .raw_lengths(location, spacing)?
            .into_iter()
            .map(|(index, raw)| {
                let value = if index == selected {
                    known_mm
                } else {
                    calibration.apply(raw)?
                };
                Ok((index, value))
            })
            .collect::<Result<Vec<_>, CalibrationError>>()?;
        self.replace_lengths(location, updates, Some(calibration));
        Ok(())
    }

    pub(crate) fn reset_active_length_calibration(&mut self) -> Result<bool, CalibrationError> {
        let location = self.active_length_location();
        if !self.has_active_length_calibration() {
            return Ok(false);
        }
        let spacing = self.slice_plane_spacing()?;
        let updates = self.raw_lengths(location, spacing)?;
        self.replace_lengths(location, updates, None);
        Ok(true)
    }

    pub(crate) fn last_active_length(&self) -> Option<(usize, [f32; 2], [f32; 2])> {
        let location = self.active_length_location();
        self.annotations
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, item)| match item {
                Annotation::Length {
                    p1,
                    p2,
                    location: Some(source),
                    ..
                } if *source == location => Some((index, *p1, *p2)),
                _ => None,
            })
    }

    fn raw_lengths(
        &self,
        location: AnnotationLocation,
        spacing: [f64; 2],
    ) -> Result<Vec<(usize, f32)>, CalibrationError> {
        self.annotations
            .iter()
            .enumerate()
            .filter_map(|(index, item)| match item {
                Annotation::Length {
                    p1,
                    p2,
                    location: Some(source),
                    ..
                } if *source == location => Some(
                    Annotation::compute_length_checked(*p1, *p2, spacing)
                        .map(|value| (index, value)),
                ),
                _ => None,
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(CalibrationError::from)
    }

    fn replace_lengths(
        &mut self,
        location: AnnotationLocation,
        updates: Vec<(usize, f32)>,
        calibration: Option<ManualLengthCalibration>,
    ) {
        for (index, value) in updates {
            if let Some(Annotation::Length { length_mm, .. }) = self.annotations.get_mut(index) {
                *length_mm = value;
            }
        }
        self.length_calibrations
            .retain(|item| !item.applies_to(location));
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
    fn anisotropic_reference_scales_existing_and_new_lengths() {
        let location = AnnotationLocation { plane: 0, slice: 2 };
        let calibration = ManualLengthCalibration::from_reference(location, 5.0, 8.0)
            .expect("positive reference defines a factor");
        assert_eq!(calibration.apply(5.0), Ok(8.0));
        assert_eq!(calibration.apply(2.5), Ok(4.0));
        assert!(!calibration.applies_to(AnnotationLocation { plane: 0, slice: 3 }));
    }

    #[test]
    fn invalid_reference_is_typed() {
        let location = AnnotationLocation { plane: 0, slice: 0 };
        assert_eq!(
            ManualLengthCalibration::from_reference(location, 0.0, 8.0),
            Err(CalibrationError::InvalidReference)
        );
    }
}
