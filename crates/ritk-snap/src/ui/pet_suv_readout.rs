//! PET acquisition and SUV readout text.
//!
//! All SUV values come from [`crate::dicom::suv::compute_suvbw`] through
//! [`crate::dicom::pet::PetAcquisitionParams`]; this module only formats them.

/// Becquerel per megabecquerel for injected-activity display.
const BECQUEREL_PER_MEGABECQUEREL: f64 = 1_000_000.0;
/// Seconds per minute for radionuclide half-life display.
const SECONDS_PER_MINUTE: f64 = 60.0;

/// One line per displayed PET quantity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PetSuvReadout {
    /// SUVbw at the pointer.
    pub(crate) pointer: String,
    /// SUVbw at the linked cursor.
    pub(crate) cursor: String,
    /// Patient body weight.
    pub(crate) weight: String,
    /// Injected radionuclide activity.
    pub(crate) dose: String,
    /// Radionuclide physical half-life.
    pub(crate) half_life: String,
    /// DICOM decay correction mode.
    pub(crate) decay: String,
}

/// Formats a readout for a PET series; any other modality has no readout.
///
/// Non-finite SUV values and non-positive physical quantities display as a dash
/// instead of a number.
pub(crate) fn pet_suv_readout(
    modality: Option<&str>,
    pointer_suv: Option<f32>,
    cursor_suv: Option<f32>,
    patient_weight_kg: Option<f64>,
    injected_dose_bq: Option<f64>,
    half_life_s: Option<f64>,
    decay_correction: Option<&str>,
) -> Option<PetSuvReadout> {
    if modality != Some("PT") {
        return None;
    }
    let suv = |label: &str, value: Option<f32>| match value {
        Some(value) if value.is_finite() => format!("{label}: {value:.2} g/mL"),
        _ => format!("{label}: —"),
    };
    let positive = |value: Option<f64>| value.filter(|value| *value > 0.0);
    Some(PetSuvReadout {
        pointer: suv("Pointer", pointer_suv),
        cursor: suv("Cursor", cursor_suv),
        weight: positive(patient_weight_kg).map_or_else(
            || "Weight: —".to_owned(),
            |kg| format!("Weight: {kg:.1} kg"),
        ),
        dose: positive(injected_dose_bq).map_or_else(
            || "Dose: —".to_owned(),
            |bq| format!("Dose: {:.1} MBq", bq / BECQUEREL_PER_MEGABECQUEREL),
        ),
        half_life: positive(half_life_s).map_or_else(
            || "T½: —".to_owned(),
            |seconds| format!("T½: {:.2} min", seconds / SECONDS_PER_MINUTE),
        ),
        decay: decay_correction
            .filter(|mode| !mode.is_empty())
            .map_or_else(|| "Decay: —".to_owned(), |mode| format!("Decay: {mode}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fluorine-18: 370 MBq injected, half-life 6584.04 s = 109.73 min.
    fn fluorine_readout(pointer: Option<f32>, cursor: Option<f32>) -> Option<PetSuvReadout> {
        pet_suv_readout(
            Some("PT"),
            pointer,
            cursor,
            Some(70.0),
            Some(370_000_000.0),
            Some(6584.04),
            Some("START"),
        )
    }

    #[test]
    fn readout_is_absent_for_non_pt_modality() {
        let readout = pet_suv_readout(Some("CT"), None, None, None, None, None, None);
        assert_eq!(readout, None);
    }

    #[test]
    fn readout_is_absent_for_missing_modality() {
        let readout = pet_suv_readout(None, None, None, None, None, None, None);
        assert_eq!(readout, None);
    }

    #[test]
    fn readout_formats_finite_suv_and_acquisition_values() {
        let readout = fluorine_readout(Some(3.5), Some(2.1)).expect("PET readout");
        assert_eq!(readout.pointer, "Pointer: 3.50 g/mL");
        assert_eq!(readout.cursor, "Cursor: 2.10 g/mL");
        assert_eq!(readout.weight, "Weight: 70.0 kg");
        assert_eq!(readout.dose, "Dose: 370.0 MBq");
        assert_eq!(readout.half_life, "T½: 109.73 min");
        assert_eq!(readout.decay, "Decay: START");
    }

    #[test]
    fn readout_dashes_missing_values() {
        let readout =
            pet_suv_readout(Some("PT"), None, None, None, None, None, None).expect("PET readout");
        assert_eq!(readout.pointer, "Pointer: —");
        assert_eq!(readout.cursor, "Cursor: —");
        assert_eq!(readout.weight, "Weight: —");
        assert_eq!(readout.dose, "Dose: —");
        assert_eq!(readout.half_life, "T½: —");
        assert_eq!(readout.decay, "Decay: —");
    }

    #[test]
    fn readout_dashes_non_finite_suv() {
        let readout = fluorine_readout(Some(f32::NAN), Some(f32::INFINITY)).expect("PET readout");
        assert_eq!(readout.pointer, "Pointer: —");
        assert_eq!(readout.cursor, "Cursor: —");
    }
}
