//! Adapters from PET data to body-weight SUV measurements.
//!
//! # Responsibility
//!
//! [`PetAcquisitionParams`] bridges [`crate::LoadedVolume`] (which carries raw
//! DICOM PET fields) and [`super::suv::SuvParams`] (which performs the SUVbw
//! formula) when an activity reference is supplied explicitly. Classic PET
//! viewer measurements use the source-qualified series calibration from RITK IO.
//!
//! # Decay correction modes (DICOM PS3.3 §C.8.9.1)
//!
//! | (0054,1102) value | Meaning                                  | Decay factor |
//! |-------------------|------------------------------------------|--------------|
//! | `"START"`         | Corrected to acquisition start           | exp(−λΔt)    |
//! | `"ADMIN"`         | Corrected to administration time         | 1.0          |
//! | `"NONE"`          | Raw pixel activity at the frame reference   | exp(−λΔt)    |
//!
//! # Usage
//!
//! ```ignore
//! if let Some(pet) = PetAcquisitionParams::from_loaded_volume(&vol) {
//!     let suv = pet.pixel_to_suvbw(activity_concentration, delta_t_s);
//! }
//! ```

use aequitas::systems::si::quantities::{ActivityConcentration, Mass, MassDensity, Time};
use aequitas::systems::si::units::Gram;
use ritk_io::Radioactivity;

use super::suv::{normalized_pet_activity, SuvParams};
use crate::LoadedVolume;

// ── DecayCorrectionKind ───────────────────────────────────────────────────────

/// Pixel decay-correction mode as encoded in DICOM (0054,1102).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayCorrectionKind {
    /// Pixels corrected to acquisition start.
    Start,
    /// Pixels corrected to administration time; `decay_factor = 1.0`.
    Admin,
    /// Raw activity at scan acquisition; physical decay factor must be applied.
    None,
}

impl DecayCorrectionKind {
    /// Parse a DICOM (0054,1102) Decay Correction string (case-sensitive, per PS3.3).
    ///
    /// Returns `None` for anything other than START, ADMIN, or NONE.
    pub fn from_dicom_str(s: &str) -> Option<Self> {
        match s.trim() {
            "START" => Some(Self::Start),
            "ADMIN" => Some(Self::Admin),
            "NONE" => Some(Self::None),
            _ => None,
        }
    }
}

// ── PetAcquisitionParams ──────────────────────────────────────────────────────

/// Acquisition parameters required for SUVbw normalisation of a PET series.
///
/// # Invariants
///
/// All numeric fields are strictly positive (enforced by
/// [`from_loaded_volume`][Self::from_loaded_volume], which returns `None` when
/// any required field is absent, nonfinite, or ≤ 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PetAcquisitionParams {
    /// Patient body weight \[kg\]. Converted to \[g\] internally for `SuvParams`.
    pub patient_weight_kg: Mass,
    /// Injected radionuclide total dose at injection time \[Bq\].
    pub injected_dose_bq: Radioactivity,
    /// Radionuclide physical half-life \[s\].
    pub radionuclide_half_life_s: Time,
    /// Pixel decay-correction mode from (0054,1102).
    pub decay_correction: DecayCorrectionKind,
}

impl PetAcquisitionParams {
    /// Attempt to construct from a [`LoadedVolume`].
    ///
    /// Returns `None` when any of `patient_weight_kg`, `injected_dose_bq`, or
    /// `radionuclide_half_life_s` is absent or ≤ 0.
    ///
    /// Scanned DICOM data must declare Bq/mL pixel units and a recognized mode.
    /// For a classic PET viewer measurement use the source-qualified calibration
    /// carried by `DicomReadMetadata`.
    pub fn from_loaded_volume(vol: &LoadedVolume) -> Option<Self> {
        if vol
            .metadata
            .as_ref()
            .is_some_and(|metadata| !metadata.pet_pixels_in_bq_per_ml)
        {
            return None;
        }
        let patient_weight_kg = vol
            .patient_weight_kg
            .filter(|w| w.as_base().is_finite() && *w.as_base() > 0.0)?;
        let injected_dose_bq = vol.injected_dose_bq?;
        let radionuclide_half_life_s = vol
            .radionuclide_half_life_s
            .filter(|h| h.as_base().is_finite() && *h.as_base() > 0.0)?;

        let decay_correction = vol
            .decay_correction
            .as_deref()
            .and_then(DecayCorrectionKind::from_dicom_str)?;

        Some(Self {
            patient_weight_kg,
            injected_dose_bq,
            radionuclide_half_life_s,
            decay_correction,
        })
    }

    /// Convert to [`SuvParams`] for use with [`compute_suvbw`][super::suv::compute_suvbw].
    ///
    /// `delta_t_s` is the verified elapsed time from administration to the
    /// activity reference time in seconds. It is ignored for `Admin`.
    pub fn to_suv_params(&self, delta_t_s: f64) -> SuvParams {
        let weight_g = self.patient_weight_kg.in_unit::<Gram>();
        match self.decay_correction {
            DecayCorrectionKind::Admin => {
                SuvParams::without_decay_correction(self.injected_dose_bq.becquerels(), weight_g)
            }
            DecayCorrectionKind::Start | DecayCorrectionKind::None => {
                SuvParams::with_decay_correction(
                    self.injected_dose_bq.becquerels(),
                    weight_g,
                    self.radionuclide_half_life_s.into_base(),
                    delta_t_s,
                )
            }
        }
    }

    /// Compute SUVbw for a single voxel expressed in \[Bq/mL\].
    ///
    /// `delta_t_s` is the verified elapsed time to the activity reference in
    /// seconds and is ignored for `Admin`.
    #[inline]
    pub fn pixel_to_suvbw(
        &self,
        activity_concentration: ActivityConcentration,
        delta_t_s: f64,
    ) -> MassDensity {
        normalized_pet_activity(activity_concentration, &self.to_suv_params(delta_t_s))
    }
}

/// Classic PET normalization prepared from decoded modality and source metadata.
#[derive(Clone, Copy)]
pub(crate) enum ClassicPetDisplay {
    BodyWeight,
    Activity(ritk_io::format::dicom::ClassicPetCalibration),
}

impl ClassicPetDisplay {
    pub(crate) fn for_volume(volume: &LoadedVolume) -> Option<Self> {
        let metadata = volume.metadata.as_deref()?;
        if metadata.pixels_are_body_weight_suv() {
            return Some(Self::BodyWeight);
        }
        metadata.classic_pet_calibration.map(Self::Activity)
    }

    pub(crate) fn suvbw(self, pixel: f64) -> Option<f64> {
        use aequitas::systems::si::units::{BecquerelPerMilliliter, GramPerCubicCentimeter};

        let suv = match self {
            Self::BodyWeight => pixel,
            Self::Activity(calibration) => calibration
                .suvbw(ActivityConcentration::from_unit::<BecquerelPerMilliliter>(
                    pixel,
                ))
                .in_unit::<GramPerCubicCentimeter>(),
        };
        suv.is_finite().then_some(suv)
    }
}

#[cfg(test)]
mod tests;
