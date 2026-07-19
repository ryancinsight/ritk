//! PET acquisition parameter SSOT for SUVbw normalisation.
//!
//! # Responsibility
//!
//! [`PetAcquisitionParams`] bridges [`crate::LoadedVolume`] (which carries raw
//! DICOM PET fields) and [`super::suv::SuvParams`] (which performs the SUVbw
//! formula). All field validation and decay-correction mode dispatch live here.
//!
//! # Decay correction modes (DICOM PS3.3 §C.8.9.1)
//!
//! | (0054,1102) value | Meaning                                  | Decay factor |
//! |-------------------|------------------------------------------|--------------|
//! | `"START"`         | Corrected to series start (injection)    | 1.0          |
//! | `"ADMIN"`         | Corrected to administration time         | 1.0          |
//! | `"NONE"`          | Raw pixel activity at scan acquisition   | exp(−λΔt)    |
//!
//! # Usage
//!
//! ```ignore
//! if let Some(pet) = PetAcquisitionParams::from_loaded_volume(&vol) {
//!     let suv = pet.pixel_to_suvbw(pixel_bqml, delta_t_s);
//! }
//! ```

use super::suv::{compute_suvbw, SuvParams};
use crate::LoadedVolume;

// ── DecayCorrectionKind ───────────────────────────────────────────────────────

/// Pixel decay-correction mode as encoded in DICOM (0054,1102).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayCorrectionKind {
    /// Pixels corrected to injection time; `decay_factor = 1.0`.
    Start,
    /// Pixels corrected to administration time; treated as `Start` for SUVbw.
    Admin,
    /// Raw activity at scan acquisition; physical decay factor must be applied.
    None,
}

impl DecayCorrectionKind {
    /// Parse a DICOM (0054,1102) Decay Correction string (case-sensitive, per PS3.3).
    ///
    /// "START" → [`Start`][Self::Start], "ADMIN" → [`Admin`][Self::Admin],
    /// anything else (including "NONE" and absent) → [`None`][Self::None].
    pub fn from_dicom_str(s: &str) -> Self {
        match s.trim() {
            "START" => Self::Start,
            "ADMIN" => Self::Admin,
            _ => Self::None,
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
/// any required field is absent or ≤ 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PetAcquisitionParams {
    /// Patient body weight \[kg\]. Converted to \[g\] internally for `SuvParams`.
    pub patient_weight_kg: f64,
    /// Injected radionuclide total dose at injection time \[Bq\].
    pub injected_dose_bq: f64,
    /// Radionuclide physical half-life \[s\].
    pub radionuclide_half_life_s: f64,
    /// Pixel decay-correction mode from (0054,1102).
    pub decay_correction: DecayCorrectionKind,
}

impl PetAcquisitionParams {
    /// Attempt to construct from a [`LoadedVolume`].
    ///
    /// Returns `None` when any of `patient_weight_kg`, `injected_dose_bq`, or
    /// `radionuclide_half_life_s` is absent or ≤ 0.
    ///
    /// `decay_correction` defaults to [`DecayCorrectionKind::None`] when the
    /// DICOM field is absent.
    pub fn from_loaded_volume(vol: &LoadedVolume) -> Option<Self> {
        let patient_weight_kg = vol.patient_weight_kg.filter(|&w| w > 0.0)?;
        let injected_dose_bq = vol.injected_dose_bq.filter(|&d| d > 0.0)?;
        let radionuclide_half_life_s = vol.radionuclide_half_life_s.filter(|&h| h > 0.0)?;

        let decay_correction = vol
            .decay_correction
            .as_deref()
            .map(DecayCorrectionKind::from_dicom_str)
            .unwrap_or(DecayCorrectionKind::None);

        Some(Self {
            patient_weight_kg,
            injected_dose_bq,
            radionuclide_half_life_s,
            decay_correction,
        })
    }

    /// Convert to [`SuvParams`] for use with [`compute_suvbw`].
    ///
    /// `delta_t_s` is the elapsed time from injection to scan start \[s\].
    /// Ignored (forced to 0) when `decay_correction` is
    /// [`Start`][DecayCorrectionKind::Start] or [`Admin`][DecayCorrectionKind::Admin].
    pub fn to_suv_params(&self, delta_t_s: f64) -> SuvParams {
        let weight_g = self.patient_weight_kg * 1_000.0;
        match self.decay_correction {
            DecayCorrectionKind::Start | DecayCorrectionKind::Admin => {
                SuvParams::without_decay_correction(self.injected_dose_bq, weight_g)
            }
            DecayCorrectionKind::None => SuvParams::with_decay_correction(
                self.injected_dose_bq,
                weight_g,
                self.radionuclide_half_life_s,
                delta_t_s,
            ),
        }
    }

    /// Compute SUVbw for a single voxel expressed in \[Bq/mL\].
    ///
    /// `delta_t_s` is the elapsed time from injection to scan start \[s\];
    /// ignored for `Start` and `Admin` decay correction modes.
    #[inline]
    pub fn pixel_to_suvbw(&self, pixel_bqml: f64, delta_t_s: f64) -> f64 {
        compute_suvbw(pixel_bqml, &self.to_suv_params(delta_t_s))
    }

    /// Compute elapsed time \[s\] from injection to scan using `LoadedVolume` time fields.
    ///
    /// Parses `vol.radiopharmaceutical_start_time` (0018,1072) and `vol.series_time`
    /// (0008,0031) as DICOM TM strings. Returns `0.0` when either field is absent or
    /// unparseable (safe fallback for Start/Admin corrected scans where delta_t is
    /// unused in the SUVbw formula).
    pub fn delta_t_s_from_vol(vol: &LoadedVolume) -> f64 {
        let rph_start = vol
            .radiopharmaceutical_start_time
            .as_deref()
            .and_then(parse_dicom_tm);
        let series = vol.series_time.as_deref().and_then(parse_dicom_tm);
        match (rph_start, series) {
            (Some(t0), Some(t1)) => compute_delta_t_s(t0, t1),
            _ => 0.0,
        }
    }
}

// ── DICOM TM time-field parsing ───────────────────────────────────────────────

/// Parse a DICOM TM value string (HH[MM[SS[.FFFFFF]]]) to seconds since midnight.
///
/// DICOM PS3.5 §6.2 TM format: each integer component is exactly 2 ASCII digits.
/// Returns `None` when the string is malformed, has non-digit characters in the
/// integer part, or the hours component is ≥ 24.
pub fn parse_dicom_tm(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.len() < 2 {
        return None;
    }

    let (int_part, frac_part) = if let Some(dot) = s.find('.') {
        (&s[..dot], &s[dot + 1..])
    } else {
        (s, "")
    };

    if int_part.len() < 2 || !int_part.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let hh: f64 = int_part[..2].parse().ok()?;
    if hh >= 24.0 {
        return None;
    }

    let mm: f64 = if int_part.len() >= 4 {
        int_part[2..4].parse().ok()?
    } else {
        0.0
    };

    let ss: f64 = if int_part.len() >= 6 {
        int_part[4..6].parse().ok()?
    } else {
        0.0
    };

    let frac: f64 = if !frac_part.is_empty() {
        let digits = frac_part.len().min(6);
        let n: u32 = frac_part[..digits].parse().ok()?;
        let denom = 10u32.pow(digits as u32) as f64;
        n as f64 / denom
    } else {
        0.0
    };

    Some(hh * 3600.0 + mm * 60.0 + ss + frac)
}

/// Compute elapsed time \[s\] from radiopharmaceutical injection to series acquisition.
///
/// Handles midnight rollover: if `series_time_s < rph_start_s`, the scan crossed
/// midnight and 86 400 s is added. Result ∈ [0, 86 400).
pub fn compute_delta_t_s(rph_start_s: f64, series_time_s: f64) -> f64 {
    let diff = series_time_s - rph_start_s;
    if diff < 0.0 {
        diff + 86_400.0
    } else {
        diff
    }
}

#[cfg(test)]
mod tests;
