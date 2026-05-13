//! PET acquisition parameter SSOT for SUVbw normalisation.
//!
//! # Responsibility
//!
//! [`PetAcquisitionParams`] bridges [`crate::LoadedVolume`] (which carries raw
//! DICOM PET fields) and [`super::suv::SuvParams`] (which performs the SUVbw
//! formula).  All field validation and decay-correction mode dispatch live here.
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
    /// Patient body weight [kg].  Converted to [g] internally for `SuvParams`.
    pub patient_weight_kg: f64,
    /// Injected radionuclide total dose at injection time [Bq].
    pub injected_dose_bq: f64,
    /// Radionuclide physical half-life [s].
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
    /// `delta_t_s` is the elapsed time from injection to scan start [s].
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

    /// Compute SUVbw for a single voxel expressed in [Bq/mL].
    ///
    /// `delta_t_s` is the elapsed time from injection to scan start [s];
    /// ignored for `Start` and `Admin` decay correction modes.
    #[inline]
    pub fn pixel_to_suvbw(&self, pixel_bqml: f64, delta_t_s: f64) -> f64 {
        compute_suvbw(pixel_bqml, &self.to_suv_params(delta_t_s))
    }

    /// Compute elapsed time [s] from injection to scan using `LoadedVolume` time fields.
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

/// Compute elapsed time [s] from radiopharmaceutical injection to series acquisition.
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// ¹⁸F physical half-life = 6 584.04 s.  Source: NNDC Nuclear Data (2023).
    const F18_HALF_LIFE_S: f64 = 6_584.04;

    fn minimal_vol(
        patient_weight_kg: Option<f64>,
        injected_dose_bq: Option<f64>,
        radionuclide_half_life_s: Option<f64>,
        decay_correction: Option<&str>,
    ) -> LoadedVolume {
        LoadedVolume {
            data: Arc::new(vec![]),
            shape: [0, 0, 0],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: Some("PT".to_string()),
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
            series_time: None,
            patient_weight_kg,
            injected_dose_bq,
            radionuclide_half_life_s,
            radiopharmaceutical_start_time: None,
            decay_correction: decay_correction.map(str::to_string),
        }
    }

    // ── from_loaded_volume — missing field guards ─────────────────────────────

    #[test]
    fn from_loaded_volume_returns_none_when_weight_absent() {
        let vol = minimal_vol(
            None,
            Some(370_000_000.0),
            Some(F18_HALF_LIFE_S),
            Some("START"),
        );
        assert!(
            PetAcquisitionParams::from_loaded_volume(&vol).is_none(),
            "must return None when patient_weight_kg is absent"
        );
    }

    #[test]
    fn from_loaded_volume_returns_none_when_dose_absent() {
        let vol = minimal_vol(Some(70.0), None, Some(F18_HALF_LIFE_S), Some("START"));
        assert!(
            PetAcquisitionParams::from_loaded_volume(&vol).is_none(),
            "must return None when injected_dose_bq is absent"
        );
    }

    #[test]
    fn from_loaded_volume_returns_none_when_half_life_absent() {
        let vol = minimal_vol(Some(70.0), Some(370_000_000.0), None, Some("START"));
        assert!(
            PetAcquisitionParams::from_loaded_volume(&vol).is_none(),
            "must return None when radionuclide_half_life_s is absent"
        );
    }

    #[test]
    fn from_loaded_volume_returns_none_when_weight_zero() {
        let vol = minimal_vol(
            Some(0.0),
            Some(370_000_000.0),
            Some(F18_HALF_LIFE_S),
            Some("START"),
        );
        assert!(
            PetAcquisitionParams::from_loaded_volume(&vol).is_none(),
            "must return None when patient_weight_kg = 0"
        );
    }

    #[test]
    fn from_loaded_volume_returns_none_when_dose_negative() {
        let vol = minimal_vol(Some(70.0), Some(-1.0), Some(F18_HALF_LIFE_S), Some("START"));
        assert!(
            PetAcquisitionParams::from_loaded_volume(&vol).is_none(),
            "must return None when injected_dose_bq ≤ 0"
        );
    }

    #[test]
    fn from_loaded_volume_complete_fields_returns_some_with_correct_values() {
        let vol = minimal_vol(
            Some(70.0),
            Some(370_000_000.0),
            Some(F18_HALF_LIFE_S),
            Some("START"),
        );
        let pet = PetAcquisitionParams::from_loaded_volume(&vol)
            .expect("complete PET volume must yield Some");
        assert_eq!(pet.patient_weight_kg, 70.0);
        assert_eq!(pet.injected_dose_bq, 370_000_000.0);
        assert_eq!(pet.radionuclide_half_life_s, F18_HALF_LIFE_S);
        assert_eq!(pet.decay_correction, DecayCorrectionKind::Start);
    }

    #[test]
    fn from_loaded_volume_absent_decay_correction_defaults_to_none_kind() {
        let vol = minimal_vol(Some(70.0), Some(370_000_000.0), Some(F18_HALF_LIFE_S), None);
        let pet = PetAcquisitionParams::from_loaded_volume(&vol).unwrap();
        assert_eq!(
            pet.decay_correction,
            DecayCorrectionKind::None,
            "absent decay_correction field must default to DecayCorrectionKind::None"
        );
    }

    // ── DecayCorrectionKind::from_dicom_str ───────────────────────────────────

    #[test]
    fn decay_correction_kind_start_parses_correctly() {
        assert_eq!(
            DecayCorrectionKind::from_dicom_str("START"),
            DecayCorrectionKind::Start
        );
    }

    #[test]
    fn decay_correction_kind_admin_parses_correctly() {
        assert_eq!(
            DecayCorrectionKind::from_dicom_str("ADMIN"),
            DecayCorrectionKind::Admin
        );
    }

    #[test]
    fn decay_correction_kind_none_string_maps_to_none_kind() {
        assert_eq!(
            DecayCorrectionKind::from_dicom_str("NONE"),
            DecayCorrectionKind::None
        );
    }

    #[test]
    fn decay_correction_kind_unknown_string_maps_to_none_kind() {
        assert_eq!(
            DecayCorrectionKind::from_dicom_str("UNKNOWN_VALUE"),
            DecayCorrectionKind::None
        );
    }

    #[test]
    fn decay_correction_kind_trims_whitespace() {
        assert_eq!(
            DecayCorrectionKind::from_dicom_str("  START  "),
            DecayCorrectionKind::Start
        );
    }

    // ── to_suv_params ─────────────────────────────────────────────────────────

    /// Start correction → decay_factor = 1.0 regardless of delta_t_s.
    #[test]
    fn to_suv_params_start_gives_unit_decay_factor() {
        let pet = PetAcquisitionParams {
            patient_weight_kg: 70.0,
            injected_dose_bq: 370_000_000.0,
            radionuclide_half_life_s: F18_HALF_LIFE_S,
            decay_correction: DecayCorrectionKind::Start,
        };
        let params = pet.to_suv_params(3_600.0);
        assert_eq!(
            params.decay_factor, 1.0,
            "Start decay correction must yield decay_factor = 1.0; got {}",
            params.decay_factor
        );
    }

    /// Admin correction → decay_factor = 1.0 (treated as Start).
    #[test]
    fn to_suv_params_admin_gives_unit_decay_factor() {
        let pet = PetAcquisitionParams {
            patient_weight_kg: 70.0,
            injected_dose_bq: 370_000_000.0,
            radionuclide_half_life_s: F18_HALF_LIFE_S,
            decay_correction: DecayCorrectionKind::Admin,
        };
        let params = pet.to_suv_params(3_600.0);
        assert_eq!(
            params.decay_factor, 1.0,
            "Admin decay correction must yield decay_factor = 1.0; got {}",
            params.decay_factor
        );
    }

    /// None correction at Δt = T½ → decay_factor = 0.5.
    ///
    /// Proof: F(T½) = exp(−ln 2 · T½ / T½) = exp(−ln 2) = 0.5.
    #[test]
    fn to_suv_params_none_at_half_life_gives_half_decay_factor() {
        let pet = PetAcquisitionParams {
            patient_weight_kg: 70.0,
            injected_dose_bq: 370_000_000.0,
            radionuclide_half_life_s: F18_HALF_LIFE_S,
            decay_correction: DecayCorrectionKind::None,
        };
        let params = pet.to_suv_params(F18_HALF_LIFE_S);
        assert!(
            (params.decay_factor - 0.5).abs() < 1e-12,
            "decay_factor at Δt = T½ must equal 0.5; got {}",
            params.decay_factor
        );
    }

    /// `to_suv_params` must convert kg → g for patient_weight_g.
    ///
    /// Proof: patient_weight_g = patient_weight_kg × 1000.
    #[test]
    fn to_suv_params_converts_kg_to_g() {
        let pet = PetAcquisitionParams {
            patient_weight_kg: 70.0,
            injected_dose_bq: 370_000_000.0,
            radionuclide_half_life_s: F18_HALF_LIFE_S,
            decay_correction: DecayCorrectionKind::Start,
        };
        let params = pet.to_suv_params(0.0);
        assert_eq!(
            params.patient_weight_g, 70_000.0,
            "patient_weight_g must be patient_weight_kg × 1000; got {}",
            params.patient_weight_g
        );
    }

    // ── pixel_to_suvbw — realistic ¹⁸F-FDG case ──────────────────────────────

    /// ¹⁸F-FDG PET: 370 MBq injected, 70 kg, 1 h PI, 10 000 Bq/mL tumour voxel.
    ///
    /// DICOM pixels are already decay-corrected (Start); SUVbw = pixel × BW / dose.
    /// Expected: 10 000 × 70 000 / 370 000 000 ≈ 1.8919 g/mL.
    #[test]
    fn pixel_to_suvbw_start_corrected_realistic_fdg() {
        let pet = PetAcquisitionParams {
            patient_weight_kg: 70.0,
            injected_dose_bq: 370_000_000.0,
            radionuclide_half_life_s: F18_HALF_LIFE_S,
            decay_correction: DecayCorrectionKind::Start,
        };
        let pixel_bqml = 10_000.0_f64;
        let expected = pixel_bqml * 70_000.0 / 370_000_000.0;
        let suv = pet.pixel_to_suvbw(pixel_bqml, 0.0);
        assert!(
            (suv - expected).abs() < 1e-10,
            "expected SUVbw = {expected:.6}, got {suv:.6}"
        );
        assert!(
            suv > 1.0 && suv < 5.0,
            "realistic FDG tumour SUVbw must be in (1, 5); got {suv:.4}"
        );
    }

    /// Raw pixels (None), 1 h PI.  Decay factor reduces effective dose;
    /// SUVbw must exceed the Start-corrected value.
    #[test]
    fn pixel_to_suvbw_none_correction_exceeds_start_correction() {
        let pixel_bqml = 10_000.0_f64;
        let start = PetAcquisitionParams {
            patient_weight_kg: 70.0,
            injected_dose_bq: 370_000_000.0,
            radionuclide_half_life_s: F18_HALF_LIFE_S,
            decay_correction: DecayCorrectionKind::Start,
        };
        let none = PetAcquisitionParams {
            decay_correction: DecayCorrectionKind::None,
            ..start
        };
        let suv_start = start.pixel_to_suvbw(pixel_bqml, 3_600.0);
        let suv_none = none.pixel_to_suvbw(pixel_bqml, 3_600.0);
        assert!(
            suv_none > suv_start,
            "None-corrected SUVbw must exceed Start-corrected at Δt > 0; \
             start = {suv_start:.4}, none = {suv_none:.4}"
        );
    }

    // ── parse_dicom_tm ────────────────────────────────────────────────────────

    #[test]
    fn parse_dicom_tm_hhmmss_gives_correct_seconds() {
        // 13:30:45 = 13*3600 + 30*60 + 45 = 48645 s
        let s = parse_dicom_tm("133045").unwrap();
        assert!(
            (s - 48_645.0).abs() < 1e-9,
            "133045 must be 48645 s; got {s}"
        );
    }

    #[test]
    fn parse_dicom_tm_hhmm_gives_correct_seconds() {
        // 08:00 = 8*3600 = 28800 s
        let s = parse_dicom_tm("0800").unwrap();
        assert!((s - 28_800.0).abs() < 1e-9, "0800 must be 28800 s; got {s}");
    }

    #[test]
    fn parse_dicom_tm_hh_gives_correct_seconds() {
        // 06h = 6*3600 = 21600 s
        let s = parse_dicom_tm("06").unwrap();
        assert!((s - 21_600.0).abs() < 1e-9, "06 must be 21600 s; got {s}");
    }

    #[test]
    fn parse_dicom_tm_fractional_seconds_parsed_correctly() {
        // 12:00:00.500 = 43200.5 s
        let s = parse_dicom_tm("120000.500").unwrap();
        assert!(
            (s - 43_200.5).abs() < 1e-9,
            "120000.500 must be 43200.5 s; got {s}"
        );
    }

    #[test]
    fn parse_dicom_tm_invalid_returns_none() {
        assert!(parse_dicom_tm("").is_none(), "empty string must be None");
        assert!(
            parse_dicom_tm("AB0000").is_none(),
            "non-digit HH must be None"
        );
        assert!(parse_dicom_tm("250000").is_none(), "HH=25 must be None");
    }

    // ── compute_delta_t_s ─────────────────────────────────────────────────────

    #[test]
    fn compute_delta_t_s_normal_same_day() {
        // inject 08:00:00, scan 09:00:00 → 3600 s
        let d = compute_delta_t_s(28_800.0, 32_400.0);
        assert!(
            (d - 3_600.0).abs() < 1e-9,
            "same-day delta must be 3600 s; got {d}"
        );
    }

    #[test]
    fn compute_delta_t_s_midnight_rollover() {
        // inject 23:50:00 (85800 s), scan 00:10:00 (600 s) → 1200 s
        let d = compute_delta_t_s(85_800.0, 600.0);
        assert!(
            (d - 1_200.0).abs() < 1e-9,
            "midnight rollover must be 1200 s; got {d}"
        );
    }

    // ── delta_t_s_from_vol ────────────────────────────────────────────────────

    #[test]
    fn delta_t_s_from_vol_parses_both_fields() {
        let mut vol = minimal_vol(
            Some(70.0),
            Some(370_000_000.0),
            Some(F18_HALF_LIFE_S),
            Some("START"),
        );
        vol.radiopharmaceutical_start_time = Some("080000".to_string());
        vol.series_time = Some("090000".to_string());
        let delta = PetAcquisitionParams::delta_t_s_from_vol(&vol);
        assert!(
            (delta - 3_600.0).abs() < 1e-9,
            "delta_t_s must be 3600 s; got {delta}"
        );
    }

    #[test]
    fn delta_t_s_from_vol_missing_series_time_returns_zero() {
        let vol = minimal_vol(
            Some(70.0),
            Some(370_000_000.0),
            Some(F18_HALF_LIFE_S),
            Some("START"),
        );
        let delta = PetAcquisitionParams::delta_t_s_from_vol(&vol);
        assert_eq!(delta, 0.0, "missing series_time must return 0.0");
    }
}
