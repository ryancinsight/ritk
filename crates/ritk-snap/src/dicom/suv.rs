//! Standard Uptake Value — body-weight normalisation (SUVbw).
//!
//! # Mathematical specification
//!
//! ## Definition (SNMMI guidelines / IAEA Human Health Series No. 9)
//!
//! ```text
//! SUVbw(t) = C(t) / (A₀ · F(t) / BW)
//!
//! where
//!   C(t)  = pixel activity concentration at scan time t   [Bq/mL]
//!   A₀    = injected activity at injection time            [Bq]
//!   F(t)  = physical decay factor at scan time             [dimensionless]
//!         = exp(−λ · Δt),  λ = ln 2 / T½
//!   Δt    = elapsed time from injection to scan start      [s]
//!   T½    = radionuclide physical half-life                 [s]
//!   BW    = patient body weight                            [g]
//! ```
//!
//! ## Unit proof
//!
//! ```text
//! [SUVbw] = [Bq/mL] / ([Bq] · 1 / [g])
//!         = [Bq/mL] · [g/Bq]
//!         = [g/mL]
//! ```
//!
//! At tissue density ρ ≈ 1 g/mL the value is effectively dimensionless.
//! SUVbw = 1 ⟺ voxel uptake equals the whole-body average.
//!
//! ## Decay-corrected DICOM pixels
//!
//! When DICOM attribute (0054,1102) Decay Correction = "START", the scanner has
//! already corrected pixel values to injection-time activity.  In that case set
//! `decay_factor = 1.0` via [`SuvParams::without_decay_correction`].
//!
//! When Decay Correction = "NONE", raw pixel values represent activity at scan
//! time; compute the decay factor via [`SuvParams::with_decay_correction`].
//!
//! ## References
//!
//! - SNMMI Procedure Guideline for Tumor Imaging with ¹â¸F-FDG PET/CT, v4.0 (2022)
//! - IAEA Human Health Series No. 9, *Nuclear Medicine Physics* (2014) §10.3
//! - DICOM PS3.3 §C.8.9.1 (PET Series Module), §C.8.9.4.1.1 (Rescale Slope/Intercept)

use std::f64::consts::LN_2;

// ── SuvParams ─────────────────────────────────────────────────────────────────

/// Parameters required to compute SUVbw for a single voxel.
///
/// # Invariants
///
/// - `injected_dose_bq > 0` — zero or negative dose is unphysical and produces
///   ±∞ or NaN from the normalisation denominator.
/// - `patient_weight_g > 0` — same.
/// - `decay_factor ∈ (0, 1]` — decay reduces activity; must be strictly positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuvParams {
    /// Injected activity at injection time \[Bq\].
    pub injected_dose_bq: f64,
    /// Patient body weight \[g\].
    pub patient_weight_g: f64,
    /// Physical decay factor F(t) = exp(−λ · Δt) \[dimensionless, ∈ (0, 1\].
    ///
    /// Set to `1.0` when DICOM pixels are already decay-corrected to injection
    /// time (Decay Correction = "START").
    pub decay_factor: f64,
}

impl SuvParams {
    /// Construct `SuvParams` for decay-corrected DICOM pixels.
    ///
    /// Use when (0054,1102) Decay Correction = "START": pixel values already
    /// represent injection-time activity; `decay_factor = 1.0`.
    #[inline]
    pub fn without_decay_correction(injected_dose_bq: f64, patient_weight_g: f64) -> Self {
        debug_assert!(
            injected_dose_bq > 0.0,
            "injected_dose_bq must be positive, got {injected_dose_bq}"
        );
        debug_assert!(
            patient_weight_g > 0.0,
            "patient_weight_g must be positive, got {patient_weight_g}"
        );
        Self {
            injected_dose_bq,
            patient_weight_g,
            decay_factor: 1.0,
        }
    }

    /// Construct `SuvParams` with physical decay correction.
    ///
    /// Use when (0054,1102) Decay Correction = "NONE": pixel values represent
    /// raw activity at scan time.
    ///
    /// ```text
    /// F(t) = exp(−ln 2 · Δt / T½)
    /// ```
    ///
    /// # Common half-lives
    ///
    /// | Radionuclide | T½ \[s\] |
    /// |--------------|----------|
    /// | ¹â¸F          | 6 584.04 |
    /// | ¹¹C          | 1 223.4  |
    /// | ⁶⁸Ga         | 4 065.0  |
    /// | ⁸²Rb         |   75.0   |
    #[inline]
    pub fn with_decay_correction(
        injected_dose_bq: f64,
        patient_weight_g: f64,
        half_life_s: f64,
        delta_t_s: f64,
    ) -> Self {
        debug_assert!(
            injected_dose_bq > 0.0,
            "injected_dose_bq must be positive, got {injected_dose_bq}"
        );
        debug_assert!(
            patient_weight_g > 0.0,
            "patient_weight_g must be positive, got {patient_weight_g}"
        );
        debug_assert!(
            half_life_s > 0.0,
            "half_life_s must be positive, got {half_life_s}"
        );
        debug_assert!(
            delta_t_s >= 0.0,
            "delta_t_s must be non-negative, got {delta_t_s}"
        );
        let decay_factor = (-LN_2 * delta_t_s / half_life_s).exp();
        Self {
            injected_dose_bq,
            patient_weight_g,
            decay_factor,
        }
    }
}

// ── compute_suvbw ─────────────────────────────────────────────────────────────

/// Compute SUVbw for a single voxel.
///
/// # Mathematical definition
///
/// ```text
/// SUVbw = C(t) / (A₀ · F(t) / BW)
///       = pixel_bqml · patient_weight_g / (injected_dose_bq · decay_factor)
/// ```
///
/// # Returns
///
/// SUVbw in \[g/mL\] (dimensionless at tissue density ≈ 1 g/mL).
/// Returns `0.0` when `pixel_bqml = 0.0`.
/// Returns `+∞` when the normalisation denominator is zero (invalid `SuvParams`).
#[inline]
pub fn compute_suvbw(pixel_bqml: f64, params: &SuvParams) -> f64 {
    pixel_bqml / (params.injected_dose_bq * params.decay_factor / params.patient_weight_g)
}

#[cfg(test)]
#[path = "tests_suv.rs"]
mod tests;
