use super::*;

/// ¹⁸F physical half-life = 109.734 min × 60 s/min = 6 584.04 s.
/// Source: NNDC Nuclear Data (2023).
const F18_HALF_LIFE_S: f64 = 6_584.04;

// ── compute_suvbw — analytical identity cases ─────────────────────────────

/// SUVbw = 1 when activity concentration equals the whole-body average.
///
/// Proof: choose C = A₀/BW, F = 1.
/// SUVbw = (A₀/BW) / (A₀ · 1 / BW) = 1.
#[test]
fn suvbw_unit_dose_equals_one() {
    let dose_bq = 70_000_000.0_f64;
    let weight_g = 70_000.0_f64;
    let pixel_bqml = dose_bq / weight_g; // whole-body average
    let params = SuvParams::without_decay_correction(dose_bq, weight_g);
    let suv = compute_suvbw(pixel_bqml, &params);
    assert!(
        (suv - 1.0).abs() < 1e-12,
        "SUVbw at whole-body average concentration must equal 1.0; got {suv}"
    );
}

/// SUVbw = 2 when concentration is exactly twice the whole-body average.
///
/// Proof: C = 2 · A₀/BW → SUVbw = 2(A₀/BW)/(A₀/BW) = 2.
#[test]
fn suvbw_double_concentration_equals_two() {
    let dose_bq = 70_000_000.0_f64;
    let weight_g = 70_000.0_f64;
    let pixel_bqml = 2.0 * dose_bq / weight_g;
    let params = SuvParams::without_decay_correction(dose_bq, weight_g);
    let suv = compute_suvbw(pixel_bqml, &params);
    assert!(
        (suv - 2.0).abs() < 1e-12,
        "SUVbw at 2× whole-body average must equal 2.0; got {suv}"
    );
}

/// SUVbw = 0 for zero-activity pixels (air / background voxels).
#[test]
fn suvbw_zero_pixel_gives_zero() {
    let params = SuvParams::without_decay_correction(370_000_000.0, 70_000.0);
    let suv = compute_suvbw(0.0, &params);
    assert_eq!(suv, 0.0, "zero pixel activity must yield SUVbw = 0");
}

/// Negative pixel values (background-subtraction artefacts) yield negative
/// SUVbw without panic, NaN, or infinity.
#[test]
fn suvbw_negative_pixel_yields_negative_finite() {
    let params = SuvParams::without_decay_correction(370_000_000.0, 70_000.0);
    let suv = compute_suvbw(-100.0, &params);
    assert!(
        suv < 0.0,
        "negative pixel must yield negative SUVbw; got {suv}"
    );
    assert!(
        suv.is_finite(),
        "SUVbw for negative pixel must be finite; got {suv}"
    );
}

// ── SuvParams::without_decay_correction ───────────────────────────────────

/// `without_decay_correction` must set `decay_factor = 1.0`.
#[test]
fn without_decay_correction_sets_factor_to_one() {
    let params = SuvParams::without_decay_correction(370_000_000.0, 70_000.0);
    assert_eq!(
        params.decay_factor, 1.0,
        "decay_factor for pre-corrected inputs must be 1.0"
    );
}

// ── SuvParams::with_decay_correction ─────────────────────────────────────

/// After exactly one half-life F(T½) = exp(−ln 2) = 1/2.
///
/// Proof: F(T½) = exp(−ln 2 · T½ / T½) = exp(−ln 2) = 0.5.
#[test]
fn decay_factor_at_one_half_life_is_half() {
    let params = SuvParams::with_decay_correction(
        370_000_000.0,
        70_000.0,
        F18_HALF_LIFE_S,
        F18_HALF_LIFE_S, // Δt = T½
    );
    assert!(
        (params.decay_factor - 0.5).abs() < 1e-12,
        "decay_factor at Δt = T½ must equal 0.5; got {}",
        params.decay_factor
    );
}

/// At Δt = 0 (immediate scan), F(0) = exp(0) = 1.
#[test]
fn decay_factor_at_zero_time_is_one() {
    let params = SuvParams::with_decay_correction(370_000_000.0, 70_000.0, F18_HALF_LIFE_S, 0.0);
    assert!(
        (params.decay_factor - 1.0).abs() < 1e-15,
        "decay_factor at Δt = 0 must equal 1.0; got {}",
        params.decay_factor
    );
}

/// Decay correction doubles effective SUVbw after exactly one half-life.
///
/// Proof: with C = A₀/BW (whole-body average at injection time),
/// SUVbw = C / (A₀ · 0.5 / BW) = 2 · C · BW / A₀ = 2.
#[test]
fn decay_correction_doubles_suv_at_one_half_life() {
    let dose_bq = 70_000_000.0_f64;
    let weight_g = 70_000.0_f64;
    let pixel_bqml = dose_bq / weight_g; // whole-body average concentration
    let params =
        SuvParams::with_decay_correction(dose_bq, weight_g, F18_HALF_LIFE_S, F18_HALF_LIFE_S);
    let suv = compute_suvbw(pixel_bqml, &params);
    assert!(
        (suv - 2.0).abs() < 1e-12,
        "SUVbw at Δt = T½ with whole-body average pixel must equal 2.0; got {suv}"
    );
}

// ── Adversarial: realistic ¹⁸F-FDG case ──────────────────────────────────

/// ¹⁸F-FDG PET: 370 MBq injected, 70 kg, 1 h PI, 10 000 Bq/mL tumour voxel.
///
/// Expected value derived analytically; result must be in the clinically
/// plausible tumour SUVbw range (1, 25).
#[test]
fn suvbw_realistic_fdg_pet_case() {
    let dose_bq = 370_000_000.0_f64;
    let weight_g = 70_000.0_f64;
    let pixel_bqml = 10_000.0_f64;
    let params = SuvParams::with_decay_correction(
        dose_bq,
        weight_g,
        F18_HALF_LIFE_S,
        3_600.0, // 1 h post-injection
    );
    let suv = compute_suvbw(pixel_bqml, &params);
    let expected_decay = (-LN_2 * 3_600.0 / F18_HALF_LIFE_S).exp();
    let expected = pixel_bqml / (dose_bq * expected_decay / weight_g);
    assert!(
        (suv - expected).abs() < 1e-10,
        "realistic FDG case: expected {expected:.6}, got {suv:.6}"
    );
    assert!(
        suv > 1.0 && suv < 25.0,
        "realistic FDG tumour SUVbw must be in (1, 25); got {suv:.3}"
    );
}
