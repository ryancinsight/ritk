use super::*;
use aequitas::systems::si::units::{BecquerelPerMilliliter, GramPerCubicCentimeter};
use arrayvec::ArrayString;
use std::sync::Arc;

/// ¹â¸F physical half-life = 6 584.04 s. Source: NNDC Nuclear Data (2023).
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
        channels: 1,
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: Some(ArrayString::from("PT").expect("infallible: validated precondition")),
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
        series_time: None,
        patient_weight_kg: patient_weight_kg.map(Mass::from_base),
        injected_dose_bq: injected_dose_bq.and_then(Radioactivity::from_becquerels),
        radionuclide_half_life_s: radionuclide_half_life_s.map(Time::from_base),
        radiopharmaceutical_start_time: None,
        decay_correction: decay_correction
            .map(|s| ArrayString::from(s).expect("infallible: validated precondition")),
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
    assert_eq!(pet.patient_weight_kg.into_base(), 70.0);
    assert_eq!(pet.injected_dose_bq.becquerels(), 370_000_000.0);
    assert_eq!(pet.radionuclide_half_life_s.into_base(), F18_HALF_LIFE_S);
    assert_eq!(pet.decay_correction, DecayCorrectionKind::Start);
}

#[test]
fn from_loaded_volume_requires_explicit_decay_correction() {
    let volume = minimal_vol(Some(70.0), Some(370_000_000.0), Some(F18_HALF_LIFE_S), None);
    assert!(PetAcquisitionParams::from_loaded_volume(&volume).is_none());
}

// ── DecayCorrectionKind::from_dicom_str ───────────────────────────────────

#[test]
fn decay_correction_kind_start_parses_correctly() {
    assert_eq!(
        DecayCorrectionKind::from_dicom_str("START"),
        Some(DecayCorrectionKind::Start)
    );
}

#[test]
fn decay_correction_kind_admin_parses_correctly() {
    assert_eq!(
        DecayCorrectionKind::from_dicom_str("ADMIN"),
        Some(DecayCorrectionKind::Admin)
    );
}

#[test]
fn decay_correction_kind_none_string_maps_to_none_kind() {
    assert_eq!(
        DecayCorrectionKind::from_dicom_str("NONE"),
        Some(DecayCorrectionKind::None)
    );
}

#[test]
fn decay_correction_kind_unknown_string_is_unavailable() {
    assert_eq!(DecayCorrectionKind::from_dicom_str("UNKNOWN_VALUE"), None);
}

#[test]
fn decay_correction_kind_trims_whitespace() {
    assert_eq!(
        DecayCorrectionKind::from_dicom_str(" START "),
        Some(DecayCorrectionKind::Start)
    );
}

// ── to_suv_params ─────────────────────────────────────────────────────────

/// The START dose at one physical half-life is half the administered dose.
#[test]
fn to_suv_params_start_decays_the_administered_dose() {
    let pet = PetAcquisitionParams {
        patient_weight_kg: Mass::from_base(70.0),
        injected_dose_bq: Radioactivity::from_becquerels(370_000_000.0)
            .expect("the fixture dose is finite and positive"),
        radionuclide_half_life_s: Time::from_base(F18_HALF_LIFE_S),
        decay_correction: DecayCorrectionKind::Start,
    };
    let params = pet.to_suv_params(F18_HALF_LIFE_S);
    assert_eq!(params.decay_factor, 0.5);
}

/// ADMIN values share the time reference of the administered dose.
#[test]
fn to_suv_params_admin_gives_unit_decay_factor() {
    let pet = PetAcquisitionParams {
        patient_weight_kg: Mass::from_base(70.0),
        injected_dose_bq: Radioactivity::from_becquerels(370_000_000.0)
            .expect("the fixture dose is finite and positive"),
        radionuclide_half_life_s: Time::from_base(F18_HALF_LIFE_S),
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
        patient_weight_kg: Mass::from_base(70.0),
        injected_dose_bq: Radioactivity::from_becquerels(370_000_000.0)
            .expect("the fixture dose is finite and positive"),
        radionuclide_half_life_s: Time::from_base(F18_HALF_LIFE_S),
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
        patient_weight_kg: Mass::from_base(70.0),
        injected_dose_bq: Radioactivity::from_becquerels(370_000_000.0)
            .expect("the fixture dose is finite and positive"),
        radionuclide_half_life_s: Time::from_base(F18_HALF_LIFE_S),
        decay_correction: DecayCorrectionKind::Start,
    };
    let params = pet.to_suv_params(0.0);
    assert_eq!(
        params.patient_weight_g, 70_000.0,
        "patient_weight_g must be patient_weight_kg × 1000; got {}",
        params.patient_weight_g
    );
}

// ── pixel_to_suvbw — realistic ¹â¸F-FDG case ──────────────────────────────

/// ¹â¸F-FDG PET: 370 MBq injected, 70 kg, 1 h PI, 10 000 Bq/mL tumour voxel.
///
/// DICOM pixels are ADMIN-corrected. SUVbw = pixel × BW / dose.
/// Expected: 10 000 × 70 000 / 370 000 000 ≈ 1.8919 g/mL.
#[test]
fn pixel_to_suvbw_admin_corrected_realistic_fdg() {
    let pet = PetAcquisitionParams {
        patient_weight_kg: Mass::from_base(70.0),
        injected_dose_bq: Radioactivity::from_becquerels(370_000_000.0)
            .expect("the fixture dose is finite and positive"),
        radionuclide_half_life_s: Time::from_base(F18_HALF_LIFE_S),
        decay_correction: DecayCorrectionKind::Admin,
    };
    let pixel_bqml = 10_000.0_f64;
    let expected = pixel_bqml * 70_000.0 / 370_000_000.0;
    let suv = pet
        .pixel_to_suvbw(
            ActivityConcentration::from_unit::<BecquerelPerMilliliter>(pixel_bqml),
            0.0,
        )
        .in_unit::<GramPerCubicCentimeter>();
    assert!(
        (suv - expected).abs() < 1e-10,
        "expected SUVbw = {expected:.6}, got {suv:.6}"
    );
    assert!(
        suv > 1.0 && suv < 5.0,
        "realistic FDG tumour SUVbw must be in (1, 5); got {suv:.4}"
    );
}

/// Raw pixels (None), 1 h PI. Decay factor reduces effective dose;
/// SUVbw must exceed the ADMIN value.
#[test]
fn pixel_to_suvbw_none_correction_exceeds_admin_correction() {
    let pixel_bqml = 10_000.0_f64;

    let start = PetAcquisitionParams {
        patient_weight_kg: Mass::from_base(70.0),
        injected_dose_bq: Radioactivity::from_becquerels(370_000_000.0)
            .expect("the fixture dose is finite and positive"),
        radionuclide_half_life_s: Time::from_base(F18_HALF_LIFE_S),
        decay_correction: DecayCorrectionKind::Admin,
    };
    let none = PetAcquisitionParams {
        decay_correction: DecayCorrectionKind::None,
        ..start
    };

    let suv_start = start
        .pixel_to_suvbw(
            ActivityConcentration::from_unit::<BecquerelPerMilliliter>(pixel_bqml),
            3_600.0,
        )
        .in_unit::<GramPerCubicCentimeter>();
    let suv_none = none
        .pixel_to_suvbw(
            ActivityConcentration::from_unit::<BecquerelPerMilliliter>(pixel_bqml),
            3_600.0,
        )
        .in_unit::<GramPerCubicCentimeter>();

    assert!(
        suv_none > suv_start,
        "None-corrected SUVbw must exceed ADMIN at Δt > 0; \
         start = {suv_start:.4}, none = {suv_none:.4}"
    );
}
