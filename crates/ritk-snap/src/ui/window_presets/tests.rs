use super::*;

// ── ct_presets ────────────────────────────────────────────────────────────

/// `ct_presets()` must return exactly 14 entries — one for every entry in
/// the published reference table.
#[test]
fn test_ct_presets_count() {
    assert_eq!(
        WindowPreset::ct_presets().len(),
        14,
        "ct_presets() must return 14 entries"
    );
}

/// All CT preset widths must be strictly positive (a zero-width window
/// maps every value to the same output and is undefined under the
/// normalisation formula).
#[test]
fn test_ct_presets_widths_positive() {
    for p in WindowPreset::ct_presets() {
        assert!(
            p.width > 0.0,
            "CT preset '{}' has non-positive width {}",
            p.name,
            p.width
        );
    }
}

/// The Brain preset must have centre = 40 and width = 80 (radiology
/// standard soft-tissue brain window).
#[test]
fn test_ct_brain_preset_values() {
    let brain = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Brain")
        .expect("ct_presets() must include a 'Brain' preset");
    assert_eq!(brain.center, 40.0, "Brain preset centre must be 40 HU");
    assert_eq!(brain.width, 80.0, "Brain preset width must be 80 HU");
}

/// The Lung preset must have centre = −400 and width = 1 500
/// (DICOM standard parenchyma window).
#[test]
fn test_ct_lung_preset_values() {
    let lung = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Lung")
        .expect("ct_presets() must include a 'Lung' preset");
    assert_eq!(lung.center, -400.0, "Lung preset centre must be −400 HU");
    assert_eq!(lung.width, 1500.0, "Lung preset width must be 1 500 HU");
}

/// The Bone preset must have centre = 400 and width = 1 000.
#[test]
fn test_ct_bone_preset_values() {
    let bone = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Bone")
        .expect("ct_presets() must include a 'Bone' preset");
    assert_eq!(bone.center, 400.0, "Bone preset centre must be 400 HU");
    assert_eq!(bone.width, 1000.0, "Bone preset width must be 1 000 HU");
}

/// All CT preset names must be non-empty and distinct.
#[test]
fn test_ct_preset_names_non_empty_and_distinct() {
    let presets = WindowPreset::ct_presets();
    for p in presets {
        assert!(!p.name.is_empty(), "CT preset name must not be empty");
    }
    for i in 0..presets.len() {
        for j in (i + 1)..presets.len() {
            assert_ne!(
                presets[i].name, presets[j].name,
                "CT presets must have distinct names; duplicate at [{i}] and [{j}]"
            );
        }
    }
}

// ── mr_presets ────────────────────────────────────────────────────────────

/// `mr_presets()` must return exactly 4 entries.
#[test]
fn test_mr_presets_count() {
    assert_eq!(
        WindowPreset::mr_presets().len(),
        4,
        "mr_presets() must return 4 entries"
    );
}

/// All MR preset widths must be strictly positive.
#[test]
fn test_mr_presets_widths_positive() {
    for p in WindowPreset::mr_presets() {
        assert!(
            p.width > 0.0,
            "MR preset '{}' has non-positive width {}",
            p.name,
            p.width
        );
    }
}

/// Brain T1 preset must have centre = 500 and width = 800.
#[test]
fn test_mr_brain_t1_values() {
    let t1 = WindowPreset::mr_presets()
        .iter()
        .find(|p| p.name == "Brain T1")
        .expect("mr_presets() must include 'Brain T1'");
    assert_eq!(t1.center, 500.0, "Brain T1 centre must be 500");
    assert_eq!(t1.width, 800.0, "Brain T1 width must be 800");
}

// ── for_modality dispatch ─────────────────────────────────────────────────

/// `for_modality(Some("CT"))` must return the CT preset list.
#[test]
fn test_for_modality_ct() {
    let presets = WindowPreset::for_modality(Some("CT"));
    assert_eq!(
        presets.len(),
        WindowPreset::ct_presets().len(),
        "for_modality('CT') must return the CT preset list"
    );
    // Verify identity by checking the first preset's centre.
    assert_eq!(
        presets[0].center,
        WindowPreset::ct_presets()[0].center,
        "for_modality('CT') first preset centre must match ct_presets()[0]"
    );
}

/// `for_modality(Some("MR"))` must return the MR preset list.
#[test]
fn test_for_modality_mr() {
    let presets = WindowPreset::for_modality(Some("MR"));
    assert_eq!(
        presets.len(),
        WindowPreset::mr_presets().len(),
        "for_modality('MR') must return the MR preset list"
    );
}

/// `for_modality(None)` must return the CT preset list (safe default).
#[test]
fn test_for_modality_none_defaults_to_ct() {
    let presets = WindowPreset::for_modality(None);
    assert_eq!(
        presets.len(),
        WindowPreset::ct_presets().len(),
        "for_modality(None) must fall back to the CT preset list"
    );
}

/// `for_modality` must be case-insensitive: `"ct"` and `"mr"` must
/// resolve the same as their upper-case equivalents.
#[test]
fn test_for_modality_case_insensitive() {
    assert_eq!(
        WindowPreset::for_modality(Some("ct")).len(),
        WindowPreset::ct_presets().len(),
        "for_modality('ct') must equal for_modality('CT')"
    );
    assert_eq!(
        WindowPreset::for_modality(Some("mr")).len(),
        WindowPreset::mr_presets().len(),
        "for_modality('mr') must equal for_modality('MR')"
    );
}

/// Unknown modality strings must fall back to CT presets.
#[test]
fn test_for_modality_unknown_falls_back_to_ct() {
    for m in &["NM", "US", "XA", "CR", "DX", "OT", "UNKNOWN"] {
        let presets = WindowPreset::for_modality(Some(m));
        assert_eq!(
            presets.len(),
            WindowPreset::ct_presets().len(),
            "for_modality('{m}') must fall back to CT presets"
        );
    }
}

// ── pt_presets ────────────────────────────────────────────────────────────

/// `pt_presets()` must return exactly 3 entries.
#[test]
fn test_pt_presets_count() {
    assert_eq!(
        WindowPreset::pt_presets().len(),
        3,
        "pt_presets() must return 3 entries"
    );
}

/// All PT preset widths must be strictly positive.
#[test]
fn test_pt_presets_widths_positive() {
    for p in WindowPreset::pt_presets() {
        assert!(
            p.width > 0.0,
            "PT preset '{}' has non-positive width {}",
            p.name,
            p.width
        );
    }
}

/// "SUV whole body" preset must have centre = 3.0 and width = 6.0.
///
/// Basis: covers [0.0, 6.0] SUVbw, the typical whole-body FDG distribution
/// range per SNMMI Procedure Guideline v4.0 (2022).
#[test]
fn test_pt_suv_whole_body_values() {
    let wb = WindowPreset::pt_presets()
        .iter()
        .find(|p| p.name == "SUV whole body")
        .expect("pt_presets() must include 'SUV whole body'");
    assert_eq!(wb.center, 3.0, "SUV whole body centre must be 3.0");
    assert_eq!(wb.width, 6.0, "SUV whole body width must be 6.0");
}

/// `for_modality(Some("PT"))` must return the PT preset list.
#[test]
fn test_for_modality_pt() {
    let presets = WindowPreset::for_modality(Some("PT"));
    assert_eq!(
        presets.len(),
        WindowPreset::pt_presets().len(),
        "for_modality('PT') must return the PT preset list"
    );
    assert_eq!(
        presets[0].center,
        WindowPreset::pt_presets()[0].center,
        "for_modality('PT') first preset centre must match pt_presets()[0]"
    );
}
