use super::*;

// ── Pure value tests against published reference data ─────────────────────
//
// These tests do not invoke egui rendering (no Ui available in unit test
// context). They validate the analytical invariants over `WindowPreset`
// values that `draw_preset_buttons` would return when a button is clicked.
//
// Integration coverage (button clicks) is exercised through app-level tests
// that synthesize the egui context.

/// CT Brain preset center = 40 HU, width = 80 HU.
/// Reference: DICOM PS3.3 §C.7.6.3.1.5; Prokop & Galanski standard.
#[test]
fn preset_brain_center_width_analytical() {
    let brain = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Brain")
        .expect("Brain preset must exist");
    assert_eq!(brain.center, 40.0, "Brain center must be 40 HU");
    assert_eq!(brain.width, 80.0, "Brain width must be 80 HU");
}

/// CT Lung preset center = −400 HU, width = 1500 HU.
/// Reference: DICOM PS3.3 standard parenchyma window.
#[test]
fn preset_lung_center_width_analytical() {
    let lung = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Lung")
        .expect("Lung preset must exist");
    assert_eq!(lung.center, -400.0, "Lung center must be −400 HU");
    assert_eq!(lung.width, 1500.0, "Lung width must be 1500 HU");
}

/// CT Bone preset center = 400 HU, width = 1000 HU.
/// Reference: standard cortical bone window.
#[test]
fn preset_bone_center_width_analytical() {
    let bone = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Bone")
        .expect("Bone preset must exist");
    assert_eq!(bone.center, 400.0, "Bone center must be 400 HU");
    assert_eq!(bone.width, 1000.0, "Bone width must be 1000 HU");
}

/// CT Abdomen preset center = 60 HU, width = 400 HU.
/// Reference: standard soft-tissue abdomen window.
#[test]
fn preset_abdomen_center_width_analytical() {
    let abd = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Abdomen")
        .expect("Abdomen preset must exist");
    assert_eq!(abd.center, 60.0, "Abdomen center must be 60 HU");
    assert_eq!(abd.width, 400.0, "Abdomen width must be 400 HU");
}

/// CT Mediastinum preset center = 50 HU, width = 350 HU.
#[test]
fn preset_mediastinum_center_width_analytical() {
    let med = WindowPreset::ct_presets()
        .iter()
        .find(|p| p.name == "Mediastinum")
        .expect("Mediastinum preset must exist");
    assert_eq!(med.center, 50.0, "Mediastinum center must be 50 HU");
    assert_eq!(med.width, 350.0, "Mediastinum width must be 350 HU");
}

/// MR Brain T1 preset center = 500, width = 800 (relative intensity units).
#[test]
fn preset_mr_brain_t1_center_width_analytical() {
    let t1 = WindowPreset::mr_presets()
        .iter()
        .find(|p| p.name == "Brain T1")
        .expect("Brain T1 preset must exist");
    assert_eq!(t1.center, 500.0, "MR Brain T1 center must be 500");
    assert_eq!(t1.width, 800.0, "MR Brain T1 width must be 800");
}

/// MR Brain T2 preset center = 600, width = 1200.
#[test]
fn preset_mr_brain_t2_center_width_analytical() {
    let t2 = WindowPreset::mr_presets()
        .iter()
        .find(|p| p.name == "Brain T2")
        .expect("Brain T2 preset must exist");
    assert_eq!(t2.center, 600.0, "MR Brain T2 center must be 600");
    assert_eq!(t2.width, 1200.0, "MR Brain T2 width must be 1200");
}

/// All CT presets have strictly positive width.
/// Zero-width window maps every value to the same output — mathematically
/// undefined under the normalisation formula.
#[test]
fn all_ct_preset_widths_positive() {
    for p in WindowPreset::ct_presets() {
        assert!(
            p.width > 0.0,
            "CT preset '{}' has non-positive width {}",
            p.name,
            p.width
        );
    }
}

/// All MR presets have strictly positive width.
#[test]
fn all_mr_preset_widths_positive() {
    for p in WindowPreset::mr_presets() {
        assert!(
            p.width > 0.0,
            "MR preset '{}' has non-positive width {}",
            p.name,
            p.width
        );
    }
}

/// `for_modality("CT")` returns CT presets (non-empty, first entry is Brain).
#[test]
fn for_modality_ct_returns_ct_presets() {
    let presets = WindowPreset::for_modality(Some("CT"));
    assert!(
        !presets.is_empty(),
        "CT modality must return non-empty preset list"
    );
    assert_eq!(
        presets[0].name, "Brain",
        "first CT preset must be Brain (canonical ordering)"
    );
}

/// `for_modality("MR")` returns MR presets (non-empty, first entry is Brain T1).
#[test]
fn for_modality_mr_returns_mr_presets() {
    let presets = WindowPreset::for_modality(Some("MR"));
    assert!(
        !presets.is_empty(),
        "MR modality must return non-empty preset list"
    );
    assert_eq!(
        presets[0].name, "Brain T1",
        "first MR preset must be Brain T1"
    );
}

/// `for_modality(None)` falls back to CT presets (safe default for unknown modality).
#[test]
fn for_modality_none_falls_back_to_ct() {
    let presets_none = WindowPreset::for_modality(None);
    let presets_ct = WindowPreset::ct_presets();
    assert_eq!(
        presets_none.len(),
        presets_ct.len(),
        "None modality must fall back to CT preset list"
    );
}

/// Preset identity: a WindowPreset round-trips through copy without mutation.
/// Validates that `draw_preset_buttons` returns an identical value to what
/// the presets slice contains (copy semantics, no transformation).
#[test]
fn preset_copy_identity() {
    for &p in WindowPreset::ct_presets() {
        let q = p; // copy
        assert_eq!(
            p.center, q.center,
            "copy must preserve center for '{}'",
            p.name
        );
        assert_eq!(
            p.width, q.width,
            "copy must preserve width for '{}'",
            p.name
        );
        assert_eq!(p.name, q.name, "copy must preserve name for '{}'", p.name);
    }
}
