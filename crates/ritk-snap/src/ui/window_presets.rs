//! Standard clinical window/level presets for DICOM display.
//!
//! # Mathematical specification
//!
//! A [`WindowPreset`] defines a linear intensity windowing function:
//!
//! ```text
//! L = center − width / 2       (lower display bound)
//! U = center + width / 2       (upper display bound)
//!
//! output(v) = 0     if v ≤ L
//!           = 255   if v ≥ U
//!           = round((v − L) / (U − L) × 255)   otherwise
//! ```
//!
//! Center and width values are given in Hounsfield Units (HU) for CT and in
//! relative intensity units for MR.
//!
//! ## CT reference values
//!
//! Derived from radiology standards and verified against:
//! - DICOM PS3.3 §C.7.6.3.1.5 (VOI LUT)
//! - Prokop & Galanski, *Spiral and Multislice Computed Tomography of the Body*
//! - ACR–AAPM Technical Standard for Diagnostic Medical Physics Performance
//!   Monitoring of Computed Tomography Equipment
//!
//! ## MR reference values
//!
//! MR signal is modality- and sequence-specific; values are expressed in
//! relative intensity units (scanner ADU range ≈ 0–4095 for most clinical MR
//! systems).  The supplied presets represent typical starting points for
//! interactive adjustment.

// ── WindowPreset ──────────────────────────────────────────────────────────────

/// A named clinical window/level display preset.
///
/// Both `center` and `width` are stored as `f64` in the native intensity units
/// of the modality (HU for CT; relative intensity for MR).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowPreset {
    /// Human-readable preset name for the UI menu.
    pub name: &'static str,
    /// Display window centre (midpoint of the visible intensity range).
    pub center: f64,
    /// Display window width (span of the visible intensity range).
    ///
    /// Must be > 0 for a well-defined mapping; `for_modality` guarantees
    /// this for all presets returned by this module.
    pub width: f64,
}

impl WindowPreset {
    // ── CT presets ────────────────────────────────────────────────────────────

    /// Standard CT window/level presets, derived from published radiology
    /// references.
    ///
    /// | Name                     | Centre (HU) | Width (HU) | Visible range (HU)   |
    /// |--------------------------|-------------|------------|----------------------|
    /// | Brain                    |   40        |    80      | [0, 80]              |
    /// | Brain (wide)             |   40        |   375      | [−147, 228]          |
    /// | Subdural                 |   80        |   200      | [−20, 180]           |
    /// | Stroke                   |   32        |     8      | [28, 36]             |
    /// | Lung                     | −400        | 1 500      | [−1 150, 350]        |
    /// | Lung (soft)              | −600        | 1 600      | [−1 400, 200]        |
    /// | Mediastinum              |   50        |   350      | [−125, 225]          |
    /// | Bone                     |  400        | 1 000      | [−100, 900]          |
    /// | Abdomen                  |   60        |   400      | [−140, 260]          |
    /// | Liver                    |   60        |   160      | [−20, 140]           |
    /// | Spine (soft tissue)      |   50        |   250      | [−75, 175]           |
    /// | Spine (bone)             |  400        | 1 000      | [−100, 900]          |
    /// | Angio                    |  300        |   600      | [0, 600]             |
    /// | Head (temporal bone)     |  500        | 4 000      | [−1 500, 2 500]      |
    pub fn ct_presets() -> &'static [WindowPreset] {
        &[
            WindowPreset {
                name: "Brain",
                center: 40.0,
                width: 80.0,
            },
            WindowPreset {
                name: "Brain (wide)",
                center: 40.0,
                width: 375.0,
            },
            WindowPreset {
                name: "Subdural",
                center: 80.0,
                width: 200.0,
            },
            WindowPreset {
                name: "Stroke",
                center: 32.0,
                width: 8.0,
            },
            WindowPreset {
                name: "Lung",
                center: -400.0,
                width: 1500.0,
            },
            WindowPreset {
                name: "Lung (soft)",
                center: -600.0,
                width: 1600.0,
            },
            WindowPreset {
                name: "Mediastinum",
                center: 50.0,
                width: 350.0,
            },
            WindowPreset {
                name: "Bone",
                center: 400.0,
                width: 1000.0,
            },
            WindowPreset {
                name: "Abdomen",
                center: 60.0,
                width: 400.0,
            },
            WindowPreset {
                name: "Liver",
                center: 60.0,
                width: 160.0,
            },
            WindowPreset {
                name: "Spine (soft)",
                center: 50.0,
                width: 250.0,
            },
            WindowPreset {
                name: "Spine (bone)",
                center: 400.0,
                width: 1000.0,
            },
            WindowPreset {
                name: "Angio",
                center: 300.0,
                width: 600.0,
            },
            WindowPreset {
                name: "Head (temporal bone)",
                center: 500.0,
                width: 4000.0,
            },
        ]
    }

    // ── MR presets ────────────────────────────────────────────────────────────

    /// Standard MR window/level presets for common brain and spine sequences.
    ///
    /// Values are expressed in relative intensity units (typical 12-bit ADU
    /// range 0–4095).
    ///
    /// | Name          | Centre | Width |
    /// |---------------|--------|-------|
    /// | Brain T1      |  500   |  800  |
    /// | Brain T2      |  600   | 1200  |
    /// | Brain FLAIR   |  400   |  800  |
    /// | Spine         |  600   | 1200  |
    pub fn mr_presets() -> &'static [WindowPreset] {
        &[
            WindowPreset {
                name: "Brain T1",
                center: 500.0,
                width: 800.0,
            },
            WindowPreset {
                name: "Brain T2",
                center: 600.0,
                width: 1200.0,
            },
            WindowPreset {
                name: "Brain FLAIR",
                center: 400.0,
                width: 800.0,
            },
            WindowPreset {
                name: "Spine",
                center: 600.0,
                width: 1200.0,
            },
        ]
    }

    // ── Modality dispatch ─────────────────────────────────────────────────────

    /// Auto-select the appropriate preset list for `modality`.
    ///
    /// | Modality prefix | Returns            |
    /// |-----------------|--------------------|
    /// | `"CT"`          | [`ct_presets()`]   |
    /// | `"MR"`          | [`mr_presets()`]   |
    /// | `None` / other  | [`ct_presets()`] (safe default; widest applicable set) |
    ///
    /// The match is case-insensitive and checks the first two characters to
    /// handle modality strings like `"CT"`, `"CTa"`, `"MR"`, `"MRI"` etc.
    ///
    /// [`ct_presets()`]: WindowPreset::ct_presets
    /// [`mr_presets()`]: WindowPreset::mr_presets
    pub fn for_modality(modality: Option<&str>) -> &'static [WindowPreset] {
        match modality {
            Some(m) => {
                let upper = m.to_uppercase();
                if upper.starts_with("MR") {
                    Self::mr_presets()
                } else {
                    // CT, PT, NM, US, XA, CR, DR, DX, MG, RF, and unknown all
                    // default to the CT preset list which is the most complete
                    // and provides a safe initial view.
                    Self::ct_presets()
                }
            }
            None => Self::ct_presets(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        for m in &["PT", "NM", "US", "XA", "CR", "DX", "OT", "UNKNOWN"] {
            let presets = WindowPreset::for_modality(Some(m));
            assert_eq!(
                presets.len(),
                WindowPreset::ct_presets().len(),
                "for_modality('{m}') must fall back to CT presets"
            );
        }
    }
}
