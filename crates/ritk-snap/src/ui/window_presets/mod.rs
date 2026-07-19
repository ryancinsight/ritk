//! Standard clinical window/level presets for DICOM display.
//!
//! # Mathematical specification
//!
//! A [`WindowPreset`] defines a linear intensity windowing function:
//!
//! ```text
//! L = center âˆ’ width / 2   (lower display bound)
//! U = center + width / 2   (upper display bound)
//!
//! output(v) = 0   if v â‰¤ L
//!           = 255 if v â‰¥ U
//!           = round((v âˆ’ L) / (U âˆ’ L) Ã— 255)  otherwise
//! ```
//!
//! Center and width values are given in Hounsfield Units (HU) for CT and in
//! relative intensity units for MR.
//!
//! ## CT reference values
//!
//! Derived from radiology standards and verified against:
//! - DICOM PS3.3 Â§C.7.6.3.1.5 (VOI LUT)
//! - Prokop & Galanski, *Spiral and Multislice Computed Tomography of the Body*
//! - ACRâ€“AAPM Technical Standard for Diagnostic Medical Physics Performance
//!   Monitoring of Computed Tomography Equipment
//!
//! ## MR reference values
//!
//! MR signal is modality- and sequence-specific; values are expressed in
//! relative intensity units (scanner ADU range â‰ˆ 0â€“4095 for most clinical MR
//! systems). The supplied presets represent typical starting points for
//! interactive adjustment.

// â”€â”€ WindowPreset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    // â”€â”€ CT presets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Standard CT window/level presets, derived from published radiology
    /// references.
    ///
    /// | Name                   | Centre (HU) | Width (HU) | Visible range (HU)   |
    /// |--------------------------|-------------|------------|----------------------|
    /// | Brain                  | 40          | 80         | [0, 80]              |
    /// | Brain (wide)           | 40          | 375        | [âˆ’147, 228]          |
    /// | Subdural               | 80          | 200        | [âˆ’20, 180]           |
    /// | Stroke                 | 32          | 8          | [28, 36]             |
    /// | Lung                   | âˆ’400        | 1 500      | [âˆ’1 150, 350]        |
    /// | Lung (soft)            | âˆ’600        | 1 600      | [âˆ’1 400, 200]        |
    /// | Mediastinum            | 50          | 350        | [âˆ’125, 225]          |
    /// | Bone                   | 400         | 1 000      | [âˆ’100, 900]          |
    /// | Abdomen                | 60          | 400        | [âˆ’140, 260]          |
    /// | Liver                  | 60          | 160        | [âˆ’20, 140]           |
    /// | Spine (soft tissue)    | 50          | 250        | [âˆ’75, 175]           |
    /// | Spine (bone)           | 400         | 1 000      | [âˆ’100, 900]          |
    /// | Angio                  | 300         | 600        | [0, 600]             |
    /// | Head (temporal bone)   | 500         | 4 000      | [âˆ’1 500, 2 500]      |
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

    // â”€â”€ PT (PET) presets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Standard PET window/level presets expressed in SUVbw units \[g/mL\].
    ///
    /// At tissue density â‰ˆ 1 g/mL, SUVbw is effectively dimensionless.
    /// SUVbw = 1.0 âŸº voxel uptake equals the whole-body average.
    ///
    /// | Name              | Centre (SUV) | Width (SUV) | Visible range (SUV) |
    /// |-------------------|-------------|------------|---------------------|
    /// | SUV whole body    | 3.0         | 6.0        | [0.0, 6.0]          |
    /// | SUV brain (FDG)   | 6.0         | 12.0       | [0.0, 12.0]         |
    /// | SUV tumour        | 5.0         | 10.0       | [0.0, 10.0]         |
    ///
    /// References:
    /// - SNMMI Procedure Guideline for Â¹â¸F-FDG PET/CT, v4.0 (2022)
    /// - EANM FDG PET/CT: EANM Procedure Guidelines for Tumour Imaging (2015)
    pub fn pt_presets() -> &'static [WindowPreset] {
        &[
            WindowPreset {
                name: "SUV whole body",
                center: 3.0,
                width: 6.0,
            },
            WindowPreset {
                name: "SUV brain (FDG)",
                center: 6.0,
                width: 12.0,
            },
            WindowPreset {
                name: "SUV tumour",
                center: 5.0,
                width: 10.0,
            },
        ]
    }

    // â”€â”€ MR presets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Standard MR window/level presets for common brain and spine sequences.
    ///
    /// Values are expressed in relative intensity units (typical 12-bit ADU
    /// range 0â€“4095).
    ///
    /// | Name        | Centre | Width |
    /// |-------------|--------|-------|
    /// | Brain T1    | 500    | 800   |
    /// | Brain T2    | 600    | 1200  |
    /// | Brain FLAIR | 400    | 800   |
    /// | Spine       | 600    | 1200  |
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

    // â”€â”€ Modality dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Auto-select the appropriate preset list for `modality`.
    ///
    /// | Modality prefix | Returns            |
    /// |-----------------|--------------------|
    /// | `"CT"`          | [`ct_presets()`]   |
    /// | `"MR"`          | [`mr_presets()`]   |
    /// | `"PT"`          | [`pt_presets()`]   |
    /// | `None` / other  | [`ct_presets()`] (safe default; widest applicable set) |
    ///
    /// The match is case-insensitive and checks the first two characters to
    /// handle modality strings like `"CT"`, `"CTa"`, `"MR"`, `"MRI"` etc.
    ///
    /// [`ct_presets()`]: WindowPreset::ct_presets
    /// [`mr_presets()`]: WindowPreset::mr_presets
    /// [`pt_presets()`]: WindowPreset::pt_presets
    pub fn for_modality(modality: Option<&str>) -> &'static [WindowPreset] {
        match modality {
            Some(m) => {
                let upper = m.to_uppercase();
                if upper.starts_with("MR") {
                    Self::mr_presets()
                } else if upper.starts_with("PT") {
                    Self::pt_presets()
                } else {
                    // CT, NM, US, XA, CR, DR, DX, MG, RF, and unknown all
                    // default to the CT preset list which is the most complete
                    // and provides a safe initial view.
                    Self::ct_presets()
                }
            }
            None => Self::ct_presets(),
        }
    }
}

#[cfg(test)]
mod tests;
