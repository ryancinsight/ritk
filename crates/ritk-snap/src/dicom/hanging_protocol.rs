//! Deterministic hanging protocol selection for viewer startup.
//!
//! This module is the SSOT for translating study metadata into viewer
//! presentation defaults. The decision is pure and side-effect free.

/// Suggested viewer layout mode derived from DICOM modality/study metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutSuggestion {
    /// Single-pane axial view.
    SinglePane,
    /// 2Ã—2 multi-planar reformat (axial, coronal, sagittal + 3D).
    MultiPlanarReformat,
}

/// Selected startup presentation policy for one loaded study.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HangingProtocolDecision {
    /// Human-readable protocol name for status/debug reporting.
    pub protocol_name: &'static str,
    /// Suggested window center.
    pub window_center: f32,
    /// Suggested window width.
    pub window_width: f32,
    /// Preferred primary axis: 0=axial, 1=coronal, 2=sagittal.
    pub preferred_axis: usize,
    /// Suggested viewer layout mode.
    pub layout: LayoutSuggestion,
}

/// Select a hanging protocol from modality and series description.
///
/// The matching contract is deterministic and case-insensitive:
/// - CT rules prioritize lung, bone, angio, brain, abdomen/soft tissue.
/// - MR rules prioritize FLAIR, T1, T2, spine.
/// - Unknown/absent metadata falls back to a generic protocol.
///
/// `shape` is used only to guard axis validity for degenerate dimensions.
pub fn select_hanging_protocol(
    modality: Option<&str>,
    series_description: Option<&str>,
    shape: [usize; 3],
) -> HangingProtocolDecision {
    let modality_upper = modality.unwrap_or("").to_uppercase();
    let desc_upper = series_description.unwrap_or("").to_uppercase();

    let mut decision = if modality_upper.starts_with("CT") {
        if contains_any(&desc_upper, &["LUNG", "CHEST"]) {
            CT_LUNG
        } else if contains_any(&desc_upper, &["ANGIO", "CTA", "VASC"]) {
            CT_ANGIO
        } else if contains_any(&desc_upper, &["BONE", "SPINE", "TEMPORAL"]) {
            CT_BONE
        } else if contains_any(&desc_upper, &["BRAIN", "HEAD", "STROKE"]) {
            CT_BRAIN
        } else {
            CT_SOFT_TISSUE
        }
    } else if modality_upper.starts_with("MR") {
        if contains_any(&desc_upper, &["FLAIR"]) {
            MR_FLAIR
        } else if contains_any(&desc_upper, &["T1", "MPRAGE"]) {
            MR_T1
        } else if contains_any(&desc_upper, &["T2"]) {
            MR_T2
        } else if contains_any(&desc_upper, &["SPINE"]) {
            MR_SPINE
        } else {
            MR_T2
        }
    } else if modality_upper.starts_with("PT") {
        PET_SUV
    } else {
        GENERIC
    };

    if shape[decision.preferred_axis.min(2)] == 0 {
        decision.preferred_axis = first_non_empty_axis(shape);
    }
    decision
}

fn contains_any(haystack_upper: &str, needles_upper: &[&str]) -> bool {
    needles_upper
        .iter()
        .any(|needle| haystack_upper.contains(needle))
}

fn first_non_empty_axis(shape: [usize; 3]) -> usize {
    if shape[0] > 0 {
        0
    } else if shape[1] > 0 {
        1
    } else {
        2
    }
}

// â”€â”€ Protocol constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// All window/axis/layout values live here as named consts.
// Adding a new protocol: add one const, update `select_hanging_protocol`.

const MPR: LayoutSuggestion = LayoutSuggestion::MultiPlanarReformat;
const SGL: LayoutSuggestion = LayoutSuggestion::SinglePane;

const CT_BRAIN: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "CT Brain",
    window_center: 40.0,
    window_width: 80.0,
    preferred_axis: 0,
    layout: MPR,
};
const CT_LUNG: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "CT Lung",
    window_center: -400.0,
    window_width: 1500.0,
    preferred_axis: 0,
    layout: MPR,
};
const CT_BONE: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "CT Bone",
    window_center: 400.0,
    window_width: 1000.0,
    preferred_axis: 0,
    layout: MPR,
};
const CT_ANGIO: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "CT Angio",
    window_center: 300.0,
    window_width: 600.0,
    preferred_axis: 0,
    layout: MPR,
};
const CT_SOFT_TISSUE: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "CT Soft Tissue",
    window_center: 60.0,
    window_width: 400.0,
    preferred_axis: 0,
    layout: MPR,
};
const MR_T1: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "MR T1",
    window_center: 500.0,
    window_width: 800.0,
    preferred_axis: 0,
    layout: MPR,
};
const MR_T2: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "MR T2",
    window_center: 600.0,
    window_width: 1200.0,
    preferred_axis: 0,
    layout: MPR,
};
const MR_FLAIR: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "MR FLAIR",
    window_center: 400.0,
    window_width: 800.0,
    preferred_axis: 0,
    layout: MPR,
};
const MR_SPINE: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "MR Spine",
    window_center: 600.0,
    window_width: 1200.0,
    preferred_axis: 1,
    layout: MPR,
};
const GENERIC: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "Generic",
    window_center: 128.0,
    window_width: 256.0,
    preferred_axis: 0,
    layout: SGL,
};
const PET_SUV: HangingProtocolDecision = HangingProtocolDecision {
    protocol_name: "PET SUV",
    window_center: 3.0,
    window_width: 6.0,
    preferred_axis: 0,
    layout: MPR,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_lung_series_selects_lung_protocol() {
        let decision =
            select_hanging_protocol(Some("CT"), Some("Chest Lung HRCT"), [128, 512, 512]);
        assert_eq!(decision.protocol_name, "CT Lung");
        assert_eq!(decision.window_center, -400.0);
        assert_eq!(decision.window_width, 1500.0);
        assert!(decision.layout == LayoutSuggestion::MultiPlanarReformat);
    }

    #[test]
    fn ct_brain_series_selects_brain_protocol() {
        let decision = select_hanging_protocol(Some("CT"), Some("Head Stroke"), [64, 256, 256]);
        assert_eq!(decision.protocol_name, "CT Brain");
        assert_eq!(decision.window_center, 40.0);
        assert_eq!(decision.window_width, 80.0);
    }

    #[test]
    fn mr_flair_series_selects_flair_protocol() {
        let decision = select_hanging_protocol(Some("MR"), Some("Brain Ax FLAIR"), [120, 320, 320]);
        assert_eq!(decision.protocol_name, "MR FLAIR");
        assert_eq!(decision.window_center, 400.0);
        assert_eq!(decision.window_width, 800.0);
    }

    #[test]
    fn mr_spine_prefers_coronal_axis() {
        let decision = select_hanging_protocol(Some("MR"), Some("Spine Survey"), [90, 256, 256]);
        assert_eq!(decision.protocol_name, "MR Spine");
        assert_eq!(decision.preferred_axis, 1);
    }

    #[test]
    fn unknown_modality_falls_back_to_generic() {
        let decision = select_hanging_protocol(Some("US"), Some("Abdomen"), [10, 100, 120]);
        assert_eq!(decision.protocol_name, "Generic");
        assert_eq!(decision.layout, LayoutSuggestion::SinglePane);
    }

    #[test]
    fn pet_series_selects_suv_protocol() {
        let decision = select_hanging_protocol(Some("PT"), Some("FDG PET"), [120, 256, 256]);
        assert_eq!(decision.protocol_name, "PET SUV");
        assert_eq!(decision.window_center, 3.0);
        assert_eq!(decision.window_width, 6.0);
        assert_eq!(decision.preferred_axis, 0);
        assert_eq!(decision.layout, LayoutSuggestion::MultiPlanarReformat);
    }

    #[test]
    fn preferred_axis_is_repaired_when_axis_is_empty() {
        let decision = select_hanging_protocol(Some("MR"), Some("Spine"), [20, 0, 0]);
        assert_eq!(decision.preferred_axis, 0);
    }
}
