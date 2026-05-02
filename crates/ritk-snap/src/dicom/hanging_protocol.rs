//! Deterministic hanging protocol selection for viewer startup.
//!
//! This module is the SSOT for translating study metadata into viewer
//! presentation defaults. The decision is pure and side-effect free.

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
    /// Whether 2x2 multi-planar layout should be enabled.
    pub multi_planar: bool,
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
            ct_lung_protocol()
        } else if contains_any(&desc_upper, &["ANGIO", "CTA", "VASC"]) {
            ct_angio_protocol()
        } else if contains_any(&desc_upper, &["BONE", "SPINE", "TEMPORAL"]) {
            ct_bone_protocol()
        } else if contains_any(&desc_upper, &["BRAIN", "HEAD", "STROKE"]) {
            ct_brain_protocol()
        } else {
            ct_soft_tissue_protocol()
        }
    } else if modality_upper.starts_with("MR") {
        if contains_any(&desc_upper, &["FLAIR"]) {
            mr_flair_protocol()
        } else if contains_any(&desc_upper, &["T1", "MPRAGE"]) {
            mr_t1_protocol()
        } else if contains_any(&desc_upper, &["T2"]) {
            mr_t2_protocol()
        } else if contains_any(&desc_upper, &["SPINE"]) {
            mr_spine_protocol()
        } else {
            mr_t2_protocol()
        }
    } else {
        generic_protocol()
    };

    if shape[decision.preferred_axis.min(2)] == 0 {
        decision.preferred_axis = first_non_empty_axis(shape);
    }
    decision
}

fn contains_any(haystack_upper: &str, needles_upper: &[&str]) -> bool {
    needles_upper.iter().any(|needle| haystack_upper.contains(needle))
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

fn ct_brain_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "CT Brain",
        window_center: 40.0,
        window_width: 80.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn ct_lung_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "CT Lung",
        window_center: -400.0,
        window_width: 1500.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn ct_bone_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "CT Bone",
        window_center: 400.0,
        window_width: 1000.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn ct_angio_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "CT Angio",
        window_center: 300.0,
        window_width: 600.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn ct_soft_tissue_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "CT Soft Tissue",
        window_center: 60.0,
        window_width: 400.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn mr_t1_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "MR T1",
        window_center: 500.0,
        window_width: 800.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn mr_t2_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "MR T2",
        window_center: 600.0,
        window_width: 1200.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn mr_flair_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "MR FLAIR",
        window_center: 400.0,
        window_width: 800.0,
        preferred_axis: 0,
        multi_planar: true,
    }
}

fn mr_spine_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "MR Spine",
        window_center: 600.0,
        window_width: 1200.0,
        preferred_axis: 1,
        multi_planar: true,
    }
}

fn generic_protocol() -> HangingProtocolDecision {
    HangingProtocolDecision {
        protocol_name: "Generic",
        window_center: 128.0,
        window_width: 256.0,
        preferred_axis: 0,
        multi_planar: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_lung_series_selects_lung_protocol() {
        let decision = select_hanging_protocol(Some("CT"), Some("Chest Lung HRCT"), [128, 512, 512]);
        assert_eq!(decision.protocol_name, "CT Lung");
        assert_eq!(decision.window_center, -400.0);
        assert_eq!(decision.window_width, 1500.0);
        assert!(decision.multi_planar);
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
        assert!(!decision.multi_planar);
    }

    #[test]
    fn preferred_axis_is_repaired_when_axis_is_empty() {
        let decision = select_hanging_protocol(Some("MR"), Some("Spine"), [20, 0, 0]);
        assert_eq!(decision.preferred_axis, 0);
    }
}
