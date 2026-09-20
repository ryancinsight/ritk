//! Domain-partitioned test modules for `SnapApp`.
//!
//! Each submodule exercises one cohesive domain: cursor, navigation, session,
//! tool, measurement, colormap, segmentation loading, and RT dose/plan.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use crate::app::state::SnapApp;
use crate::render::NamedColorMap;
#[cfg(feature = "eframe-shell")]
use crate::tools::kind::ToolKind;
use crate::LoadedVolume;
#[cfg(feature = "eframe-shell")]
use crate::ViewerState;
use arrayvec::ArrayString;
use std::sync::Arc;

#[cfg(test)]
mod action_adapter;
#[cfg(test)]
mod browser_semantics;
#[cfg(test)]
mod browser_slice_selection;
#[cfg(test)]
mod colormap;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod cursor;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod dicom_workflows;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod distribution;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "eframe-shell")]
mod load_tasks;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod measurement;
#[cfg(test)]
mod navigation;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod physical_aspect;
#[cfg(test)]
mod presentation_snapshot;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod rt;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod seg_load;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod session;
#[cfg(test)]
#[cfg(feature = "eframe-shell")]
mod tool;

/// Constructs a zero-filled `LoadedVolume` with the given shape and identity
/// spatial metadata.  Used across all test modules as a minimal fixture.
pub(crate) fn test_volume(shape: [usize; 3]) -> LoadedVolume {
    let voxel_count = shape[0] * shape[1] * shape[2];
    LoadedVolume {
        data: Arc::new(vec![0.0; voxel_count]),
        shape,
        channels: 1,
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: Some("Test".to_string()),
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    }
}
