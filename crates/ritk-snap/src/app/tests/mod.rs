//! Domain-partitioned test modules for `SnapApp`.
//!
//! Each submodule exercises one cohesive domain: cursor, navigation, session,
//! tool, measurement, colormap, segmentation loading, and RT dose/plan.

use crate::app::state::SnapApp;
use crate::render::colormap::Colormap;
use crate::tools::kind::ToolKind;
use crate::{LoadedVolume, ViewerState};
use std::sync::Arc;

#[cfg(test)]
mod colormap;
#[cfg(test)]
mod cursor;
#[cfg(test)]
mod distribution;
#[cfg(test)]
mod measurement;
#[cfg(test)]
mod navigation;
#[cfg(test)]
mod rt;
#[cfg(test)]
mod seg_load;
#[cfg(test)]
mod session;
#[cfg(test)]
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
        modality: Some("CT".to_string()),
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
