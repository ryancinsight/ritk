//! Viewer session snapshot persistence.
//!
//! A session snapshot captures complete presentation state including
//! navigation, window/level, colormap, annotations, and tool selections.
//! Pixel data, DICOM metadata, and filesystem caches are not persisted.
//!
//! # SSOT file I/O
//!
//! [`save_to_file`] and [`load_from_file`] are the canonical implementations
//! for session serialization.  `app.rs` delegates to these functions; no
//! JSON serialization logic appears outside this module.

use crate::render::colormap::Colormap;
use crate::tools::interaction::Annotation;
use crate::tools::kind::ToolKind;
use crate::ui::sidebar::SidebarTab;
use crate::ViewerState;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Serializable viewer state used for save/load session workflows.
///
/// # Invariants
///
/// - `axis âˆˆ [0, 2]` â€” clamped to 2 on restore.
/// - `zoom âˆˆ [MIN_ZOOM, MAX_ZOOM]` â€” clamped on restore.
/// - `cine_fps > 0` â€” not validated here; validation is the caller's
///   responsibility on restore.
/// - `annotations` contains only complete (committed) annotations; in-progress
///   gesture state is never captured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerSessionSnapshot {
    /// Optional source path for the loaded study.
    pub source: Option<PathBuf>,
    /// Primary viewer navigation and window/level state.
    pub viewer_state: ViewerState,
    /// Active colormap.
    pub colormap: Colormap,
    /// Primary axis: 0 axial, 1 coronal, 2 sagittal.
    pub axis: usize,
    /// Active interaction tool.
    pub active_tool: ToolKind,
    /// Whether 2Ã—2 MPR layout is active.
    pub multi_planar: bool,
    /// Overlay visibility.
    pub show_overlay: bool,
    /// Crosshair visibility.
    pub show_crosshair: bool,
    /// RT-STRUCT overlay visibility.
    pub show_rt_struct_overlay: bool,
    /// RT-DOSE overlay visibility.
    #[serde(default)]
    pub show_rt_dose_overlay: bool,
    /// RT-DOSE overlay opacity.
    #[serde(default = "default_rt_dose_opacity")]
    pub rt_dose_opacity: f32,
    /// Series browser visibility.
    pub show_series_browser: bool,
    /// Sidebar tab.
    pub sidebar_tab: SidebarTab,
    /// Coronal slice index.
    pub coronal_slice: usize,
    /// Sagittal slice index.
    pub sagittal_slice: usize,
    /// Viewport pan offset in screen pixels.
    pub pan_offset: [f32; 2],
    /// Viewport zoom multiplier.
    pub zoom: f32,
    /// Cine playback enabled state.
    pub cine_enabled: bool,
    /// Cine playback target frame rate.
    pub cine_fps: f32,
    /// Completed measurement and ROI annotations.
    ///
    /// Stored in image-pixel coordinates; physical (mm) derived fields
    /// (length_mm, angle_deg, area_mm2, statistics) are stored verbatim so
    /// they survive round-trip without requiring the volume to be loaded.
    #[serde(default)]
    pub annotations: Vec<Annotation>,
}

impl ViewerSessionSnapshot {
    /// Construct an empty default snapshot.
    pub fn empty() -> Self {
        Self {
            source: None,
            viewer_state: ViewerState::new(),
            colormap: Colormap::Grayscale,
            axis: 0,
            active_tool: ToolKind::WindowLevel,
            multi_planar: false,
            show_overlay: true,
            show_crosshair: false,
            show_rt_struct_overlay: true,
            show_rt_dose_overlay: false,
            rt_dose_opacity: 0.5,
            show_series_browser: true,
            sidebar_tab: SidebarTab::Series,
            coronal_slice: 0,
            sagittal_slice: 0,
            pan_offset: [0.0, 0.0],
            zoom: 1.0,
            cine_enabled: false,
            cine_fps: 12.0,
            annotations: Vec::new(),
        }
    }
}

impl Default for ViewerSessionSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

fn default_rt_dose_opacity() -> f32 {
    0.5
}

// â”€â”€â”€ SSOT file I/O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Serialize `snapshot` as pretty-printed JSON and write to `path`.
///
/// # Errors
/// Returns an error when JSON serialization fails or `path` cannot be written.
pub fn save_to_file(snapshot: &ViewerSessionSnapshot, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(snapshot)
        .context("JSON serialization of ViewerSessionSnapshot failed")?;
    std::fs::write(path, json).with_context(|| format!("writing session file {}", path.display()))
}

/// Deserialize a [`ViewerSessionSnapshot`] from the JSON file at `path`.
///
/// # Errors
/// Returns an error when `path` cannot be read or its content is not valid
/// JSON matching the `ViewerSessionSnapshot` schema.
pub fn load_from_file(path: &Path) -> Result<ViewerSessionSnapshot> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("reading session file {}", path.display()))?;
    serde_json::from_str::<ViewerSessionSnapshot>(&json)
        .with_context(|| format!("deserializing session from {}", path.display()))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
