//! Viewer session snapshot persistence.
//!
//! A session snapshot captures presentation state only. Pixel data, DICOM
//! metadata, annotations, and filesystem caches remain outside this module.

use crate::render::colormap::Colormap;
use crate::tools::kind::ToolKind;
use crate::ui::sidebar::SidebarTab;
use crate::ViewerState;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Serializable viewer state used for save/load session workflows.
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
    /// Whether 2x2 MPR layout is active.
    pub multi_planar: bool,
    /// Overlay visibility.
    pub show_overlay: bool,
    /// Crosshair visibility.
    pub show_crosshair: bool,
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
            show_series_browser: true,
            sidebar_tab: SidebarTab::Series,
            coronal_slice: 0,
            sagittal_slice: 0,
            pan_offset: [0.0, 0.0],
            zoom: 1.0,
            cine_enabled: false,
            cine_fps: 12.0,
        }
    }
}

impl Default for ViewerSessionSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
