//! Viewer UI components.
//!
//! # Sub-modules
//!
//! | Module           | Contents                                                |
//! |------------------|---------------------------------------------------------|
//! | [`layout`]       | [`LayoutMode`] and [`ViewportId`] enumerations.        |
//! | [`viewport`]     | [`ViewportState`] and [`ViewportPanel`] — the core MPR  |
//! |                  | slice display widget.                                   |
//! | [`toolbar`]      | [`ToolbarState`] and [`ToolbarPanel`] — top toolbar.    |
//! | [`sidebar`]      | [`SidebarPanel`] — series browser + metadata tab.       |
//! | [`overlay`]      | [`OverlayRenderer`] — DICOM 4-corner text overlays.     |
//! | [`measurements`] | [`MeasurementLayer`] — annotation drawing helpers.     |
//! | [`mpr_cursor`]   | [`LinkedCursor`] and viewport/voxel transforms.         |
//! | [`cine`]         | [`CinePlayback`] — per-frame playback timing state.     |
//! | [`cursor_info`]  | [`voxel_to_lps`] — ITK affine voxel → LPS mm transform. |
//! | [`export_plan`]  | Deterministic all-axis MPR PNG export planning.          |
//! | [`rtstruct_overlay`] | RT-STRUCT patient-space contour projection.          |
//! | [`zoom`]         | Scroll-wheel zoom policy and clamped zoom mapping.       |
//! | [`window_level`] | W/L drag mapping SSOT and sensitivity constant.          |
//! | [`window_presets`] | [`WindowPreset`] with standard CT/MR presets.         |

pub mod layout;
pub mod measurements;
pub mod cine;
pub mod cursor_info;
pub mod export_plan;
pub mod mpr_cursor;
pub mod overlay;
pub mod rtstruct_overlay;
pub mod sidebar;
pub mod toolbar;
pub mod viewport;
pub mod window_level;
pub mod window_presets;
pub mod zoom;

pub use cine::CinePlayback;
pub use cursor_info::{format_lps, voxel_to_lps};
pub use export_plan::{axis_folder_name, axis_slice_total, plan_all_mpr_exports, PlannedSliceExport};
pub use layout::{LayoutMode, ViewportId};
pub use measurements::MeasurementLayer;
pub use mpr_cursor::{axis_slice_dimensions, map_view_row_col_to_voxel, viewport_point_to_voxel, LinkedCursor};
pub use overlay::OverlayRenderer;
pub use rtstruct_overlay::{project_rt_struct_contours_for_slice, ProjectedRtContour};
pub use sidebar::SidebarPanel;
pub use toolbar::{ToolbarPanel, ToolbarState};
pub use viewport::{ViewportPanel, ViewportState};
pub use window_level::{clamp_window_width, window_level_from_drag_delta, WINDOW_LEVEL_SENSITIVITY};
pub use window_presets::WindowPreset;
pub use zoom::{
	fit_view_transform, should_zoom_with_scroll, zoom_from_drag_delta, zoom_from_scroll, FIT_ZOOM,
	MAX_ZOOM, MIN_ZOOM,
};
pub use window_level::MIN_WINDOW_WIDTH;
